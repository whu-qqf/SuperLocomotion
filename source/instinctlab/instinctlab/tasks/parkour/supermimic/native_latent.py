"""Native Hiking single-model visual SuperMimic learner.

No RSSM, old-policy initialization or PPO. Native
Hiking generates commands/rewards without motion imitation.
The imagined-camera route predicts physical state, contacts and readouts,
then queries depth at predicted poses. Raw camera histories seed each rollout.
"""

import time
import torch
from dataclasses import asdict, dataclass, field

from instinctlab_supermimic.depth import DEPTH_SHAPE, LATENT_DIM, DepthAutoencoder, DepthHistory, DepthLatentPolicy
from instinctlab_supermimic.networks import Critic
from instinctlab_supermimic.physical import unflatten_state
from instinctlab_supermimic.rotations import mv, yaw_mat

from isaaclab.managers import RecorderTermCfg
from isaaclab.utils import configclass

from instinctlab.envs.mdp.observations.exteroception import visualizable_image

from .native_execution import optimizer_step
from .native_imagination import CRITIC_HISTORY_DIM, NativeImaginationStart
from .native_physics_audit import make_native_physics
from .native_policy import NativePolicyUpdateCfg, update_native_policy
from .native_statistics import NativeEpisodeStatistics
from .native_task import AMP_WIDTHS, POLICY_WIDTHS, NativeTaskHeads
from .native_training_replay import NativeTrainingReplay, history_frames, select_rows
from .native_transitions import HikingTransitionRecorder, NativeHikingRecorderCfg


def camera_image(env):
    return visualizable_image(env, data_type="distance_to_image_plane_noised")[:, 0]


def configure_without_amp(cfg):
    """Remove motion-library dependencies, keeping the critic's measured state history.

    The legacy amp_policy group is just robot proprioception. Retaining it
    preserves critic dimensions for the reward ablation; no discriminator
    consumes it when AMP is disabled.
    """
    cfg.scene.motion_reference = None
    cfg.observations.amp_reference = None
    cfg.terminations.dataset_exhausted = None
    return cfg


class LatentTransitionRecorder(HikingTransitionRecorder):
    def record_pre_step(self):
        from instinctlab.tasks.parkour.mdp.sgs import sgs_pre_step

        sgs_pre_step(self._env)
        return super().record_pre_step()

    def record_post_step(self):
        result = super().record_post_step()
        # Read after physics, before same-step reset. Native camera is updated
        # every control, including a terminal control, by its lazy data read.
        self.next_depth = camera_image(self._env).detach().clone()
        return result


@configclass
class NativeLatentRecorderCfg(NativeHikingRecorderCfg):
    transition = RecorderTermCfg(class_type=LatentTransitionRecorder)


@dataclass(frozen=True)
class NativeLatentCfg:
    use_amp: bool = False
    world_model_heightmap: bool = False
    terrain_locomotion: bool = False
    imagined_camera: bool = False
    tgi_depth: bool = False
    tgi_aligned: bool = False
    policy_microbatch: int = 128
    tracking_reference_dir: str = ""
    paired_reference_dir: str = ""
    collect_steps: int = 8
    warmup_steps: int = 48
    replay_steps: int = 128
    model_only_iterations: int = 500
    model_updates: int = 10
    model_batch: int = 32
    model_horizon: int = 8
    actor_batch: int = 128
    style_batch: int = 128
    learning_rate: float = 1e-4
    exploration_std: float = 0.2
    hidden: tuple = (2048, 2048, 2048)
    policy: NativePolicyUpdateCfg = field(default_factory=NativePolicyUpdateCfg)

    def __post_init__(self):
        if self.use_amp or self.tracking_reference_dir or self.paired_reference_dir:
            raise ValueError("This repository supports reference-free Hiking locomotion only.")
        if self.tgi_aligned and (not self.tgi_depth or self.policy_microbatch < 1):
            raise ValueError("Aligned TGI training requires terrain dynamics and a positive policy microbatch.")
        if self.tgi_depth and (not self.imagined_camera or not self.world_model_heightmap):
            raise ValueError("TGI depth requires imagined camera and privileged terrain model inputs.")
        if self.imagined_camera and not self.terrain_locomotion:
            raise ValueError("Imagined camera training requires reference-free terrain locomotion.")
        if self.terrain_locomotion and self.world_model_heightmap and not self.tgi_depth:
            raise ValueError("Terrain locomotion requires depth-only F, no AMP and no motion references.")
        if (
            min(
                self.collect_steps,
                self.warmup_steps,
                self.replay_steps,
                self.model_updates,
                self.model_batch,
                self.model_horizon,
                self.actor_batch,
                self.style_batch,
            )
            < 1
        ):
            raise ValueError("Latent training sizes must be positive.")
        if self.learning_rate <= 0 or self.exploration_std < 0 or self.model_only_iterations < 0:
            raise ValueError("Invalid latent optimizer/warmup settings.")
        if self.policy.use_failure_head:
            raise ValueError("Latent route uses physical terminations; no uncalibrated failure head.")


class NativeLatentRunner:
    def __init__(self, vec, cfg=NativeLatentCfg()):
        self.vec, self.env, self.cfg = vec, vec.unwrapped, cfg
        if bool(getattr(self.env.cfg, "terrain_locomotion", False)) != cfg.terrain_locomotion:
            raise ValueError("Environment and learner terrain locomotion modes differ.")
        self.obs, info = vec.reset()
        self.observations = info["observations"]
        env, device = self.env, self.env.device
        if abs(env.scene["camera"].cfg.update_period - env.step_dt) > 1e-8:
            raise ValueError("Latent route requires a fresh native image each control step.")
        self.recorder = env.recorder_manager._terms["transition"]
        self.physics = make_native_physics(
            env,
            self.recorder,
            z_dim=0 if cfg.tgi_depth else LATENT_DIM,
            hidden=cfg.hidden,
            model_heightmap=cfg.world_model_heightmap,
            predict_latent=not cfg.imagined_camera,
            terrain_features=cfg.tgi_depth,
            mean_contact_loss=cfg.tgi_aligned,
        )
        self.task = NativeTaskHeads(env, self.physics)
        self.perception = DepthAutoencoder().to(device)
        self.imagined_camera = None
        if cfg.imagined_camera:
            from .native_imagined_camera import NativeImaginedDepthCamera

            self.imagined_camera = NativeImaginedDepthCamera(env)
        self.reference_bank = None
        self.reference_player = None
        self.paired_library = None
        reference_dim = 0
        self.actor = DepthLatentPolicy(
            reference_dim, detach_latent=not cfg.tgi_depth, normalize_inputs=cfg.tgi_aligned
        ).to(device)
        self.history = DepthHistory(camera_image(env))
        physical_dim = self.task.critic_features(
            unflatten_state(self.recorder.state()),
            self.task.command(),
            torch.zeros(env.num_envs, 4, device=device),
            self.physics.context(env.action_manager.action),
        ).shape[-1]
        self.critic = Critic(physical_dim + CRITIC_HISTORY_DIM + LATENT_DIM + reference_dim).to(device)
        self.style = None
        self.modules = dict(
            world_model=self.physics.model,
            depth=self.perception,
            actor=self.actor,
            critic=self.critic,
        )
        # TGI gives F and depth separate ownership so policy returns can train
        # perception while F stays frozen. Older modes keep joint supervision.
        self.supervised = torch.nn.ModuleList([self.physics.model, self.perception])
        self.optimizers = dict(
            model=torch.optim.Adam(
                self.physics.model.parameters() if cfg.tgi_depth else self.supervised.parameters(), lr=cfg.learning_rate
            ),
            actor=torch.optim.Adam(
                self.actor.parameters(), lr=cfg.learning_rate, betas=(0.9, 0.95) if cfg.tgi_aligned else (0.9, 0.999)
            ),
            critic=torch.optim.Adam(self.critic.parameters(), lr=cfg.learning_rate),
        )
        if cfg.tgi_depth:
            self.optimizers["depth"] = torch.optim.Adam(self.perception.parameters(), lr=cfg.learning_rate)
        self.update_counts = dict.fromkeys(self.optimizers, 0)
        for name, optimizer in self.optimizers.items():
            optimizer.register_step_post_hook(lambda opt, args, kwargs, key=name: self._count(key))
        self.replay = NativeTrainingReplay(env.num_envs, cfg.replay_steps)
        self.episodes = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        self.episode_statistics = NativeEpisodeStatistics(env.num_envs, device)
        # Keep a fixed environment split for model validation, never used in fitting.
        self.holdout_envs = tuple(range(7, env.num_envs, 8))
        self.train_envs = tuple(i for i in range(env.num_envs) if i not in self.holdout_envs)
        self.iteration = 0
        self.contract = dict(
            architecture="native_hiking_single_wm_depth_latent_v1",
            physical=self.physics.checkpoint_contract(),
            depth_shape=DEPTH_SHAPE,
            depth_stride=5,
            control_dt=env.step_dt,
            terrain_locomotion=cfg.terrain_locomotion,
            camera=env.scene["camera"].cfg.to_dict(),
            critic_layout="physical_proprio8_amp10_z32",
            amp_reference_manifest_sha256=getattr(env.cfg, "lafan_reference_manifest_sha256", None),
        )
        if getattr(env.cfg, "sgs_curriculum", None):
            self.contract["terrain_curriculum"] = env.cfg.sgs_curriculum
        if cfg.imagined_camera:
            self.contract["architecture"] = "native_hiking_physics_imagined_depth_v1"
            self.contract["visual_feedback"] = "detached_fk_raycast_full_history_v1"
            self.contract["reward_weights"] = dict(
                native={name: term.weight for name, term in self.task.reward_cfg.items()},
                locomotion=env.cfg.locomotion_reward_weights,
            )
        if cfg.tgi_depth:
            self.contract["architecture"] = "native_hiking_tgi_terrain_depth_actor_v1"
            self.contract["model_terrain"] = "base35_shoe15_per_foot_height_relative_v1"
            self.contract["encoder_learning"] = "real_reconstruction_and_imagined_policy_return_v1"
        if cfg.tgi_aligned:
            self.contract["architecture"] = "native_hiking_tgi_aligned_depth_v2"
            self.contract["policy_recipe"] = "global_advantage_lr01_sym2_imag_noise02_normalized_v1"
            self.contract["contact_loss"] = "mean_channels_weight05_v1"
            self.contract["reward_recipe"] = "tgi_core_native_root_shoe_geometry_v3_center_clearance_pose_std"
        if cfg.terrain_locomotion:
            self.contract["locomotion_reward"] = dict(
                version="sole_gap_and_local_top_support_v2",
                support_window_m=0.5,
                support_hole_depth_m=0.25,
                support_sigma_m=0.05,
                terrain_forward="nearest",
                terrain_backward="bilinear",
                sole_points="native_shoe_four_corners_v1",
            )

    def _count(self, name):
        self.update_counts[name] += 1

    @torch.no_grad()
    def collect(self, steps=None):
        count = (self.cfg.collect_steps if self.replay.controls else self.cfg.warmup_steps) if steps is None else steps
        reference_metrics = {}
        locomotion_metrics = {}
        for _ in range(count):
            depth = self.history.stack()
            depth_history = self.history.images.half().cpu().clone() if self.cfg.imagined_camera else None
            context = self.physics.context(self.env.action_manager.action)
            sensor = self.env.scene["contact_forces"].data
            ids = self.task.foot_contact_ids
            start = dict(
                state=self.recorder.state(),
                previous_action=self.env.action_manager.action.clone(),
                timers=torch.cat(
                    (sensor.current_air_time.torch[:, ids], sensor.current_contact_time.torch[:, ids]), -1
                ),
                physical_context=context,
                task_context=self.task.context(),
                command=self.task.command(),
                sensor_context=self.task.sensor_context(),
                policy_history=history_frames(self.obs[:, :768], POLICY_WIDTHS, 8),
                amp_history=history_frames(self.observations["amp_policy"], AMP_WIDTHS, 10),
                control_step=self.env.episode_length_buf.long().clone(),
            )
            reference = None
            latent = self.perception.encode(depth)
            if self.cfg.tgi_aligned:
                self.actor.proprio_norm.update(self.obs[:, :768])
                self.actor.latent_norm.update(latent)
            action = self.actor(self.obs[:, :768], latent, reference)
            action = (action + self.cfg.exploration_std * torch.randn_like(action)).clamp(-20, 20)
            self.obs, reward, done, info = self.vec.step(action)
            transition = self.recorder.take_transition()
            # The vec wrapper packs reward groups as N x G. Use the scalar
            # task group before adding N-valued terrain terms; broadcasting
            # N x 1 plus N would otherwise create an N x N reward matrix.
            reward = transition.rewards["rewards"] if isinstance(transition.rewards, dict) else transition.rewards
            if self.cfg.tgi_aligned:
                from .native_rewards import tgi_core_reward

                reward, _ = tgi_core_reward(
                    self.task,
                    unflatten_state(transition.next_state),
                    action,
                    start["previous_action"],
                    start["command"]["velocity"],
                    start["task_context"],
                )
            locomotion_reward = torch.zeros_like(reward)
            if self.cfg.terrain_locomotion:
                extra_terms, _ = self.task.locomotion_readout(
                    unflatten_state(transition.next_state),
                    start["task_context"],
                    self.physics.contact_targets(transition)[:, -2:],
                    start["command"]["velocity"],
                )
                locomotion_reward = sum(extra_terms.values())
                reward = reward + locomotion_reward
                for name, value in extra_terms.items():
                    locomotion_metrics[name] = locomotion_metrics.get(name, 0.0) + float(value.mean()) / count
            torch.testing.assert_close(transition.action, action, rtol=0, atol=0)
            torch.testing.assert_close(done.bool(), transition.terminated | transition.truncated, rtol=0, atol=0)
            next_depth = self.history.advance(self.recorder.next_depth)
            pool = torch.tensor(self.train_envs, device=self.env.device)
            self.physics.observe(type(transition)(**select_rows(vars(transition), pool)), context[pool])
            self.observations = info["observations"]
            learning = dict(start=start)
            self.replay.add_transition(
                transition,
                context,
                self.episodes,
                self.physics.contact_targets(transition),
                self.physics.auxiliary_targets(transition),
                learning=learning,
                task_reward=(
                    reward
                    if self.cfg.tgi_aligned
                    else (
                        (transition.rewards["rewards"] if isinstance(transition.rewards, dict) else transition.rewards)
                        + locomotion_reward
                    )
                ),
            )
            # Store pixels, never a persistent encoder output. Half storage is
            # sufficient for normalized native depth; fitting converts to float.
            self.replay.rows[-1].update(depth=depth.half().cpu(), next_depth=next_depth.half().cpu())
            if self.cfg.imagined_camera:
                self.replay.rows[-1].update(depth_history=depth_history, source_env=torch.arange(self.env.num_envs))
            self.episode_statistics.observe(
                reward,
                done,
                info,
                {
                    name: self.env.termination_manager.get_term(name)
                    for name in self.env.termination_manager.active_terms
                },
            )
            reset_ids = done.nonzero(as_tuple=True)[0]
            if len(reset_ids):
                self.episodes[reset_ids] += 1
                self.history.reset(reset_ids, camera_image(self.env))
        return dict(
            collected_controls=count,
            real_controls=self.replay.controls,
            episode=self.episode_statistics.summary(),
            reference=reference_metrics,
            locomotion=locomotion_metrics,
        )

    def model_batch(self, pool, horizon=None, generator=None):
        return self.replay.sample_physics(
            self.cfg.model_batch,
            horizon or self.cfg.model_horizon,
            pool,
            self.env.device,
            generator=generator,
            extra_fields=("depth", "next_depth")
            + (
                ("depth_history", "source_env")
                if self.cfg.imagined_camera and (not self.cfg.tgi_aligned or horizon is not None)
                else ()
            ),
        )

    def model_loss(self, batch, update_statistics=True):
        if getattr(getattr(self, "cfg", None), "tgi_depth", False):
            mask = batch["mask"]
            physical = {
                k: v for k, v in batch.items() if k not in ("depth", "next_depth", "depth_history", "source_env")
            }
            loss, metrics = self.physics.window_loss(**physical)
            reconstruction = self.perception.reconstruction_loss(
                torch.cat((batch["depth"][mask], batch["next_depth"][mask]))
            )
            factor = len(mask) / mask.sum()
            metrics = {name: value * factor for name, value in metrics.items() if name != "valid"}
            metrics["reconstruction_l1"] = reconstruction.detach()
            return loss * factor + reconstruction, metrics
        if getattr(self, "imagined_camera", None) is not None:
            return self.imagined_model_loss(batch, update_statistics)
        images, following = batch["depth"], batch["next_depth"]
        mask = batch["mask"]
        # Encode valid rows only; invalid padding is allowed to contain NaNs.
        with torch.no_grad():
            z = images.new_zeros((*mask.shape, LATENT_DIM), dtype=torch.float32)
            z1 = torch.zeros_like(z)
            z[mask] = self.perception.encode(images[mask])
            z1[mask] = self.perception.encode(following[mask])
            if update_statistics:
                # CNN latents can vary by only 1e-3 per control. Starting
                # their EMA at unit variance suppresses the latent objective
                # while allowing disproportionately large residual steps.
                rate = 1.0 if self.update_counts["model"] == 0 else 0.01
                self.physics.model.update_z_stats(z, z1 - z, mask, rate=rate)
        physical_batch = {k: v for k, v in batch.items() if k not in ("depth", "next_depth")}
        loss, metrics = self.physics.window_loss(**physical_batch, latent=z, next_latent=z1)
        reconstruction = self.perception.reconstruction_loss(torch.cat((images[mask], following[mask])))
        # F's multi-step loss is reported per valid transition, as is reconstruction.
        loss = loss * len(mask) / mask.sum() + reconstruction
        metrics = {name: value * len(mask) / mask.sum() for name, value in metrics.items() if name != "valid"}
        metrics["reconstruction_l1"] = reconstruction.detach()
        return loss, metrics

    def imagined_model_loss(self, batch, update_statistics=True):
        from .native_imagined_camera import ImaginedDepthSequence

        mask = batch["mask"]
        if update_statistics:
            with torch.no_grad():
                z = batch["depth"].new_zeros((*mask.shape, LATENT_DIM), dtype=torch.float32)
                z[mask] = self.perception.encode(batch["depth"][mask])
                rate = 1.0 if self.update_counts["model"] == 0 else 0.01
                self.physics.model.update_z_stats(z, None, mask, rate=rate)
        sequence = ImaginedDepthSequence(
            self.imagined_camera, self.perception, batch["depth_history"][:, 0], batch["source_env"][:, 0]
        )
        physical = {k: v for k, v in batch.items() if k not in ("depth", "next_depth", "depth_history", "source_env")}
        loss, metrics = self.physics.window_loss(**physical, latent_observation=sequence)
        reconstruction = self.perception.reconstruction_loss(
            torch.cat((batch["depth"][mask], batch["next_depth"][mask]))
        )
        loss = loss * len(mask) / mask.sum() + reconstruction
        metrics = {name: value * len(mask) / mask.sum() for name, value in metrics.items() if name != "valid"}
        metrics["reconstruction_l1"] = reconstruction.detach()
        return loss, metrics

    @torch.no_grad()
    def sample_start(self):
        rows = list(self.replay.rows)
        allowed = torch.stack([row["learning"]["start"]["command"]["time_left"] for row in rows])
        allowed = allowed > self.cfg.policy.horizon * self.env.step_dt
        allowed[:, list(self.holdout_envs)] = False
        candidates = allowed.nonzero().tolist()
        if not candidates:
            raise RuntimeError("No actor starts with a complete command horizon; collect more data.")
        from .native_training_replay import stack_rows

        picks = torch.randint(len(candidates), (self.cfg.actor_batch,)).tolist()
        values = [select_rows(rows[candidates[i][0]]["learning"]["start"], candidates[i][1]) for i in picks]
        start = stack_rows(values, self.env.device)
        images = torch.stack([rows[candidates[i][0]]["depth"][candidates[i][1]] for i in picks]).to(self.env.device)
        extra = {}
        if self.cfg.imagined_camera:
            extra = dict(
                depth_history=torch.stack(
                    [rows[candidates[i][0]]["depth_history"][candidates[i][1]] for i in picks]
                ).to(self.env.device),
                source_env=torch.tensor([candidates[i][1] for i in picks], device=self.env.device),
            )
        return NativeImaginationStart(
            **start, perception_state=None, visual_latent=self.perception.encode(images), **extra
        )

    @torch.no_grad()
    def evaluate_models(self):
        """Autoregressive 1/8/16/32-step errors, compared with unchanged z.

        Partial episodes are masked, so valid sample counts must accompany
        errors. These are model errors, not full-episode policy success.
        """
        reports = {}
        generator = torch.Generator().manual_seed(self.iteration)
        for name, pool in (("train", self.train_envs), ("holdout", self.holdout_envs)):
            if not pool:
                continue
            batch = self.model_batch(pool, horizon=32, generator=generator)
            initial_z = self.perception.encode(batch["depth"][:, 0])
            latent = initial_z.clone()
            sequence = None
            if self.cfg.imagined_camera:
                from .native_imagined_camera import ImaginedDepthSequence

                sequence = ImaginedDepthSequence(
                    self.imagined_camera, self.perception, batch["depth_history"][:, 0], batch["source_env"][:, 0]
                )
            current = unflatten_state(batch["state"][:, 0])
            reports[name] = {}
            for index in range(32):
                active = batch["mask"][:, index]
                if not active.any():
                    break
                selected = tuple(value[active] for value in current)
                predicted, contact_logits, next_z = self.physics.model.step(
                    selected,
                    batch["actions"][active, index],
                    batch["context"][active, index],
                    self.physics.model_scan(selected),
                    latent[active],
                )
                ids = active.nonzero(as_tuple=True)[0]
                current = tuple(old.index_copy(0, ids, new) for old, new in zip(current, predicted))
                if sequence is not None:
                    next_z = sequence(predicted, active, index + 1)
                latent = latent.index_copy(0, ids, next_z)
                if index + 1 in (1, 8, 16, 32):
                    target_z = self.perception.encode(batch["next_depth"][active, index])
                    target_state = unflatten_state(batch["next_state"][active, index])
                    physical_error = self.physics.model.state_error(predicted, target_state)
                    feet_error = self.physics.foot_points(predicted) - self.physics.foot_points(target_state)
                    contact_probability = contact_logits[:, -2:].sigmoid()
                    target_contact = batch["contacts"][active, index, -2:]
                    stance = target_contact.bool()
                    predicted_contact = contact_probability >= 0.5
                    tp = (predicted_contact & stance).sum()
                    fp = (predicted_contact & ~stance).sum()
                    fn = (~predicted_contact & stance).sum()
                    foot_mse = feet_error.square().mean((-2, -1))
                    modes = dict(stance=stance, swing=~stance)
                    if index:
                        previous_contact = batch["contacts"][active, index - 1, -2:].bool()
                        modes["touchdown"] = stance & ~previous_contact
                    strata = {
                        mode: dict(
                            foot_samples=int(selected.sum()),
                            feet_rmse_m=float(foot_mse[selected].mean().sqrt()) if selected.any() else None,
                        )
                        for mode, selected in modes.items()
                    }
                    reports[name][str(index + 1)] = dict(
                        valid=int(active.sum()),
                        root_rmse_m=float(physical_error[:, :3].square().mean().sqrt()),
                        feet_rmse_m=float(feet_error.square().mean().sqrt()),
                        joint_rmse_rad=float(physical_error[:, 12:41].square().mean().sqrt()),
                        contact_brier=float((contact_probability - target_contact).square().mean()),
                        measured_contact_fraction=float(target_contact.mean()),
                        predicted_contact_fraction=float(predicted_contact.float().mean()),
                        contact_true_positive=int(tp),
                        contact_false_positive=int(fp),
                        contact_false_negative=int(fn),
                        latent_rmse=float((next_z - target_z).square().mean().sqrt()),
                        constant_latent_rmse=float((initial_z[active] - target_z).square().mean().sqrt()),
                        contact_strata=strata,
                    )
        return reports

    def update(self):
        before = self.update_counts.copy()
        sampling_seconds = model_seconds = 0.0
        for _ in range(self.cfg.model_updates):
            started = time.perf_counter()
            batch = self.model_batch(self.train_envs)
            sampled = time.perf_counter()
            sampling_seconds += sampled - started
            loss, metrics = self.model_loss(batch)
            if self.cfg.tgi_depth:
                self.optimizers["depth"].zero_grad(set_to_none=True)
            gradient = optimizer_step(loss, self.supervised, self.optimizers["model"])
            if self.cfg.tgi_depth:
                self.optimizers["depth"].step()
            model_seconds += time.perf_counter() - sampled
        policy_started = time.perf_counter()
        style = {}
        enabled = self.iteration >= self.cfg.model_only_iterations
        policy = None
        if enabled:
            start = self.sample_start()
            for module in self.modules.values():
                module.zero_grad(set_to_none=True)
            if self.cfg.tgi_aligned:
                from .native_tgi_policy import update_tgi_depth_policy

                policy = update_tgi_depth_policy(
                    self.task,
                    self.actor,
                    self.perception,
                    self.critic,
                    start,
                    self.optimizers["actor"],
                    self.optimizers["depth"],
                    self.optimizers["critic"],
                    self.cfg.policy,
                    self.imagined_camera,
                    microbatch=self.cfg.policy_microbatch,
                )
            else:
                policy = update_native_policy(
                    self.task,
                    self.actor,
                    self.perception,
                    self.critic,
                    self.style,
                    start,
                    self.optimizers["actor"],
                    self.optimizers["critic"],
                    self.cfg.policy,
                    reference_bank=self.reference_bank,
                    paired_library=self.paired_library,
                    imagined_camera=self.imagined_camera,
                    depth_optimizer=self.optimizers.get("depth"),
                )
            for name in ("world_model", "depth", "style"):
                if name == "depth" and self.cfg.tgi_depth:
                    continue
                if name in self.modules and any(p.grad is not None for p in self.modules[name].parameters()):
                    raise RuntimeError(f"Policy gradient leaked into {name} parameters.")
        expected = dict(
            model=self.cfg.model_updates,
            actor=int(enabled),
            critic=self.cfg.policy.critic_epochs * self.cfg.policy.critic_minibatches if enabled else 0,
        )
        if self.cfg.tgi_depth:
            expected["depth"] = self.cfg.model_updates + int(enabled)
        actual = {name: self.update_counts[name] - before[name] for name in before}
        if actual != expected:
            raise RuntimeError(f"Latent optimizer updates differ: {actual} != {expected}.")
        self.iteration += 1
        return dict(
            iteration=self.iteration,
            model_only=not enabled,
            amp_enabled=self.style is not None,
            model={k: float(v) for k, v in metrics.items()},
            model_grad_norm=gradient,
            style={k: float(v) for k, v in style.items()},
            policy=policy,
            actual_updates=actual,
            update_timing=dict(
                replay_sampling_seconds=sampling_seconds,
                model_training_seconds=model_seconds,
                policy_and_style_seconds=time.perf_counter() - policy_started,
            ),
        )

    def save(self, path, include_replay=True):
        from pathlib import Path

        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        sgs = getattr(self.env, "_hiking_sgs", None)
        torch.save(
            dict(
                format=self.contract["architecture"],
                cfg=asdict(self.cfg),
                contract=self.contract,
                iteration=self.iteration,
                modules={name: module.state_dict() for name, module in self.modules.items()},
                optimizers={name: opt.state_dict() for name, opt in self.optimizers.items()},
                update_counts=self.update_counts,
                replay=self.replay.state_dict() if include_replay else None,
                reference_bank=None,
                sgs=sgs["sampler"].state_dict() if sgs is not None else None,
                execution=getattr(self, "execution", {}),
            ),
            temporary,
        )
        temporary.replace(path)

    def load(self, path):
        """Resume learned modules/replay; the simulator starts a fresh collection segment."""
        if self.iteration or self.replay.controls:
            raise RuntimeError("Load before collecting in the fresh simulator.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["format"] != self.contract["architecture"]:
            raise ValueError("Only this latent architecture can be resumed.")
        if checkpoint["replay"] is None:
            raise ValueError("Weights-only evaluation snapshot; resume from latest.pt with replay.")
        saved_cfg = dict(checkpoint["cfg"])
        # Checkpoints predating the switch always trained with AMP.
        saved_cfg.setdefault("use_amp", True)
        saved_cfg.setdefault("tracking_reference_dir", "")
        saved_cfg.setdefault("paired_reference_dir", "")
        saved_cfg.setdefault("world_model_heightmap", True)
        saved_cfg.setdefault("imagined_camera", False)
        saved_cfg.setdefault("tgi_depth", False)
        saved_cfg.setdefault("tgi_aligned", False)
        saved_cfg.setdefault("policy_microbatch", 128)
        if saved_cfg != asdict(self.cfg) or checkpoint["contract"] != self.contract:
            raise ValueError("Latent checkpoint configuration or physical/camera contract differs.")
        for name, module in self.modules.items():
            module.load_state_dict(checkpoint["modules"][name])
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][name])
        self.replay.load_state_dict(checkpoint["replay"])
        if checkpoint.get("sgs") is not None:
            sgs = getattr(self.env, "_hiking_sgs", None)
            if sgs is None:
                raise ValueError("Checkpoint contains SGS state but the environment has no SGS curriculum.")
            sgs["sampler"].load_state_dict(checkpoint["sgs"])
            # Simulator episodes restart on resume. Redraw from restored history
            # and do not count the fresh zero-length episodes as outcomes.
            self.obs, info = self.vec.reset()
            self.observations = info["observations"]
            self.history = DepthHistory(camera_image(self.env))
        self.replay.begin_collection_segment()
        self.episodes = torch.stack([r["episode_id"] for r in self.replay.rows]).amax(0).to(self.env.device) + 1
        self.iteration = checkpoint["iteration"]
        self.update_counts.update(checkpoint["update_counts"])

    def initialize_from(self, path):
        """Transfer learned modules to a new terrain, without replay or optimizer state."""
        if not self.cfg.imagined_camera or self.iteration or self.replay.controls:
            raise ValueError("Terrain transfer requires a fresh imagined-camera runner.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["format"] != "native_hiking_physics_imagined_depth_v1":
            raise ValueError("Terrain transfer requires the physical-only imagined-depth architecture.")
        source, target = dict(checkpoint["contract"]), dict(self.contract)
        source["physical"], target["physical"] = dict(source["physical"]), dict(target["physical"])
        source["physical"].pop("terrain_sha256")
        target["physical"].pop("terrain_sha256")
        if source != target:
            raise ValueError("Terrain transfer may change geometry only; robot, camera and rewards must match.")
        for name, module in self.modules.items():
            module.load_state_dict(checkpoint["modules"][name])

    def save_controller(self, path):
        torch.save(
            dict(
                format=(
                    "hiking_depth_latent_controller_v2" if self.cfg.tgi_aligned else "hiking_depth_latent_controller_v1"
                ),
                depth=self.perception.encoder.state_dict(),
                actor=self.actor.state_dict(),
                depth_shape=DEPTH_SHAPE,
                depth_stride=5,
                proprio_layout="native_term_major_8x96",
                joint_names=self.recorder.joint_names,
                control_dt=self.env.step_dt,
                reference_bank=None,
                paired_reference=None,
            ),
            path,
        )
