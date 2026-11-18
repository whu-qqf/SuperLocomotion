"""Native Hiking collection and six disjoint dual-model optimizer updates.

This runner owns the real-data boundary. P receives camera/proprio/actions;
F receives physical states/actions/context/terrain. Neither trains the other.
"""

import torch
from dataclasses import asdict, dataclass, field, replace
from math import isfinite

from instinctlab_perception.checkpoint import save_hiking_controller
from instinctlab_perception.events import PerceptionRuntime
from instinctlab_perception.policy import make_hiking_policy
from instinctlab_perception.replay import PerceptionReplay, collate_ticks, reconstruct_prefixes
from instinctlab_perception.rssm import PerceptionModel
from instinctlab_supermimic.networks import Critic, FailureHead
from instinctlab_supermimic.physical import unflatten_state
from instinctlab_supermimic.style import StyleReward

from .native_imagination import CRITIC_HISTORY_DIM, CRITIC_LAYOUT
from .native_perception import observable_proprio
from .native_physics_audit import make_native_physics
from .native_policy import NativePolicyUpdateCfg, update_native_policy
from .native_statistics import NativeEpisodeStatistics
from .native_task import NativeTaskHeads
from .native_training_replay import NativeTrainingReplay, select_rows
from .native_value import real_value_batch, real_value_loss

MODEL_EVALUATION_INTERVAL = 50


@dataclass(frozen=True)
class NativeRunnerCfg:
    collect_steps: int | None = None
    warmup_steps: int = 64
    updates_per_control: float = 0.4
    physical_replay_ratio: float | None = None
    model_only_iterations: int = 500
    physical_updates: int = 10
    style_updates: int = 2
    replay_steps: int = 512
    perception_batch: int = 16
    perception_sequence: int = 16
    physical_batch: int = 128
    physical_horizon: int = 8
    physical_foot_weight: float = 0.0
    actor_batch: int = 128
    real_batch: int = 256
    real_value_updates: int = 0
    real_value_horizon: int = 30
    real_value_batch: int = 128
    learning_rate: float = 1e-4
    exploration_std: float = 0.2
    policy: NativePolicyUpdateCfg = field(default_factory=lambda: NativePolicyUpdateCfg(use_failure_head=True))

    def __post_init__(self):
        if self.physical_replay_ratio is not None:
            if not isfinite(self.physical_replay_ratio) or self.physical_replay_ratio <= 0:
                raise ValueError("Physical replay ratio must be finite and positive.")
            if self.collect_steps is not None:
                raise ValueError("Physical replay ratio and fixed collection steps are mutually exclusive.")
        if self.real_value_updates < 0 or min(self.real_value_horizon, self.real_value_batch) < 1:
            raise ValueError("Real value sizes must be positive and update count nonnegative.")
        if self.real_value_updates and self.real_value_horizon % 5:
            raise ValueError("Real value horizons must end on a five-control perception tick.")
        if self.policy.critic_epochs == 0:
            if not self.real_value_updates:
                raise ValueError("Disabling imagined critic fitting requires measured critic updates.")
            if self.policy.horizon % 5:
                raise ValueError("Tick-only measured critic fitting requires a tick-aligned actor endpoint.")

    def update_frequency(self, num_envs):
        """Runner iterations per vector control; ratio counts sampled F transitions.

        Like ACP, this is a nominal reuse budget, not the number of distinct
        replay transitions or the number of other model/actor optimizer steps.
        """
        if self.physical_replay_ratio is None:
            return self.updates_per_control
        if min(num_envs, self.physical_updates, self.physical_batch, self.physical_horizon) < 1:
            raise ValueError("Physical replay scheduling sizes must be positive.")
        return (
            num_envs
            * self.physical_replay_ratio
            / (self.physical_updates * self.physical_batch * self.physical_horizon)
        )


def optimizer_step(loss, module, optimizer):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite native supervised gradient; optimizer not stepped.")
    optimizer.step()
    return float(norm)


def native_failure_loss(true_logits, predicted_logits, terminated):
    """Uniform real samples need unweighted BCE to estimate failure probability.

    Positive-class weighting changes the optimum odds. The actor consumes the
    sigmoid as an actual survival probability, so a cost-sensitive classifier
    would systematically shorten its imagined horizon even after convergence.
    """
    labels = terminated.float()
    return sum(
        torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        for logits in (true_logits, predicted_logits)
    )


class NativeDualRunner:
    def __init__(self, vec, cfg=NativeRunnerCfg()):
        self.vec, self.env, self.cfg = vec, vec.unwrapped, cfg
        if (
            min(
                cfg.warmup_steps,
                cfg.physical_updates,
                cfg.style_updates,
                cfg.perception_batch,
                cfg.perception_sequence,
                cfg.physical_batch,
                cfg.physical_horizon,
                cfg.actor_batch,
                cfg.real_batch,
            )
            < 1
            or cfg.exploration_std < 0
            or cfg.physical_foot_weight < 0
            or cfg.model_only_iterations < 0
            or cfg.updates_per_control <= 0
            or (cfg.collect_steps is not None and cfg.collect_steps < 1)
        ):
            raise ValueError("Native runner sizes must be positive and exploration nonnegative.")
        self.obs, info = vec.reset()
        self.observations = info["observations"]
        env, device = self.env, self.env.device
        self.recorder = env.recorder_manager._terms["transition"]
        self.physics = make_native_physics(env, self.recorder)
        self.physics.foot_loss_weight = cfg.physical_foot_weight
        self.task = NativeTaskHeads(env, self.physics)
        self.perception = PerceptionModel().to(device)
        self.actor = make_hiking_policy(self.perception.cfg.feature_dim).to(device)
        self.runtime = PerceptionRuntime(self.perception, env.num_envs, env.step_dt)
        self.perception_replay = PerceptionReplay(env.num_envs)
        self.replay = NativeTrainingReplay(env.num_envs, cfg.replay_steps)
        self.seen = torch.full((env.num_envs,), -1, dtype=torch.long, device=device)
        self.camera = env.scene["camera"]
        with torch.no_grad():
            physical_dim = self.task.critic_features(
                unflatten_state(self.recorder.state()),
                self.task.command(),
                torch.zeros(env.num_envs, 4, device=device),
                self.physics.context(env.action_manager.action),
            ).shape[-1]
        pcfg = self.perception.cfg
        self.critic = Critic(
            physical_dim + CRITIC_HISTORY_DIM + pcfg.feature_dim + pcfg.action_steps * 29 + pcfg.action_steps
        ).to(device)
        self.failure = FailureHead(physical_dim).to(device)
        # Same native Hiking AMP architecture, losses and coefficients. The
        # shared style module has no dependency on the archived MJ checkout.
        self.style = StyleReward(
            vec.get_obs_segments("amp_policy"),
            device,
            dict(hidden_sizes=[1024, 512], nonlinearity="ReLU"),
            optimizer_kwargs=dict(lr=1e-4, betas=[0.9, 0.999]),
            reward_coef=0.25,
            gradient_penalty_coef=5.0,
            weight_decay_coef=3e-4,
            logit_weight_decay_coef=0.04,
        )
        self.modules = dict(
            perception=self.perception,
            physical=self.physics.model,
            policy=self.actor,
            critic=self.critic,
            failure=self.failure,
            style=self.style.discriminator,
        )
        self.optimizers = {
            name: torch.optim.Adam(module.parameters(), lr=cfg.learning_rate)
            for name, module in self.modules.items()
            if name != "style"
        }
        self.optimizers["style"] = self.style.optimizer
        self.update_counts = dict.fromkeys(self.modules, 0)
        parameter_ids = [
            {id(p) for group in opt.param_groups for p in group["params"]} for opt in self.optimizers.values()
        ]
        if sum(map(len, parameter_ids)) != len(set.union(*parameter_ids)):
            raise ValueError("Native optimizers share parameters.")
        for name, opt in self.optimizers.items():
            opt.register_step_post_hook(lambda optimizer, args, kwargs, key=name: self._count(key))
        self.iteration = 0
        self.episode_statistics = NativeEpisodeStatistics(env.num_envs, device)
        self.contract = dict(
            critic_layout=CRITIC_LAYOUT,
            num_envs=env.num_envs,
            control_dt=env.step_dt,
            physics_dt=env.physics_dt,
            physical=self.physics.checkpoint_contract(),
            perception_cfg=asdict(self.perception.cfg),
        )
        if hasattr(env.cfg, "native_tracking_reward_version"):
            self.contract["task_rewards"] = env.cfg.native_tracking_reward_version

    def _count(self, name):
        self.update_counts[name] += 1

    def collection_steps(self):
        if self.replay.controls == 0:
            return self.cfg.warmup_steps
        if self.cfg.collect_steps is not None:
            return self.cfg.collect_steps
        # Cumulative integer division survives resume without an accumulator.
        # Higher update frequencies may intentionally collect zero controls.
        frequency = self.cfg.update_frequency(self.replay.num_envs)
        return int((self.iteration + 1) / frequency + 1e-9) - int(self.iteration / frequency + 1e-9)

    @torch.no_grad()
    def collect(self, steps=None):
        resets = 0
        tracking_sums = {}
        count = self.collection_steps() if steps is None else steps
        if count < 0:
            raise ValueError("Native collection count cannot be negative.")
        for _ in range(count):
            torch.testing.assert_close(self.runtime.control_step, self.env.episode_length_buf.long(), rtol=0, atol=0)
            generation = self.camera.frame_generation
            belief, tick = self.runtime.before_action(
                self.obs[:, 768:].view(-1, 1, 18, 32), observable_proprio(self.obs), generation != self.seen
            )
            self.seen.copy_(generation)
            if tick is not None:
                self.perception_replay.add(tick)
            context = self.physics.context(self.env.action_manager.action)
            self.replay.add_starts(tick, self.task, context, self.obs[:, :768], self.observations["amp_policy"])
            command = self.task.command()
            root_com_b = self.task.robot.data.body_com_pos_b.torch[:, self.task.root_id].clone()
            episode = self.runtime.episode_id.clone()
            action = self.actor(self.obs[:, :768], belief)
            action = (action + self.cfg.exploration_std * torch.randn_like(action)).clamp(-20, 20)
            self.obs, rewards, done, info = self.vec.step(action)
            self.episode_statistics.observe(
                rewards,
                done,
                info,
                {
                    name: self.env.termination_manager.get_term(name)
                    for name in self.env.termination_manager.active_terms
                },
            )
            self.observations = info["observations"]
            transition = self.recorder.take_transition()
            torch.testing.assert_close(transition.action, action, rtol=0, atol=0)
            torch.testing.assert_close(done.bool(), transition.terminated | transition.truncated, rtol=0, atol=0)
            train_ids = torch.tensor(self.perception_replay.train_envs, device=self.env.device)
            self.physics.observe(replace(transition, **select_rows(vars(transition), train_ids)), context[train_ids])
            ids = self.task.foot_contact_ids
            timers = torch.cat((transition.air_time[:, ids], transition.contact_time[:, ids]), -1)
            next_context = torch.cat((action, context[:, 29:]), -1)
            # Returned AMP windows are actual manager observations, including
            # native history fills on resets. Failure targets use pre-reset F
            # targets and timers, independent of those returned observations.
            learning = dict(
                policy_window=self.observations["amp_policy"],
                reference_window=self.observations["amp_reference"],
                next_timers=timers,
                command=command,
                next_features=self.task.critic_features(
                    unflatten_state(transition.next_state), command, timers, next_context
                ),
            )
            self.replay.add_transition(
                transition,
                context,
                episode,
                self.physics.contact_targets(transition),
                self.physics.auxiliary_targets(transition),
                learning,
                task_reward=(
                    transition.rewards["rewards"] if isinstance(transition.rewards, dict) else transition.rewards
                ),
            )
            state = unflatten_state(transition.next_state)
            rotation, velocity, angular = state[1:4]
            shift = (rotation @ root_com_b[..., None]).squeeze(-1)
            measured = (
                rotation.transpose(-1, -2) @ (velocity + torch.linalg.cross(angular, shift))[..., None]
            ).squeeze(-1)
            measured_angular = (rotation.transpose(-1, -2) @ angular[..., None]).squeeze(-1)
            target = command["target"][:, :2]
            progress = (target - transition.state[:, :2]).norm(dim=-1) - (target - transition.next_state[:, :2]).norm(
                dim=-1
            )
            tracking = dict(
                velocity_error_xy=float((measured[:, :2] - command["velocity"][:, :2]).norm(dim=-1).mean()),
                velocity_error_yaw=float((measured_angular[:, 2] - command["velocity"][:, 2]).abs().mean()),
                speed_xy=float(measured[:, :2].norm(dim=-1).mean()),
                single_stance_fraction=float((transition.contact_time[:, ids].gt(0).sum(-1) == 1).float().mean()),
                goal_progress_m_per_s=float(progress.mean() / self.env.step_dt),
                commanded_speed_xy=float(command["velocity"][:, :2].norm(dim=-1).mean()),
            )
            for name, value in tracking.items():
                tracking_sums[name] = tracking_sums.get(name, 0.0) + value
            self.runtime.after_action(transition.action)
            reset_ids = done.nonzero(as_tuple=True)[0]
            if len(reset_ids):
                self.perception_replay.finish(reset_ids.cpu(), episode[reset_ids].cpu())
                self.runtime.reset(reset_ids)
                self.seen[reset_ids] = -1
                resets += len(reset_ids)
            if self.replay.controls % 16 == 0:
                print("NATIVE_DUAL_COLLECT", self.replay.controls, flush=True)
        return dict(
            resets=resets,
            collected_controls=count,
            real_controls=self.replay.controls,
            real_tracking={name: value / count for name, value in tracking_sums.items()},
            episode=self.episode_statistics.summary(),
        )

    @torch.no_grad()
    def refresh_runtime(self):
        """Rebuild the last assimilated posterior; keep pending real actions."""
        ids = (~self.runtime.first).nonzero(as_tuple=True)[0]
        prefixes = []
        for env in ids.tolist():
            episode = int(self.runtime.episode_id[env])
            last_tick = int(self.runtime.control_step[env] - self.runtime.count[env])
            prefixes.append(self.perception_replay.prefix(env, episode, last_tick))
        if len(ids):
            state = reconstruct_prefixes(self.perception, prefixes)
            for name in ("deter", "stoch", "logits"):
                getattr(self.runtime.state, name)[ids] = getattr(state, name)
        reset_ids = self.runtime.first.nonzero(as_tuple=True)[0]
        if len(reset_ids):
            state = self.perception.initial(len(reset_ids), sample=False)
            for name in ("deter", "stoch", "logits"):
                getattr(self.runtime.state, name)[reset_ids] = getattr(state, name)

    @torch.no_grad()
    def predicted_failure_features(self, real):
        predicted, _, _ = self.physics.model.step(
            unflatten_state(real["state"]),
            real["actions"],
            real["context"],
            self.physics.scan(unflatten_state(real["state"])),
        )
        next_context = torch.cat((real["actions"], real["context"][:, 29:]), -1)
        return self.task.critic_features(
            predicted, real["learning"]["command"], real["learning"]["next_timers"], next_context
        )

    @torch.no_grad()
    def evaluate_models(self):
        """Sample train/held-out environments without touching training RNG or statistics.

        These finite replay samples measure prediction/calibration, not physical
        action-derivative accuracy or learned locomotion quality.
        """
        cfg, device = self.cfg, self.env.device
        generator = torch.Generator().manual_seed(self.iteration)
        reports = {}
        for split, pool in (
            ("train", self.perception_replay.train_envs),
            ("holdout", self.perception_replay.holdout_envs),
        ):
            if not pool:
                reports[split] = None
                continue
            samples = self.perception_replay.sample(
                cfg.perception_batch,
                cfg.perception_sequence,
                generator=generator,
                holdout=split == "holdout",
            )
            prefix = reconstruct_prefixes(self.perception, [sample.prefix for sample in samples])
            perception_batch = collate_ticks([sample.target for sample in samples], self.perception.cfg, device)
            _, perception_metrics, _ = self.perception.sequence_loss(perception_batch, prefix, sample=False)
            batch = self.replay.sample_physics(
                cfg.physical_batch, cfg.physical_horizon, pool, device, generator=generator
            )
            _, physical = self.physics.window_loss(**batch)
            # window_loss sums over time and divides by batch size. Report
            # per-valid-transition errors so reset frequency cannot hide loss.
            valid = int(batch["mask"].sum())
            physical = {
                name: float(value) * cfg.physical_batch / valid for name, value in physical.items() if name != "valid"
            }
            real = self.replay.sample_real(cfg.real_batch, pool, device, generator=generator, recent_steps=48)
            labels = real["terminated"].float()
            failure = {}
            for name, features in (
                ("true_state", real["learning"]["next_features"]),
                ("predicted_state", self.predicted_failure_features(real)),
            ):
                logits = self.failure(features)
                probability = logits.sigmoid()
                failure[name] = dict(
                    predicted_rate=float(probability.mean()),
                    brier=float((probability - labels).square().mean()),
                    bce=float(torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)),
                )
            reports[split] = dict(
                environments=len(pool),
                perception={name: float(value) for name, value in perception_metrics.items()},
                physical_valid_transitions=valid,
                physical_per_transition=physical,
                failure_samples=len(labels),
                failure_positive_samples=int(labels.sum()),
                actual_failure_rate=float(labels.mean()),
                failure=failure,
            )
        return reports

    def update(self):
        cfg, device = self.cfg, self.env.device
        before = self.update_counts.copy()
        samples = self.perception_replay.sample(cfg.perception_batch, cfg.perception_sequence)
        prefix = reconstruct_prefixes(self.perception, [s.prefix for s in samples])
        batch = collate_ticks([s.target for s in samples], self.perception.cfg, device)
        loss, p_metrics, _ = self.perception.sequence_loss(batch, prefix, sample=False)
        p_grad = optimizer_step(loss, self.perception, self.optimizers["perception"])
        print("NATIVE_DUAL_PERCEPTION", self.iteration, self.update_counts["perception"], flush=True)
        self.refresh_runtime()
        pool = self.perception_replay.train_envs
        for _ in range(cfg.physical_updates):
            batch = self.replay.sample_physics(cfg.physical_batch, cfg.physical_horizon, pool, device)
            loss, f_metrics = self.physics.window_loss(**batch)
            f_grad = optimizer_step(loss, self.physics.model, self.optimizers["physical"])
            real = self.replay.sample_real(cfg.real_batch, pool, device, recent_steps=48)
            with torch.no_grad():
                predicted_features = self.predicted_failure_features(real)
                true_features = real["learning"]["next_features"]
                self.failure.norm.update(true_features)
            # Sampling is uniform over real transitions, not oversampled
            # failures. Native timeouts are excluded from these labels.
            fail_loss = native_failure_loss(
                self.failure(true_features), self.failure(predicted_features), real["terminated"]
            )
            fail_grad = optimizer_step(fail_loss, self.failure, self.optimizers["failure"])
        print("NATIVE_DUAL_PHYSICAL", self.iteration, self.update_counts["physical"], flush=True)
        for _ in range(cfg.style_updates):
            real = self.replay.sample_real(cfg.real_batch, pool, device, recent_steps=48)
            style_metrics = self.style.update(real["learning"]["policy_window"], real["learning"]["reference_window"])
        value_metrics = None
        for _ in range(cfg.real_value_updates):
            windows = self.replay.sample_value_windows(
                self.perception_replay, cfg.real_value_batch, cfg.real_value_horizon, pool
            )
            value_batch = real_value_batch(
                self.task, self.perception, self.perception_replay, self.style, windows, cfg.policy.gamma
            )
            self.critic.norm.update(torch.cat((value_batch["features"], value_batch["next_features"])))
            value_loss, value_metrics = real_value_loss(self.critic, value_batch, cfg.policy.gamma)
            optimizer_step(value_loss, self.critic, self.optimizers["critic"])
        policy_enabled = self.iteration >= cfg.model_only_iterations
        policy_metrics = None
        if policy_enabled:
            start = self.replay.sample_starts(
                self.perception_replay, self.perception, cfg.actor_batch, cfg.policy.horizon, self.env.step_dt
            )
            # Clear supervised gradients so any accidental cross-module gradient
            # from the actor is directly observable at the end of this update.
            for module in self.modules.values():
                module.zero_grad(set_to_none=True)
            policy_metrics = update_native_policy(
                self.task,
                self.actor,
                self.perception,
                self.critic,
                self.style,
                start,
                self.optimizers["policy"],
                self.optimizers["critic"],
                cfg.policy,
                self.failure,
            )
            for name in ("perception", "physical", "failure", "style"):
                if any(p.grad is not None for p in self.modules[name].parameters()):
                    raise RuntimeError(f"Actor/critic leaked a gradient into {name}.")
        delta = {name: self.update_counts[name] - before[name] for name in before}
        expected = dict(
            perception=1,
            physical=cfg.physical_updates,
            style=cfg.style_updates,
            failure=cfg.physical_updates,
            policy=int(policy_enabled),
            critic=cfg.real_value_updates
            + (cfg.policy.critic_epochs * cfg.policy.critic_minibatches if policy_enabled else 0),
        )
        if delta != expected:
            raise RuntimeError(f"Native optimizer step counts differ: {delta} != {expected}.")
        evaluation = self.evaluate_models() if self.iteration % MODEL_EVALUATION_INTERVAL == 0 else None
        self.iteration += 1
        return dict(
            actual_updates=delta,
            iteration=self.iteration,
            policy=policy_metrics,
            model_only=not policy_enabled,
            perception={k: float(v) for k, v in p_metrics.items()},
            physical={k: float(v) for k, v in f_metrics.items()},
            style={k: float(v) for k, v in style_metrics.items()},
            grad_norms=dict(perception=p_grad, physical=f_grad, failure=fail_grad),
            failure_loss=float(fail_loss.detach()),
            real_value={name: float(value) for name, value in value_metrics.items()} if value_metrics else None,
            evaluation=evaluation,
        )

    def save(self, path):
        torch.save(
            dict(
                format="native_hiking_dual_v1",
                cfg=asdict(self.cfg),
                contract=self.contract,
                iteration=self.iteration,
                modules={name: module.state_dict() for name, module in self.modules.items()},
                optimizers={name: opt.state_dict() for name, opt in self.optimizers.items()},
                update_counts=self.update_counts.copy(),
                replay=self.replay.state_dict(),
                perception_replay=self.perception_replay.state_dict(),
                resume_semantics="Models/optimizers/replay resume; simulator begins a new explicit collection segment.",
                initialization=getattr(self, "initialization", None),
            ),
            path,
        )

    def initialize_modules(self, path):
        """Reuse learned models for a new reward objective, with a fresh critic/replay/optimizers."""
        if self.iteration or self.replay.controls or any(self.update_counts.values()):
            raise RuntimeError("Initialize modules only before collecting or updating.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if checkpoint["format"] != "native_hiking_dual_v1":
            raise ValueError("Unsupported initialization checkpoint.")
        source = {k: v for k, v in checkpoint["contract"].items() if k != "task_rewards"}
        destination = {k: v for k, v in self.contract.items() if k != "task_rewards"}
        if source != destination:
            raise ValueError("Initialization physical/perception contract differs.")
        names = ("perception", "physical", "policy", "failure", "style")
        for name in names:
            original, incoming = self.modules[name].state_dict(), checkpoint["modules"][name]
            if original.keys() != incoming.keys() or any(original[k].shape != incoming[k].shape for k in original):
                raise ValueError(f"Initialization module layout differs: {name}.")
        for name in names:
            self.modules[name].load_state_dict(checkpoint["modules"][name])
        self.refresh_runtime()
        self.initialization = dict(
            checkpoint=str(path),
            source_iteration=checkpoint["iteration"],
            reused_modules=list(names),
            reset=["critic", "optimizers", "replay", "iteration", "optimizer_counts"],
        )
        return self.initialization

    def save_controller(self, path):
        save_hiking_controller(path, self.perception, self.actor, self.env.step_dt, self.recorder.joint_names)

    def load(self, path):
        if self.replay.controls:
            raise RuntimeError("Load native checkpoints before collecting in the fresh simulator.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        saved_cfg = dict(checkpoint["cfg"])
        # Legacy checkpoints used no shoe-position objective. Zero preserves
        # that exact objective; a nonzero weight still fails strict resume.
        saved_cfg.setdefault("physical_foot_weight", 0.0)
        saved_cfg.setdefault("real_value_updates", 0)
        saved_cfg.setdefault("real_value_horizon", 30)
        saved_cfg.setdefault("real_value_batch", 128)
        saved_cfg.setdefault("physical_replay_ratio", None)
        if checkpoint["format"] != "native_hiking_dual_v1" or saved_cfg != asdict(self.cfg):
            raise ValueError("Native dual checkpoint configuration differs.")
        if checkpoint.get("contract") != self.contract:
            raise ValueError("Native checkpoint physical/perception contract differs or is unrecorded.")
        for name, module in self.modules.items():
            module.load_state_dict(checkpoint["modules"][name])
            self.optimizers[name].load_state_dict(checkpoint["optimizers"][name])
        self.replay.load_state_dict(checkpoint["replay"])
        self.perception_replay.load_state_dict(checkpoint["perception_replay"])
        self.replay.begin_collection_segment()
        for env, episodes in enumerate(self.perception_replay.episodes):
            self.perception_replay.active[env] = None
            self.runtime.episode_id[env] = max(episodes, default=-1) + 1
        self.iteration = checkpoint["iteration"]
        self.update_counts.update(checkpoint["update_counts"])
        self.refresh_runtime()
