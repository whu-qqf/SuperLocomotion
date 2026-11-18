"""Explicit-budget entry for native Hiking with a single visual world model."""

import gymnasium as gym
import json
import time
import torch
from dataclasses import asdict, replace
from pathlib import Path

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

from .native_latent import NativeLatentCfg, NativeLatentRecorderCfg, NativeLatentRunner, configure_without_amp
from .native_perception import configure_native_perception
from .native_physics_audit import configure_native_physics_audit
from .native_policy import NativePolicyUpdateCfg
from .native_training import configure_flat_walk


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--precision", choices=("float32", "tf32", "bf16"), default=None)
    parser.add_argument("--compile_mode", choices=("eager", "default", "reduce-overhead"), default=None)
    parser.add_argument(
        "--initialize_from", help="Transfer imagined-depth weights to a new terrain; replay/optimizers start fresh."
    )
    parser.add_argument("--flat_walk", action="store_true")
    parser.add_argument(
        "--tgi_aligned", action="store_true", help="TGI policy objective, normalization, losses and effective batches."
    )
    parser.add_argument(
        "--tgi_depth", action="store_true", help="Terrain-only physical model and task-trained depth Actor."
    )
    parser.add_argument(
        "--imagined_camera",
        action="store_true",
        help="Physical-only world model with raycast depth feedback; enables terrain locomotion rewards.",
    )
    parser.add_argument(
        "--terrain_locomotion",
        action="store_true",
        help="Reference-free depth locomotion with terrain rewards and event returns.",
    )
    for name in (
        "collect_steps",
        "model_batch",
        "model_updates",
        "actor_batch",
        "model_only_iterations",
        "policy_microbatch",
    ):
        parser.add_argument("--" + name, type=int, default=None)
    parser.add_argument("--diagnostic", action="store_true", help="One complete update, not a locomotion experiment.")
    args, overrides = cli.parse_cli(parser, require_run_name=False)
    if args.tgi_aligned:
        args.tgi_depth = True
    if args.tgi_depth:
        args.imagined_camera = True
    if args.imagined_camera:
        args.terrain_locomotion = True
    if args.initialize_from and (args.resume or not args.imagined_camera):
        raise ValueError("--initialize_from requires --imagined_camera and cannot be combined with --resume.")
    if args.distributed:
        raise ValueError("This entry supports single-process training only.")
    if args.resume and not args.checkpoint:
        raise ValueError("Resume requires an explicit latent --checkpoint.")
    if args.max_iterations is None or args.max_iterations < 1:
        raise ValueError("An explicit positive --max_iterations budget is required.")
    if args.diagnostic and args.max_iterations != 1:
        raise ValueError("--diagnostic requires --max_iterations 1.")
    cfg, _ = cli.resolve_train_cfg(args.task, overrides)
    configure_native_perception(cfg)
    configure_native_physics_audit(cfg)
    cfg.scene.camera.update_period = cfg.sim.dt * cfg.decimation
    cfg.recorders = NativeLatentRecorderCfg()
    cfg.scene.num_envs = args.num_envs or 1024
    cfg.seed = 1652 if args.seed is None else args.seed
    if args.flat_walk:
        configure_flat_walk(cfg)
    if args.imagined_camera:
        from .native_rewards import configure_imagined_depth_rewards

        configure_imagined_depth_rewards(cfg)
    if args.tgi_aligned:
        cfg.locomotion_reward_weights["locomotion_slip"] = -0.1
        cfg.locomotion_reward_weights["locomotion_support"] = -3.0
        cfg.tgi_aligned = True
        if not args.flat_walk:
            from instinctlab.tasks.parkour.mdp.sgs import configure_sgs

            configure_sgs(cfg)
    reference = None
    configure_without_amp(cfg)
    torch.manual_seed(cfg.seed)
    settings = NativeLatentCfg(
        use_amp=False,
        terrain_locomotion=args.terrain_locomotion,
        imagined_camera=args.imagined_camera,
        tgi_depth=args.tgi_depth,
        tgi_aligned=args.tgi_aligned,
        world_model_heightmap=args.tgi_depth,
        policy=NativePolicyUpdateCfg(touchdown_cut_threshold=0.3 if args.terrain_locomotion else 0.0),
        tracking_reference_dir="",
        paired_reference_dir="",
    )
    if args.tgi_aligned:
        settings = replace(
            settings,
            model_only_iterations=300,
            model_updates=8,
            model_batch=2048,
            actor_batch=2048,
            collect_steps=3,
            warmup_steps=64,
            replay_steps=200,
            learning_rate=2e-4,
            exploration_std=0.1,
            hidden=(1024, 1024, 1024),
        )
    if args.diagnostic:
        settings = replace(
            settings,
            model_only_iterations=0,
            model_updates=1,
            model_batch=8,
            actor_batch=8,
            style_batch=16,
            policy=replace(settings.policy, horizon=32, critic_epochs=1, critic_minibatches=1),
        )
    settings = replace(
        settings,
        **{
            name: getattr(args, name)
            for name in (
                "collect_steps",
                "model_batch",
                "model_updates",
                "actor_batch",
                "model_only_iterations",
                "policy_microbatch",
            )
            if getattr(args, name) is not None
        },
    )
    execution = {}
    if args.resume:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        execution = payload.get("execution", {})
        expected_format = (
            "native_hiking_tgi_aligned_depth_v2"
            if args.tgi_aligned
            else (
                "native_hiking_tgi_terrain_depth_actor_v1"
                if args.tgi_depth
                else (
                    "native_hiking_physics_imagined_depth_v1"
                    if args.imagined_camera
                    else "native_hiking_single_wm_depth_latent_v1"
                )
            )
        )
        if payload["format"] != expected_format:
            raise ValueError("Resume requires the new latent architecture; old policies are incompatible.")
        saved = dict(payload["cfg"])
        saved.setdefault("use_amp", True)
        saved.setdefault("tracking_reference_dir", "")
        saved.setdefault("paired_reference_dir", "")
        saved.setdefault("world_model_heightmap", True)
        saved.setdefault("terrain_locomotion", False)
        saved.setdefault("imagined_camera", False)
        if saved["terrain_locomotion"] != args.terrain_locomotion:
            raise ValueError("Checkpoint locomotion objective differs. Start fresh to change terrain rewards.")
        if saved["world_model_heightmap"] != settings.world_model_heightmap:
            raise ValueError(
                "Checkpoint F uses privileged heightmap inputs. Start fresh for the depth-only world model."
            )
        if saved["use_amp"]:
            raise ValueError("Checkpoint AMP setting differs. Start a fresh run to change the training objective.")
        if saved["tracking_reference_dir"] != settings.tracking_reference_dir:
            raise ValueError("Checkpoint motion tracking setting differs. Start fresh to change reference supervision.")
        if saved["paired_reference_dir"] != settings.paired_reference_dir:
            raise ValueError("Checkpoint paired dataset differs. Start fresh to change recorded terrain supervision.")
        saved["policy"] = NativePolicyUpdateCfg(**saved["policy"])
        settings = NativeLatentCfg(**saved)
        del payload
    cfg.terrain_locomotion = settings.terrain_locomotion
    if settings.terrain_locomotion:
        from isaaclab.managers import TerminationTermCfg

        from .native_physics_audit import native_scan_out_of_bounds

        cfg.terminations.critic_scan_out_of_bounds = TerminationTermCfg(func=native_scan_out_of_bounds, time_out=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with launch_simulation(cfg, args):
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        vec = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        try:
            runner = NativeLatentRunner(vec, settings)
            if args.resume:
                runner.load(args.checkpoint)
            elif args.initialize_from:
                runner.initialize_from(args.initialize_from)
            from .native_execution import configure_network_execution

            execution = configure_network_execution(
                runner,
                precision=args.precision or execution.get("precision", "float32"),
                compile_mode=args.compile_mode or execution.get("compile_mode", "eager"),
            )
            # Microbatching changes execution memory, not the effective batch.
            # Apply overrides after strict checkpoint validation and restore.
            if args.resume and args.policy_microbatch is not None:
                runner.cfg = replace(runner.cfg, policy_microbatch=args.policy_microbatch)
                settings = runner.cfg
            (output / "config.json").write_text(
                json.dumps(
                    dict(
                        task=args.task,
                        num_envs=cfg.scene.num_envs,
                        seed=cfg.seed,
                        settings=asdict(settings),
                        execution=execution,
                        reference=reference,
                        resumed_from=args.checkpoint if args.resume else None,
                        initialized_from=args.initialize_from,
                    ),
                    indent=2,
                )
                + "\n"
            )
            training_start = time.perf_counter()
            with (output / "metrics.jsonl").open("w") as metrics:
                for index in range(args.max_iterations):
                    started = time.perf_counter()
                    collection = runner.collect()
                    collected = time.perf_counter()
                    update = runner.update()
                    if runner.iteration % 100 == 0:
                        update["evaluation"] = runner.evaluate_models()
                    elapsed = time.perf_counter() - training_start
                    record = dict(
                        collection=collection, **update, execution=execution, seconds=time.perf_counter() - started
                    )
                    episode_steps = collection["episode"]["mean_episode_length"]
                    survival_seconds = episode_steps * runner.env.step_dt if episode_steps is not None else None
                    record["collection"]["episode"]["mean_episode_seconds"] = survival_seconds
                    record["performance"] = dict(
                        collection_seconds=collected - started,
                        learning_seconds=time.perf_counter() - collected,
                        iteration_seconds=record["seconds"],
                        session_total_seconds=elapsed,
                        eta_seconds=elapsed / (index + 1) * (args.max_iterations - index - 1),
                        transitions_per_second=collection["collected_controls"]
                        * cfg.scene.num_envs
                        / record["seconds"],
                        total_timesteps=collection["real_controls"] * cfg.scene.num_envs,
                    )
                    sgs = getattr(runner.env, "_hiking_sgs", None)
                    if sgs is not None:
                        record["sgs"] = sgs["sampler"].stats()
                        if runner.iteration % 100 == 0 or args.diagnostic:
                            rows, cols = sgs["rows"], sgs["cols"]
                            sampler = sgs["sampler"]
                            probability = sampler.probs().view(rows, cols, -1)
                            success = sampler.success_rate().view(rows, cols, -1)
                            counts = sampler.count.view(rows, cols, -1)
                            groups = sampler.group.view(rows, cols, -1)[0, :, 0]
                            names = list(runner.env.scene.terrain.cfg.terrain_generator.sub_terrains)
                            record["sgs"]["by_terrain_type"] = {
                                names[group]: dict(
                                    success_by_level=success[:, groups == group].mean((1, 2)).tolist(),
                                    sample_share_by_level=probability[:, groups == group].sum((1, 2)).tolist(),
                                    outcomes_by_level=counts[:, groups == group].sum((1, 2)).tolist(),
                                )
                                for group in groups.unique().tolist()
                            }
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    print(
                        "HIKING_LATENT",
                        json.dumps(
                            dict(
                                iteration=runner.iteration,
                                mean_episode_seconds=survival_seconds,
                                completed_episodes=collection["episode"]["completed_episodes"],
                                model=update["model"],
                                actor_grad_norm=update["policy"]["actor_grad_norm"] if update["policy"] else None,
                                updates=update["actual_updates"],
                                seconds=record["seconds"],
                            )
                        ),
                        flush=True,
                    )
                    if runner.iteration % 100 == 0:
                        runner.save(output / f"model_{runner.iteration}.pt", include_replay=False)
                        runner.save(output / "latest.pt")
            evaluation = runner.evaluate_models()
            (output / "model_evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")
            runner.save(output / f"model_{runner.iteration}.pt", include_replay=False)
            runner.save(output / "latest.pt")
            runner.save_controller(output / "controller.pt")
            # Deployment exports need only the depth encoder and actor.
            from instinctlab_supermimic.depth import load_depth_controller
            from instinctlab_supermimic.networks import run_mlp

            controller = load_depth_controller(output / "controller.pt", vec.device)
            # Match the full measured history rather than filling it with one image.
            controller.history = runner.history
            with torch.no_grad():
                features = None
                expected = runner.actor(runner.obs[:, :768], runner.perception.encode(runner.history.stack()), features)
                actual = controller.actor(
                    runner.obs[:, :768], run_mlp(controller.encoder, runner.history.stack()), features
                )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            (output / "checks.json").write_text(
                json.dumps(
                    dict(
                        architecture=runner.contract["architecture"],
                        fresh_initialization=not args.resume,
                        optimizer_updates=runner.update_counts,
                        controller_roundtrip=True,
                        scope=(
                            "Dataflow/optimizer check only; not evidence of learned gait or accurate model gradients."
                        ),
                    ),
                    indent=2,
                )
                + "\n"
            )
            (output / "results.json").write_text(
                json.dumps(
                    dict(
                        iterations=args.max_iterations,
                        final_iteration=runner.iteration,
                        optimizer_updates=runner.update_counts,
                        num_envs=cfg.scene.num_envs,
                    ),
                    indent=2,
                )
                + "\n"
            )
        finally:
            vec.close()
