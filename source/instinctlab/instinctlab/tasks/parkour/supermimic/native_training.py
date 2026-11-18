"""Explicitly budgeted native Hiking dual-model training entry point."""

import gymnasium as gym
import json
import time
import torch
from dataclasses import replace
from pathlib import Path

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

from .native_perception import configure_native_perception
from .native_physics_audit import configure_native_physics_audit
from .native_policy import NativePolicyUpdateCfg
from .native_rewards import configure_native_tracking_rewards
from .native_runner import NativeDualRunner, NativeRunnerCfg
from .native_transitions import NativeHikingRecorderCfg


def configure_flat_walk(cfg):
    """Isolate basic forward locomotion with the existing network and task heads."""
    from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
    from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=0,
        size=(30.0, 8.0),
        num_rows=1,
        num_cols=1,
        border_width=3.0,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "flat_walk": MeshPlaneTerrainCfg(
                flat_patch_sampling={
                    "target": FlatPatchSamplingCfg(
                        num_patches=50,
                        patch_radius=0.2,
                        x_range=(12.0, 13.0),
                        y_range=(-0.1, 0.1),
                        max_height_diff=0.001,
                    )
                },
            )
        },
    )
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None
    command = cfg.commands.base_velocity
    command.random_velocity_terrain = []
    command.velocity_ranges = {
        "flat_walk": {"lin_vel_x": (0.5, 0.5), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)}
    }
    command.rel_standing_envs = 0.0
    return cfg


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--collect_steps", type=int, default=None)
    parser.add_argument("--physical_horizon", type=int, default=None, help="Supervised F rollout length.")
    parser.add_argument(
        "--physical_foot_weight",
        type=float,
        default=None,
        help="Experimental shoe geometry loss weight; zero preserves the original objective.",
    )
    parser.add_argument("--tracking_rewards", action="store_true", help="Add dense target progress and smooth waiting.")
    parser.add_argument("--flat_walk", action="store_true", help="Pure flat ground and forward target at 0.5 m/s.")
    parser.add_argument(
        "--initialize_from", help="Reuse learned modules, reset critic/optimizers/replay for new rewards."
    )
    parser.add_argument("--model_only_iterations", type=int, default=None)
    parser.add_argument(
        "--physical_replay_ratio",
        type=float,
        default=None,
        help="Nominal F sampled transitions per collected transition; scales update frequency with environment count.",
    )
    parser.add_argument(
        "--exploration_std",
        type=float,
        default=None,
        help="Collection noise std in raw action units, before joint scaling.",
    )
    parser.add_argument(
        "--check_task_readouts", action="store_true", help="Check real command/reward parity before reset."
    )
    parser.add_argument(
        "--append_output", action="store_true", help="Continue the same JSONL after its exact saved boundary."
    )
    parser.add_argument(
        "--check_physics_gradients",
        action="store_true",
        help="After saving: compare current F action derivatives with restored real Newton perturbations; no fitting.",
    )
    parser.add_argument(
        "--check_checkpoint",
        action="store_true",
        help="After training: verify restored collection (8 controls) and exported inference (12 controls).",
    )
    parser.add_argument(
        "--diagnostic_full_update",
        action="store_true",
        help="One-step module diagnostic schedule; not the formal training schedule.",
    )
    args, overrides = cli.parse_cli(parser, require_run_name=False)
    if args.max_iterations is None or args.max_iterations < 1:
        raise ValueError("Specify an explicit positive --max_iterations training budget.")
    if args.distributed:
        raise ValueError("Native dual runner currently supports one process only.")
    if args.resume and not args.checkpoint:
        raise ValueError("Native resume requires the exact --checkpoint path.")
    if args.append_output and not args.resume:
        raise ValueError("Appending native logs requires checkpoint resume.")
    if args.initialize_from and args.resume:
        raise ValueError("Module initialization and exact resume are mutually exclusive.")
    runner_cfg = NativeRunnerCfg()
    if args.resume:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
        settings = dict(checkpoint["cfg"])
        settings["policy"] = NativePolicyUpdateCfg(**settings["policy"])
        runner_cfg = NativeRunnerCfg(**settings)
        del checkpoint
    changes = {
        name: getattr(args, name)
        for name in (
            "collect_steps",
            "physical_horizon",
            "physical_foot_weight",
            "model_only_iterations",
            "exploration_std",
            "physical_replay_ratio",
        )
        if getattr(args, name) is not None
    }
    runner_cfg = replace(runner_cfg, **changes)
    cfg, _ = cli.resolve_train_cfg(args.task, overrides)
    cfg = configure_native_perception(cfg)
    if args.flat_walk:
        configure_flat_walk(cfg)
    if args.tracking_rewards:
        configure_native_tracking_rewards(cfg)
    configure_native_physics_audit(cfg)
    cfg.recorders = NativeHikingRecorderCfg()
    if args.check_task_readouts:
        from .native_task_audit import NativeTaskAuditRecorderCfg

        cfg.recorders = NativeTaskAuditRecorderCfg()
    cfg.scene.num_envs = args.num_envs or 128
    cfg.seed = 157 if args.seed is None else args.seed
    torch.manual_seed(cfg.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=args.append_output)
    if args.flat_walk:
        (output / "flat_walk_config.json").write_text(
            json.dumps(
                dict(
                    environment=cfg.to_dict(),
                    runner=runner_cfg.__dict__,
                    random_initialization=not (args.resume or args.initialize_from),
                ),
                default=str,
                indent=2,
            )
            + "\n"
        )
    with launch_simulation(cfg, args):
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        vec = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        try:
            if args.diagnostic_full_update:
                runner_cfg = replace(
                    runner_cfg,
                    warmup_steps=args.collect_steps or 48,
                    physical_updates=1,
                    style_updates=1,
                    model_only_iterations=0,
                )
            runner = NativeDualRunner(vec, runner_cfg)
            frequency = (
                1.0 / runner_cfg.collect_steps
                if runner_cfg.collect_steps is not None
                else runner_cfg.update_frequency(cfg.scene.num_envs)
            )
            schedule = dict(
                num_envs=cfg.scene.num_envs,
                iterations_per_vector_control=frequency,
                nominal_physical_replay_ratio=(
                    frequency
                    * runner_cfg.physical_updates
                    * runner_cfg.physical_batch
                    * runner_cfg.physical_horizon
                    / cfg.scene.num_envs
                ),
                warmup_controls=runner_cfg.warmup_steps,
                scope="Nominal sampled F transitions, excluding warmup and terminal masks; not unique samples.",
                coupled_updates=(
                    "Collection frequency also changes actor, perception, style and critic updates per real step."
                ),
            )
            (output / "training_schedule.json").write_text(json.dumps(schedule, indent=2) + "\n")
            print("NATIVE_TRAINING_SCHEDULE", json.dumps(schedule), flush=True)
            if args.resume:
                runner.load(args.checkpoint)
            if args.initialize_from:
                initialization = runner.initialize_modules(args.initialize_from)
                (output / "initialization.json").write_text(json.dumps(initialization, indent=2) + "\n")
            if args.check_task_readouts:
                vec.unwrapped.recorder_manager._terms["task_parity"].bind(runner.physics)
            if runner.cfg.real_value_updates:
                # Fresh measurements after reset; legacy reward placeholders
                # are never counted as observed zeros or used as TD labels.
                prefill = runner.collect(runner.cfg.real_value_horizon + runner.perception.cfg.action_steps)
                (output / "real_value_prefill.json").write_text(json.dumps(prefill, indent=2) + "\n")
            initial_iteration = runner.iteration
            history_start = initial_iteration
            if args.append_output:
                history = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
                if not history or history[-1]["iteration"] != initial_iteration:
                    raise ValueError("Append output must end exactly at the restored checkpoint iteration.")
                history_start = history[0]["iteration"] - 1
            training_start = time.perf_counter()
            with (output / "metrics.jsonl").open("a" if args.append_output else "w") as metrics:
                for index in range(args.max_iterations):
                    iteration_start = time.perf_counter()
                    collection = runner.collect()
                    collection_end = time.perf_counter()
                    update = runner.update()
                    update_end = time.perf_counter()
                    duration = update_end - iteration_start
                    elapsed = update_end - training_start
                    performance = dict(
                        collection_seconds=collection_end - iteration_start,
                        learning_seconds=update_end - collection_end,
                        iteration_seconds=duration,
                        session_total_seconds=elapsed,
                        eta_seconds=elapsed / (index + 1) * (args.max_iterations - index - 1),
                        transitions_per_second=collection["collected_controls"] * vec.num_envs / duration,
                        total_timesteps=collection["real_controls"] * vec.num_envs,
                        collection_action_noise_std=runner_cfg.exploration_std,
                    )
                    record = dict(collection=collection, performance=performance, **update)
                    metrics.write(json.dumps(record) + "\n")
                    metrics.flush()
                    print("NATIVE_DUAL_UPDATE", json.dumps(record), flush=True)
                    if runner.iteration % 100 == 0:
                        runner.save(output / f"model_{runner.iteration}.pt")
            runner.save(output / f"model_{runner.iteration}.pt")
            runner.save_controller(output / "controller.pt")
            if args.check_task_readouts:
                parity = vec.unwrapped.recorder_manager._terms["task_parity"].result()
                (output / "task_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
                if not parity["passed"]:
                    raise AssertionError(f"Native command/reward parity failed: {parity['failed_readouts']}")
            physics_report = None
            if args.check_physics_gradients:
                from .native_physics_audit import probe_native_physics

                before = runner.update_counts.copy()
                physics_report = probe_native_physics(runner.physics, output)
                if runner.update_counts != before:
                    raise AssertionError("Native physical gradient probe changed optimizer counters.")
                (output / "physics_gradient_probe.json").write_text(json.dumps(physics_report, indent=2) + "\n")
                print("NATIVE_PHYSICS_GRADIENT_PROBE", json.dumps(physics_report), flush=True)
            checkpoint_report = None
            if args.check_checkpoint:
                from .native_checkpoint_audit import check_native_checkpoint

                checkpoint_report = check_native_checkpoint(
                    runner, output / f"model_{runner.iteration}.pt", output / "controller.pt", output
                )
            (output / "results.json").write_text(
                json.dumps(
                    dict(
                        task=args.task,
                        envs=vec.num_envs,
                        iterations=runner.iteration - history_start,
                        segment_iterations=args.max_iterations,
                        final_iteration=runner.iteration,
                        total_optimizer_updates=runner.update_counts,
                        native_terrain=True,
                        native_camera=True,
                        checkpoint_roundtrip=checkpoint_report,
                        physics_gradient_probe=physics_report,
                        scope="Budgeted dual-model updates; short runs do not demonstrate learned locomotion.",
                    ),
                    indent=2,
                )
                + "\n"
            )
        finally:
            vec.close()
