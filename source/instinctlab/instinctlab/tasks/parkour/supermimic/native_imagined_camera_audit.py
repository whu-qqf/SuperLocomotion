"""Compare independent imagined depth with the native camera at measured poses."""

import gymnasium as gym
import json
import torch
from pathlib import Path

import warp as wp
from instinctlab_supermimic.depth import DepthHistory

from isaaclab.app import launch_simulation
from isaaclab.managers import TerminationTermCfg

import instinctlab.tasks  # noqa: F401
from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.utils import instinct_rl_cli as cli

from .native_imagined_camera import NativeImaginedDepthCamera
from .native_latent import NativeLatentRecorderCfg, camera_image, configure_without_amp
from .native_perception import configure_native_perception
from .native_physics_audit import configure_native_physics_audit
from .native_training import configure_flat_walk


def partial_timeout(env):
    """Exercise camera histories across one deliberate same-step reset."""
    return (torch.arange(env.num_envs, device=env.device) == 0) & (env.common_step_counter == 8)


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--terrain", choices=("flat", "native_hiking"), default="flat")
    args, overrides = cli.parse_cli(parser, require_run_name=False)
    cfg, _ = cli.resolve_train_cfg(args.task, overrides)
    configure_native_perception(cfg)
    configure_native_physics_audit(cfg)
    configure_without_amp(cfg)
    if args.terrain == "flat":
        configure_flat_walk(cfg)
    cfg.scene.num_envs = args.num_envs or 8
    cfg.scene.camera.update_period = cfg.sim.dt * cfg.decimation
    cfg.recorders = NativeLatentRecorderCfg()
    cfg.terminations.camera_audit_timeout = TerminationTermCfg(func=partial_timeout, time_out=True)
    cfg.seed = 1681
    torch.manual_seed(cfg.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    with launch_simulation(cfg, args):
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        vec = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        try:
            env = vec.unwrapped
            vec.reset()
            sensor = env.scene["camera"]
            recorder = env.recorder_manager._terms["transition"]
            query = NativeImaginedDepthCamera(env)
            ids = torch.arange(env.num_envs, device=env.device)
            real_history = DepthHistory(camera_image(env))
            imagined_history = DepthHistory(query.render(recorder.state(), ids)["image"])
            reports = []
            resets, history_errors = 0, []
            for step in range(args.steps):
                state = recorder.state()
                measured_raw = sensor.data.output["distance_to_image_plane"].torch.clone()
                measured_image = camera_image(env)
                before = dict(
                    raw=measured_raw,
                    processed=sensor._data.output["distance_to_image_plane_noised"].torch.clone(),
                    frame=sensor.frame_generation.clone(),
                    positions=query.mesh_positions.clone(),
                    live_mesh_positions=wp.to_torch(sensor._mesh_positions_w).clone(),
                )
                result = query.render(state, ids)
                assert not result["image"].requires_grad
                torch.testing.assert_close(
                    sensor._data.output["distance_to_image_plane"].torch, before["raw"], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    sensor._data.output["distance_to_image_plane_noised"].torch, before["processed"], rtol=0, atol=0
                )
                torch.testing.assert_close(sensor.frame_generation, before["frame"], rtol=0, atol=0)
                torch.testing.assert_close(query.mesh_positions, before["positions"], rtol=0, atol=0)
                torch.testing.assert_close(
                    wp.to_torch(sensor._mesh_positions_w), before["live_mesh_positions"], rtol=0, atol=0
                )
                report = dict(
                    step=step,
                    raw_max_error_m=float((result["raw"] - measured_raw).abs().max()),
                    processed_max_error=float((result["image"] - measured_image).abs().max()),
                    camera_position_max_error_m=float((result["position"] - sensor.data.pos_w.torch).abs().max()),
                )
                reports.append(report)
                action = torch.zeros(env.num_envs, 29, device=env.device)
                action[:, list(G1_29DOF_POLICY_JOINT_ORDER_V1).index("left_knee_joint")] = 0.3 * (step % 2)
                _, _, done, _ = vec.step(action)
                transition = recorder.take_transition()
                real_history.advance(recorder.next_depth)
                imagined_history.advance(query.render(transition.next_state, ids)["image"])
                reset_ids = done.nonzero(as_tuple=True)[0]
                resets += len(reset_ids)
                if len(reset_ids):
                    real_history.reset(reset_ids, camera_image(env))
                    reset_image = query.render(recorder.state(), ids)["image"]
                    imagined_history.reset(reset_ids, reset_image)
                history_errors.append(float((real_history.stack() - imagined_history.stack()).abs().max()))
            state = recorder.state()
            duplicate_ids = torch.tensor([0, 0, env.num_envs - 1], device=env.device)
            duplicate_state = state[duplicate_ids].clone().requires_grad_(True)
            duplicate = query.render(duplicate_state, duplicate_ids)
            torch.testing.assert_close(duplicate["raw"][0], duplicate["raw"][1], rtol=0, atol=0)
            assert not duplicate["image"].requires_grad
            original = query.render(state, ids)
            changes = []
            shoulder = 13 + list(G1_29DOF_POLICY_JOINT_ORDER_V1).index("left_shoulder_pitch_joint")
            for angle in (-1.5, 1.5):
                imagined = state.clone()
                imagined[:, shoulder] = angle
                changed = query.render(imagined, ids)
                torch.testing.assert_close(changed["position"], original["position"], rtol=0, atol=0)
                torch.testing.assert_close(changed["quaternion"], original["quaternion"], rtol=0, atol=0)
                changes.append(
                    dict(
                        angle=angle,
                        processed_max_change=float((changed["image"] - original["image"]).abs().max()),
                        changed_raw_pixels=int(((changed["raw"] - original["raw"]).abs() > 1e-5).sum()),
                    )
                )
            assert (
                max(row["processed_max_change"] for row in changes) > 1e-5
            ), "Arm poses did not affect self-occlusion."
            assert max(row["raw_max_error_m"] for row in reports) < 1e-3
            assert max(row["processed_max_error"] for row in reports) < 1e-4
            assert max(row["camera_position_max_error_m"] for row in reports) < 1e-5
            assert max(history_errors) < 1e-4
            benchmark = {}
            for batch in (env.num_envs, 128):
                sources = torch.arange(batch, device=env.device) % env.num_envs
                poses = state[sources]
                query.render(poses, sources)
                start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_event.record()
                for _ in range(16):
                    query.render(poses, sources)
                end_event.record()
                end_event.synchronize()
                milliseconds = start_event.elapsed_time(end_event) / 16
                benchmark[str(batch)] = dict(
                    milliseconds_per_query=milliseconds,
                    images_per_second=batch * 1000 / milliseconds,
                    includes="FK, self-occlusion, raycast, postprocessing; excludes encoder",
                )
            report = dict(
                terrain=args.terrain,
                num_envs=env.num_envs,
                controls=args.steps,
                parity=reports,
                duplicate_env_indices_verified=True,
                camera_gradient_detached=True,
                live_outputs_unchanged=True,
                shoulder_pose_self_occlusion=changes,
                depth_history_max_error=max(history_errors),
                measured_resets=resets,
                optimizer_updates=0,
                benchmark=benchmark,
                parity_gates_passed=True,
            )
            torch.save(
                dict(state=state.cpu(), raw=original["raw"].cpu(), image=original["image"].cpu()), output / "samples.pt"
            )
            (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print("IMAGINED_CAMERA_AUDIT", json.dumps(report), flush=True)
        finally:
            vec.close()
