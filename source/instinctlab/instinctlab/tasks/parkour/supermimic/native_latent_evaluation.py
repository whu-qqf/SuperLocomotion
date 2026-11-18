"""First-episode flat/Hiking evaluation and restored physical derivative measurements."""

import gymnasium as gym
import hashlib
import json
import torch
from pathlib import Path

from instinctlab_supermimic.depth import DepthAutoencoder, DepthHistory, DepthLatentPolicy
from instinctlab_supermimic.physical import flatten_state, unflatten_state
from instinctlab_supermimic.rotations import mv

from isaaclab.app import launch_simulation

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

from .action_gradient_audit import compare_probe_derivatives, first_action_derivative
from .lafan_reference import configure_lafan_reference
from .native_latent import NativeLatentRecorderCfg, camera_image, configure_without_amp
from .native_perception import configure_native_perception
from .native_physics import shoe_sole_points
from .native_physics_audit import (
    NativePhysicsProbeTask,
    configure_native_physics_audit,
    make_native_physics,
    native_height_field,
)
from .native_training import configure_flat_walk
from .newton_probes import collect_newton_action_probes
from .stair_kinematics import NativeLinkKinematics


def terrain_column_type_names(env):
    """Map generated terrain columns to names using their configured proportions.

    Terrain rows are difficulty levels; columns are terrain types. This mirrors
    TGI's column grouping and keeps evaluation labels independent of a random
    reset assignment.
    """
    generator = env.cfg.scene.terrain.terrain_generator
    names = tuple(generator.sub_terrains)
    proportions = torch.tensor([float(generator.sub_terrains[name].proportion) for name in names], dtype=torch.float32)
    cumulative = proportions.cumsum(0) / proportions.sum()
    columns = torch.arange(generator.num_cols, dtype=torch.float32)
    groups = torch.stack([(columns / generator.num_cols + 0.001 < limit) for limit in cumulative], -1)
    indices = groups.float().argmax(-1)
    return tuple(names[int(index)] for index in indices)


def assign_fixed_terrain_evaluation(env, terrain_type, terrain_level):
    """Pin every evaluation environment to one terrain column and difficulty row."""
    terrain = env.scene.terrain
    names = terrain_column_type_names(env)
    if terrain_type not in names:
        raise ValueError(f"Unknown terrain type {terrain_type!r}; choose from {sorted(set(names))}.")
    if terrain_level < 0 or terrain_level >= terrain.terrain_origins.shape[0]:
        raise ValueError(f"Terrain level {terrain_level} is outside [0, {terrain.terrain_origins.shape[0]}).")
    column = names.index(terrain_type)
    levels = torch.full_like(terrain.terrain_levels, terrain_level)
    columns = torch.full_like(terrain.terrain_types, column)
    origins = terrain.terrain_origins[levels, columns]
    terrain.terrain_levels[:] = levels
    terrain.terrain_types[:] = columns
    terrain.env_origins[:] = origins
    env.scene.env_origins[:] = origins
    return dict(terrain_type=terrain_type, terrain_column=column, terrain_level=terrain_level)


def collision_mesh_hash(env):
    """Identify the static collision surface before replaying recorded world poses."""
    mesh = env.scene.terrain.dynamics_mesh
    digest = hashlib.sha256()
    for value in (mesh.vertices, mesh.faces):
        digest.update(value.tobytes())
    return digest.hexdigest()


def root_com_body_velocity(state, root_com_offset_b):
    """Match native reward velocity: root COM velocity in the full body frame."""
    _, rotation, link_velocity, angular_velocity, _, _ = unflatten_state(state)
    shift_w = mv(rotation, root_com_offset_b)
    velocity_w = link_velocity + torch.linalg.cross(angular_velocity, shift_w, dim=-1)
    return mv(rotation.transpose(-1, -2), velocity_w)


def swing_clearance_summary(clearance, valid, contact, alive, command):
    """Summarize T x B x 2 measured sole gaps, excluding flight and reset frames.

    Report only the airborne foot during commanded movement and measured single
    support. A double-airborne jump must not count as successful foot lifting.
    Missing valid samples are reported as null, rather than a perfect zero error.
    """
    moving = command[..., :2].norm(dim=-1) + command[..., 2].abs() > 0.2
    single = contact[..., 0] ^ contact[..., 1]
    candidate = (alive & moving & single)[..., None] & ~contact
    selected = candidate & valid
    samples = clearance[selected]
    quantiles = samples.quantile(samples.new_tensor([0.1, 0.5, 0.9])).tolist() if samples.numel() else None
    return dict(
        sample_count=int(selected.sum()),
        invalid_terrain_sample_count=int((candidate & ~valid).sum()),
        mean_m=float(samples.mean()) if samples.numel() else None,
        p10_p50_p90_m=quantiles,
        below_8cm_fraction=float((samples < 0.08).float().mean()) if samples.numel() else None,
        per_foot_sample_count=selected.sum((0, 1)).tolist(),
        scope=(
            "Measured first-episode single-support swing during commanded movement; minimum of four sole"
            " proxy gaps above the terrain raster. Excludes flight; not an obstacle-path collision test."
        ),
    )


def replay_trace(vec, trace_path, output):
    """Render measured env0 poses through its terminal frame; never simulate them."""
    import numpy as np

    import cv2

    env = vec.unwrapped
    vec.reset()
    trace = torch.load(trace_path, map_location="cpu", weights_only=False)
    if trace.get("terrain", "flat") != env.cfg.evaluation_terrain:
        raise ValueError("Replay scene terrain differs from the recorded rollout.")
    states = [trace["initial"][0]]
    for row in trace["traces"]:
        states.append(row["state"][0])
        if row["done"][0]:
            break
    robot = env.scene["robot"]
    order = [trace["joint_names"].index(name) for name in robot.joint_names]
    if env.cfg.evaluation_terrain == "native_hiking":
        if trace.get("collision_mesh_sha256") != collision_mesh_hash(env):
            raise ValueError("Native replay collision mesh differs from the recorded terrain.")
        offset = torch.zeros(2)
    else:
        offset = states[0][:2] - env.scene.env_origins[0, :2].cpu()
    viz = env.sim.visualizers[0]
    writer = None
    try:
        for step, measured in enumerate(states):
            state = measured.clone().to(env.device)
            state[:2] -= offset.to(env.device)
            pose = state[[0, 1, 2, 4, 5, 6, 3]][None]
            joints = state[13:42][order][None]
            robot.write_root_pose_to_sim_index(root_pose=pose)
            robot.write_joint_state_to_sim(joints, torch.zeros_like(joints))
            env.sim.backend_manager.forward()
            x, y = state[:2].tolist()
            viz.set_camera_view((x + 2.1, y - 2.1, 1.5), (x, y, 0.55))
            env.sim.render()
            frame = viz.render_rgb_array()
            frame = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
            frame = cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 100), (25, 25, 25), -1)
            lines = (
                (
                    f"Checkpoint {trace.get('checkpoint_iteration', 'unknown')} | measured"
                    f" {trace.get('terrain', 'flat')} rollout | env 0 | 0.2x speed"
                ),
                f"t={step * env.step_dt:.2f}s  root height={float(state[2]):.3f}m",
                "End of first-episode recording" if step == len(states) - 1 else "Pose replay; no resimulation",
            )
            for row, line in enumerate(lines):
                cv2.putText(
                    frame, line, (15, 27 + 29 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1, cv2.LINE_AA
                )
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output / "env0_measured_slow.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height)
                )
                if not writer.isOpened():
                    raise RuntimeError("Could not open replay video writer.")
            writer.write(frame)
            if step in (0, len(states) // 2, len(states) - 1):
                cv2.imwrite(str(output / f"frame_{step:03d}.png"), frame)
        for _ in range(10):
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()


def probe_latent_physics(
    physics, latent, action, output, label, *, imagined_camera=None, perception=None, depth_history=None
):
    """Freeze the action sequence; compare the first action's physical effects."""
    initial = physics.recorder.state()
    context = physics.context(physics.env.action_manager.action)
    actions = action.detach()[:, None].expand(-1, 8, -1).clone()
    direction = torch.randn_like(action)
    direction /= direction.norm(dim=-1, keepdim=True)
    knees = [physics.recorder.joint_names.index(f"{side}_knee_joint") for side in ("left", "right")]
    # Include single-joint directions as well as normalized random directions.
    for group, joint in enumerate(knees):
        direction[group::3] = 0
        direction[group::3, joint] = 1

    def readout(states):
        states = states.to(initial.device)
        feet = physics.foot_points(unflatten_state(states[..., :71].reshape(-1, 71))).mean(-2)
        feet = feet.reshape(*states.shape[:-1], 6)
        return torch.cat((states[..., 7:10], states[..., 2:3], states[..., [13 + j for j in knees]], feet), -1)

    def predict(controls):
        state, xi, z = unflatten_state(initial), context, latent.detach()
        states = []
        sequence = None
        if imagined_camera is not None:
            from .native_imagined_camera import ImaginedDepthSequence

            sequence = ImaginedDepthSequence(
                imagined_camera, perception, depth_history, torch.arange(len(initial), device=initial.device)
            )
        active = torch.ones(len(initial), device=initial.device, dtype=torch.bool)
        for index, control in enumerate(controls.unbind(1)):
            state, _, z = physics.model.step(state, control, xi, physics.model_scan(state), z)
            if sequence is not None:
                z = sequence(state, active, index + 1)
            xi = torch.cat((control, xi[..., 29:]), -1)
            states.append(flatten_state(state))
        return readout(torch.stack(states, 1))

    before = {name: value.clone() for name, value in physics.model.state_dict().items()}
    derivative = first_action_derivative(predict, actions, direction)
    initial_contact = (
        physics.env.scene["contact_forces"].data.current_contact_time.torch[:, physics.foot_contact_ids] > 0
    ).cpu()
    probes = collect_newton_action_probes(
        physics.env, NativePhysicsProbeTask(physics), actions, direction, contact_body_ids=physics.foot_contact_ids
    )
    reports = {}
    for name, indices in (
        ("root_velocity", [0, 1, 2]),
        ("root_height", [3]),
        ("knees", [4, 5]),
        ("feet", list(range(6, 12))),
    ):
        reports[name] = compare_probe_derivatives(
            probes, lambda states, ids=indices: readout(states)[..., ids], derivative[..., indices]
        )
    contact = probes["outcomes"]["zero_a"]["foot_contact"]
    previous = torch.cat((initial_contact[:, None], contact[:, :-1]), 1)
    reports["feet_by_contact"] = {
        mode: compare_probe_derivatives(
            probes,
            lambda states: readout(states)[..., 6:12],
            derivative[..., 6:12],
            component_mask=selected[..., None].expand(-1, -1, -1, 3).flatten(-2),
        )
        for mode, selected in dict(stance=contact, swing=~contact, touchdown=contact & ~previous).items()
    }
    forward_derivatives = {}
    if imagined_camera is not None:
        # Forward perturbations include the camera's pose dependence. Compare
        # them with the actor's partial path derivative before attributing a
        # physical Jacobian error to inaccurate dynamics alone.
        with torch.no_grad():
            for epsilon in (0.05, 0.15):
                plus, minus = actions.clone(), actions.clone()
                plus[:, 0] += epsilon * direction
                minus[:, 0] -= epsilon * direction
                forward = (predict(plus) - predict(minus)) / (2 * epsilon)
                forward_derivatives[str(epsilon)] = forward.cpu()
                reports[f"feet_full_forward_{epsilon}"] = compare_probe_derivatives(
                    probes, lambda states: readout(states)[..., 6:12], forward[..., 6:12], epsilons=(epsilon,)
                )
    for name, value in physics.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.save(
        dict(
            probes=probes,
            initial_latent=latent.detach().cpu(),
            model_derivative=derivative.cpu(),
            model_forward_derivatives=forward_derivatives,
            depth_history=depth_history.detach().cpu().clone() if depth_history is not None else None,
        ),
        output / f"probes_{label}.pt",
    )
    return reports


def evaluate(vec, payload, steps, output, probes, reward_gradients=False, fixed_terrain=None):
    env, device = vec.unwrapped, vec.device
    fixed_info = None
    if fixed_terrain is not None:
        fixed_info = assign_fixed_terrain_evaluation(env, *fixed_terrain)
    obs, _ = vec.reset()
    recorder = env.recorder_manager._terms["transition"]
    foot_kinematics = NativeLinkKinematics(
        Path(env.scene["robot"].cfg.spawn.asset_path), device, ("left_ankle_roll_link", "right_ankle_roll_link")
    )
    terrain = native_height_field(env)
    from instinctlab_supermimic.reference import REFERENCE_INPUT_DIM, ReferencePlayer, ReferenceWindows

    bank = (
        ReferenceWindows.from_state_dict(payload["reference_bank"], device)
        if payload.get("reference_bank") is not None
        else None
    )
    player = ReferencePlayer(bank, env.num_envs) if bank is not None else None
    actor, depth = (
        DepthLatentPolicy(
            REFERENCE_INPUT_DIM if bank is not None else 0, normalize_inputs=payload["cfg"].get("tgi_aligned", False)
        )
        .to(device)
        .eval(),
        DepthAutoencoder().to(device).eval(),
    )
    actor.load_state_dict(payload["modules"]["actor"])
    depth.load_state_dict(payload["modules"]["depth"])
    history = DepthHistory(camera_image(env))
    physics = None
    imagined_camera = None
    if probes or reward_gradients:
        physics = make_native_physics(
            env,
            recorder,
            z_dim=0 if payload["cfg"].get("tgi_depth", False) else 32,
            hidden=payload["cfg"]["hidden"],
            model_heightmap=payload["cfg"].get("world_model_heightmap", True),
            predict_latent=not payload["cfg"].get("imagined_camera", False),
            terrain_features=payload["cfg"].get("tgi_depth", False),
            mean_contact_loss=payload["cfg"].get("tgi_aligned", False),
        )
        physics.model.load_state_dict(payload["modules"]["world_model"])
        physics.model.eval().requires_grad_(False)
        if probes and payload["cfg"].get("imagined_camera", False):
            from .native_imagined_camera import NativeImaginedDepthCamera

            imagined_camera = NativeImaginedDepthCamera(env)
            depth.requires_grad_(False)
    reward_task = None
    if reward_gradients:
        from .native_task import NativeTaskHeads

        reward_task = NativeTaskHeads(env, physics)
    alive = torch.ones(env.num_envs, device=device, dtype=torch.bool)
    initial = recorder.state()
    terrain_columns = env.scene.terrain.terrain_types.clone()
    terrain_levels = env.scene.terrain.terrain_levels.clone()
    terrain_column_names = terrain_column_type_names(env)
    initial_origins = env.scene.env_origins.clone()
    durations = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    velocity_error = torch.zeros(env.num_envs, device=device)
    speed_sum = torch.zeros_like(velocity_error)
    reference_controls = torch.zeros_like(velocity_error)
    final = initial.clone()
    reasons = {name: torch.zeros_like(alive) for name in env.termination_manager.active_terms}
    foot_sensor_ids = [
        env.scene["contact_forces"].body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")
    ]
    previous_contact = env.scene["contact_forces"].data.current_contact_time.torch[:, foot_sensor_ids] > 0
    single_controls = torch.zeros_like(velocity_error)
    flight_controls = torch.zeros_like(velocity_error)
    touchdown_counts = torch.zeros(env.num_envs, 2, device=device)
    traces, gradient_reports = [], {}
    visualizer = env.sim.visualizers[0] if getattr(env.sim, "visualizers", None) else None
    video_env = int((terrain_levels == 3).nonzero(as_tuple=True)[0][0]) if (terrain_levels == 3).any() else 0
    for step in range(steps):
        with torch.no_grad():
            z = depth.encode(history.stack())
            features = None
            if player is not None:
                features, _, valid_reference = player.prepare(
                    env.command_manager.get_command("base_velocity"), recorder.state()[:, 13:42]
                )
                reference_controls += (valid_reference & alive).float()
            action = actor(obs[:, :768], z, features).clamp(-20, 20)
        if reward_task is not None and step in (1, 20, 60):
            from .native_reward_audit import audit_native_rewards

            summaries = {}
            for horizon in (1, 8, 32):
                audit = audit_native_rewards(
                    reward_task,
                    recorder.state().detach(),
                    action[:, None].expand(-1, horizon, -1),
                    physics.context(env.action_manager.action),
                    reward_task.context(),
                    reward_task.command(),
                )
                torch.save(audit, output / f"reward_gradients_step{step}_h{horizon}.pt")
                summaries[str(horizon)] = dict(summary=audit["summary"], scope=audit["scope"])
            (output / f"reward_gradients_step{step}.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print("REWARD_GRADIENT_AUDIT", step, flush=True)
        if probes and step in (1, 20):
            gradient_reports[str(step)] = probe_latent_physics(
                physics,
                z,
                action,
                output,
                str(step),
                imagined_camera=imagined_camera,
                perception=depth,
                depth_history=history.images,
            )
            print("LATENT_PROBE", step, json.dumps(gradient_reports[str(step)]), flush=True)
        if visualizer is not None:
            # The headless Newton visualizer otherwise keeps its default camera,
            # which can record terrain while the robot is outside the frame.
            current_state = unflatten_state(recorder.state())
            root = current_state[0]
            heading = current_state[1][:, 0, :2]
            target = root + torch.cat((heading * 0.4, root.new_tensor([0.55]).expand(len(root), 1)), -1)
            camera = root + torch.cat((-heading * 2.4, root.new_tensor([1.35]).expand(len(root), 1)), -1)
            visualizer.set_camera_view(
                tuple(camera[0].detach().cpu().tolist()), tuple(target[0].detach().cpu().tolist())
            )
        command = env.command_manager.get_command("base_velocity").clone()
        robot_data = env.scene["robot"].data
        pre_rotation = unflatten_state(recorder.state())[1]
        com_offset_b = mv(
            pre_rotation.transpose(-1, -2),
            robot_data.root_com_pos_w.torch - robot_data.root_link_pos_w.torch,
        ).clone()
        obs, _, done, _ = vec.step(action)
        if player is not None:
            player.advance(done.nonzero(as_tuple=True)[0])
        transition = recorder.take_transition()
        state = transition.next_state
        contact = transition.contact_time[:, foot_sensor_ids] > 0
        with torch.no_grad():
            sole_points = shoe_sole_points(unflatten_state(state), foot_kinematics)
            ground, ground_valid = terrain.query(sole_points[..., :2], exact=True)
            clearance = (sole_points[..., 2] - ground).amin(-1)
        single_controls += (contact[:, 0] ^ contact[:, 1]) * alive
        flight_controls += (~contact.any(-1)) * alive
        touchdown_counts += (contact & ~previous_contact) * alive[:, None]
        previous_contact = contact
        durations += alive.long()
        # World velocity transformed to the robot's heading for true per-control error.
        rotation = unflatten_state(state)[1]
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        vx = yaw.cos() * state[:, 7] + yaw.sin() * state[:, 8]
        vy = -yaw.sin() * state[:, 7] + yaw.cos() * state[:, 8]
        body_velocity = root_com_body_velocity(state, com_offset_b)
        velocity_error += (body_velocity[:, :2] - command[:, :2]).norm(dim=-1) * alive
        speed_sum += vx * alive
        final[alive] = state[alive]
        terminal = alive & done.bool()
        for name in reasons:
            reasons[name] |= terminal & env.termination_manager.get_term(name)
        traces.append(
            dict(
                state=state.cpu(),
                action=action.cpu(),
                command=command.cpu(),
                alive=alive.cpu().clone(),
                done=done.cpu(),
                foot_air_time=transition.air_time.cpu(),
                foot_contact=contact.cpu(),
                foot_clearance=clearance.cpu(),
                foot_clearance_valid=ground_valid.all(-1).cpu(),
                force=transition.force_history.cpu(),
            )
        )
        alive &= ~done.bool()
        with torch.no_grad():
            history.advance(recorder.next_depth)
            history.reset(done.nonzero(as_tuple=True)[0], camera_image(env))
        if not alive.any():
            break
    seconds = durations.float() * env.step_dt
    displacement = final[:, :3] - initial[:, :3]
    clearance_inputs = {
        name: torch.stack([row[key] for row in traces])
        for name, key in (
            ("clearance", "foot_clearance"),
            ("valid", "foot_clearance_valid"),
            ("contact", "foot_contact"),
            ("alive", "alive"),
            ("command", "command"),
        )
    }

    def clearance_report(selected):
        return swing_clearance_summary(**{key: value[:, selected] for key, value in clearance_inputs.items()})

    report = dict(
        checkpoint_iteration=payload["iteration"],
        num_envs=env.num_envs,
        controls=len(traces),
        control_dt=env.step_dt,
        exploration_std=0.0,
        terrain=env.cfg.evaluation_terrain,
        terrain_column_names=terrain_column_names,
        original_rewards=not payload["cfg"].get("terrain_locomotion", False),
        terrain_locomotion=payload["cfg"].get("terrain_locomotion", False),
        observation_noise_and_reset_randomization=(
            "observation noise retained; default standing resets, no reference-state initialization"
            if player is not None
            else "retained from training"
        ),
        motion_tracking=player is not None,
        reference_active_fraction=float(reference_controls.sum() / durations.sum()) if player is not None else 0.0,
        mean_first_episode_seconds=float(seconds.mean()),
        min_first_episode_seconds=float(seconds.min()),
        max_first_episode_seconds=float(seconds.max()),
        budget_survival_fraction=float(alive.float().mean()),
        mean_forward_displacement_m=float(displacement[:, 0].mean()),
        mean_heading_forward_velocity_m_s=float((speed_sum / durations).mean()),
        mean_true_xy_velocity_error_m_s=float((velocity_error / durations).mean()),
        velocity_error_frame="root_COM_velocity_in_full_root_link_frame",
        mean_terminal_root_height_m=float(final[:, 2].mean()),
        termination_fraction={name: float(value.float().mean()) for name, value in reasons.items()},
        env0_first_episode_seconds=float(seconds[0]),
        per_env=dict(
            seconds=seconds.tolist(),
            displacement_m=displacement.tolist(),
            terminal_state=final.tolist(),
            terrain_columns=terrain_columns.tolist(),
            terrain_levels=terrain_levels.tolist(),
            initial_origins=initial_origins.tolist(),
        ),
        gait=dict(
            single_support_fraction=float(single_controls.sum() / durations.sum()),
            flight_fraction=float(flight_controls.sum() / durations.sum()),
            per_env_single_support_fraction=(single_controls / durations).tolist(),
            per_env_flight_fraction=(flight_controls / durations).tolist(),
            per_env_touchdowns_left_right=touchdown_counts.tolist(),
            swing_clearance=clearance_report(slice(None)),
            scope="Measured instantaneous contacts during first episodes, not force-history contacts.",
        ),
        gradient_probes=gradient_reports,
        optimizer_updates=0,
        fixed_terrain=fixed_info,
    )
    report["by_terrain_column"] = {}
    for column in terrain_columns.unique().tolist():
        selected = terrain_columns == column
        report["by_terrain_column"][str(column)] = dict(
            num_envs=int(selected.sum()),
            terrain_levels=terrain_levels[selected].tolist(),
            mean_first_episode_seconds=float(seconds[selected].mean()),
            mean_body_xy_velocity_error_m_s=float((velocity_error[selected] / durations[selected]).mean()),
            single_support_fraction=float(single_controls[selected].sum() / durations[selected].sum()),
            flight_fraction=float(flight_controls[selected].sum() / durations[selected].sum()),
            swing_clearance=clearance_report(selected.cpu()),
            budget_survival_fraction=float(alive[selected].float().mean()),
            termination_fraction={name: float(value[selected].float().mean()) for name, value in reasons.items()},
        )
    report["by_terrain_type"] = {}
    for terrain_name in tuple(dict.fromkeys(terrain_column_names)):
        selected = torch.tensor(
            [terrain_column_names[int(column)] == terrain_name for column in terrain_columns.cpu().tolist()],
            device=terrain_columns.device,
            dtype=torch.bool,
        )
        if not selected.any():
            continue
        report["by_terrain_type"][terrain_name] = dict(
            terrain_columns=[i for i, name in enumerate(terrain_column_names) if name == terrain_name],
            num_envs=int(selected.sum()),
            terrain_levels=terrain_levels[selected].tolist(),
            mean_first_episode_seconds=float(seconds[selected].mean()),
            mean_body_xy_velocity_error_m_s=float((velocity_error[selected] / durations[selected]).mean()),
            single_support_fraction=float(single_controls[selected].sum() / durations[selected].sum()),
            flight_fraction=float(flight_controls[selected].sum() / durations[selected].sum()),
            swing_clearance=clearance_report(selected.cpu()),
            budget_survival_fraction=float(alive[selected].float().mean()),
            termination_fraction={name: float(value[selected].float().mean()) for name, value in reasons.items()},
        )
    torch.save(
        dict(
            initial=initial.cpu(),
            traces=traces,
            joint_names=recorder.joint_names,
            checkpoint_iteration=payload["iteration"],
            terrain=env.cfg.evaluation_terrain,
            collision_mesh_sha256=collision_mesh_hash(env),
            terrain_columns=terrain_columns.cpu(),
            terrain_levels=terrain_levels.cpu(),
            initial_origins=initial_origins.cpu(),
        ),
        output / "trace.pt",
    )
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        "LATENT_EVALUATION", json.dumps({key: value for key, value in report.items() if key != "per_env"}), flush=True
    )


def main():
    parser = cli.build_play_parser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1654)
    parser.add_argument("--probes", action="store_true")
    parser.add_argument("--reward_gradients", action="store_true", help="Frozen TGI reward derivative audit.")
    parser.add_argument("--terrain", choices=("flat", "native_hiking"), default="flat")
    parser.add_argument(
        "--terrain_type",
        help="Pin native_hiking evaluation to one generated terrain type; report each type separately.",
    )
    parser.add_argument("--terrain_level", type=int, help="Difficulty row for --terrain_type (0 is easiest).")
    parser.add_argument("--replay_trace", help="Render measured env0 poses; no policy or dynamics evaluation.")
    args, overrides = cli.parse_cli(parser, require_run_name=False)
    if (args.terrain_type is None) != (args.terrain_level is None):
        raise ValueError("--terrain_type and --terrain_level must be supplied together.")
    if args.terrain_type is not None and args.terrain != "native_hiking":
        raise ValueError("Fixed terrain evaluation is only available for --terrain native_hiking.")
    payload = None
    if not args.replay_trace:
        payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if payload["cfg"].get("paired_reference_dir"):
            raise ValueError(
                "Use instinctlab.tasks.parkour.supermimic.paired_evaluation for terrain-paired checkpoints."
            )
        if payload["format"] not in (
            "native_hiking_tgi_aligned_depth_v2",
            "native_hiking_tgi_terrain_depth_actor_v1",
            "native_hiking_single_wm_depth_latent_v1",
            "native_hiking_physics_imagined_depth_v1",
        ):
            raise ValueError("Expected a native single-model depth latent checkpoint.")
    cfg, _ = cli.resolve_train_cfg(args.task, overrides)
    configure_native_perception(cfg)
    configure_native_physics_audit(cfg)
    if args.terrain == "flat":
        configure_flat_walk(cfg)
    if payload is not None and payload["cfg"].get("imagined_camera", False):
        from .native_rewards import configure_imagined_depth_rewards

        configure_imagined_depth_rewards(cfg)
        if payload["cfg"].get("tgi_aligned", False):
            cfg.locomotion_reward_weights["locomotion_slip"] = -0.1
            cfg.locomotion_reward_weights["locomotion_support"] = -3.0
            cfg.tgi_aligned = True
    if args.reward_gradients:
        if payload is None or not payload["cfg"].get("tgi_aligned", False):
            raise ValueError("Reward gradient audit requires a TGI-aligned checkpoint.")
        cfg.terrain_locomotion = True
    if args.terrain_type is not None:
        # A fixed terrain suite must not advance rows during reset. TGI treats
        # type and difficulty as separate curriculum axes and evaluates them
        # independently.
        cfg.curriculum.terrain_levels = None
        cfg.scene.terrain.max_init_terrain_level = args.terrain_level
    cfg.evaluation_terrain = args.terrain
    if payload is not None and payload["cfg"].get("use_amp", True):
        configure_lafan_reference(cfg, "logs/hiking_data/lafan1_native_torso_v1")
    else:
        configure_without_amp(cfg)
    cfg.scene.camera.update_period = cfg.sim.dt * cfg.decimation
    cfg.recorders = NativeLatentRecorderCfg()
    cfg.scene.num_envs = 1 if args.replay_trace else (args.num_envs or 32)
    cfg.seed = args.seed
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cli.pre_launch_video_config(cfg, args, prefix=f"{args.terrain}_evaluation")
    with launch_simulation(cfg, args):
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        cli.apply_video_recording(cfg, args, str(output))
        vec = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        try:
            if args.replay_trace:
                replay_trace(vec, args.replay_trace, output)
            else:
                fixed = (args.terrain_type, args.terrain_level) if args.terrain_type is not None else None
                evaluate(vec, payload, args.steps, output, args.probes, args.reward_gradients, fixed)
        finally:
            vec.close()
