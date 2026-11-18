"""Native task equations retain frame conventions, sensor geometry and gradients."""

import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import warp as wp
from instinctlab_supermimic.physical import unflatten_state
from instinctlab_supermimic.rotations import mv, quat_to_mat

from isaaclab.utils.math import quat_apply_inverse
from isaaclab.utils.string import resolve_matching_names

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.tasks.parkour.config.parkour_env_cfg import G1Rewards, ObservationsCfg, TerminationsCfg
from instinctlab.tasks.parkour.supermimic.native_task import (
    AMP_WIDTHS,
    POLICY_WIDTHS,
    NativeTaskHeads,
    NativeTaskScene,
    flatten_history,
    single_stance_time,
    tensor_view,
)
from instinctlab.utils.warp.cylinder import CylinderSpatialGrid


def fixture():
    wp.init()
    torch.manual_seed(41)
    urdf = (
        Path(__file__).resolve().parents[1]
        / "instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"
    )
    body_names = [node.attrib["name"] for node in ET.fromstring(urdf.read_text()).findall("link")]
    native_names = list(reversed(G1_29DOF_POLICY_JOINT_ORDER_V1))
    joint_ids = list(reversed(range(29)))
    reward_cfg = vars(G1Rewards())
    termination_cfg = {k: v for k, v in vars(TerminationsCfg()).items() if not v.time_out}
    termination_cfg["base_contact"].params["sensor_cfg"].body_ids = [body_names.index("torso_link")]
    for cfg in reward_cfg.values():
        for key, entity in cfg.params.items():
            if key.endswith("cfg"):
                if entity.joint_names:
                    entity.joint_ids = resolve_matching_names(entity.joint_names, native_names)[0]
                if entity.body_names:
                    entity.body_ids = resolve_matching_names(entity.body_names, body_names)[0]
    context = dict(
        default_joint_pos=torch.zeros(2, 29),
        default_joint_vel=torch.zeros(2, 29),
        soft_joint_pos_limits=torch.tensor([-1.0, 1.0]).expand(2, 29, 2),
        soft_joint_vel_limits=torch.full((2, 29), 15.0),
        joint_effort_limits=torch.full((2, 29), 30.0),
        joint_stiffness=torch.full((2, 29), 40.0),
        body_com_pos_b=torch.randn(2, len(body_names), 3) * 0.03,
        GRAVITY_VEC_W=torch.tensor([[0.0, 0.0, -9.81]]).expand(2, -1),
    )
    robot = NS(
        body_names=body_names,
        cfg=NS(spawn=NS(asset_path=str(urdf))),
        data=NS(**{k: tensor_view(v) for k, v in context.items()}),
    )
    # Sloping terrain; a true mesh query must retain the local plane derivative.
    points = wp.array([[-5, -5, -1], [5, -5, 1], [5, 5, 1], [-5, 5, -1]], dtype=wp.vec3, device="cpu")
    mesh = wp.Mesh(points, wp.array([0, 1, 2, 0, 2, 3], dtype=wp.int32, device="cpu"))
    scanners = [
        NS(
            cfg=NS(ray_alignment="yaw", mesh_prim_paths=["ground"], max_distance=100.0),
            device="cpu",
            meshes={("ground", "cpu"): mesh},
            ray_starts=tensor_view(torch.tensor([[[-0.02, 0, 20.0], [0.10, 0, 20.0]]]).expand(2, -1, -1)),
            ray_directions=tensor_view(torch.tensor([[[0.0, 0, -1.0], [0.0, 0, -1.0]]]).expand(2, -1, -1)),
            drift=tensor_view(torch.zeros(2, 3)),
            ray_cast_drift=tensor_view(torch.zeros(2, 3)),
        )
        for _ in range(2)
    ]
    vp = NS(
        body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
        _volume_points_pattern=torch.tensor([[0.04, 0, 0.02], [0, 0, 0.01]]),
        _virtual_obstacles={},
    )
    cfg = NS(
        only_positive_lin_vel_x=True,
        velocity_control_stiffness=2.0,
        heading_control_stiffness=2.0,
        ranges=NS(ang_vel_z=(-1.0, 1.0)),
        target_dis_threshold=0.4,
        lin_vel_threshold=0.0,
        ang_vel_threshold=0.0,
    )
    command = NS(cfg=cfg)
    scene = NativeTaskScene(
        robot=robot, left_height_scanner=scanners[0], right_height_scanner=scanners[1], leg_volume_points=vp
    )
    env = NS(
        scene=scene,
        device="cpu",
        step_dt=0.02,
        command_manager=NS(get_term=lambda _: command),
        reward_manager=NS(active_terms={"rewards": list(reward_cfg)}, get_term_cfg=lambda name, _: reward_cfg[name]),
        termination_manager=NS(active_terms=list(termination_cfg), get_term_cfg=lambda name: termination_cfg[name]),
        cfg=NS(rewards=NS(rewards=G1Rewards()), observations=ObservationsCfg()),
    )
    physics = NS(joint_ids=joint_ids, contact_body_names=body_names)
    task = NativeTaskHeads(env, physics)
    packed = torch.zeros(2, 71)
    packed[:, 2:4] = 1
    packed[:, 10:13] = torch.tensor([0.2, -0.1, 0.3])
    packed[:, 13:42] = torch.randn(2, 29) * 0.1
    packed[:, 42:] = torch.randn(2, 29) * 0.1
    return task, unflatten_state(packed), context


def test_com_and_origin_velocities_are_distinct_and_observations_match_native_equations():
    task, state, context = fixture()
    data = task.data(state, context)
    rotation, w = state[1], state[3]
    expected_root = state[2] + torch.linalg.cross(w, mv(rotation, context["body_com_pos_b"][:, task.root_id]))
    torch.testing.assert_close(data.root_lin_vel_b.torch, mv(rotation.transpose(-1, -2), expected_root))
    assert not torch.equal(data.body_lin_vel_w.torch, data.body_link_lin_vel_w.torch)
    body_quat = data.body_quat_w.torch
    body_rotation = quat_to_mat(body_quat[..., [3, 0, 1, 2]])
    expected_body = data.body_link_lin_vel_w.torch + torch.linalg.cross(
        data.body_link_ang_vel_w.torch, mv(body_rotation, context["body_com_pos_b"])
    )
    torch.testing.assert_close(data.body_lin_vel_w.torch, expected_body)
    torch.testing.assert_close(
        data.projected_gravity_b.torch,
        quat_apply_inverse(data.root_quat_w.torch, torch.nn.functional.normalize(context["GRAVITY_VEC_W"], dim=-1)),
    )
    action, command = torch.randn(2, 29), torch.randn(2, 3)
    frame = task.proprio(state, action, command, context)
    torch.testing.assert_close(frame[:, 9:38], state[4])
    torch.testing.assert_close(frame[:, 38:67], state[5] * 0.05)
    torch.testing.assert_close(frame[:, -29:], action)
    amp = task.amp_frame(state, context)
    torch.testing.assert_close(amp[:, 61:64], data.root_lin_vel_b.torch)
    torch.testing.assert_close(amp[:, 64:], data.root_ang_vel_b.torch)
    assert flatten_history(frame[:, None].expand(-1, 8, -1), POLICY_WIDTHS).shape == (2, 768)
    assert flatten_history(amp[:, None].expand(-1, 10, -1), AMP_WIDTHS).shape == (2, 670)


def test_native_mesh_foot_rays_keep_horizontal_offset_and_plane_gradient():
    task, state, context = fixture()
    position = state[0].clone().requires_grad_(True)
    data = task.data((position, *state[1:]), context)
    heights = task.foot_heights(data, task.sensor_context())
    feet = data.body_pos_w.torch[:, task.foot_ids]
    foot_rotation = quat_to_mat(data.body_quat_w.torch[:, task.foot_ids][..., [3, 0, 1, 2]])
    for i, height in enumerate(heights):
        yaw = torch.atan2(foot_rotation[:, i, 1, 0], foot_rotation[:, i, 0, 0])
        expected = 0.2 * (feet[:, i, 0:1] + yaw.cos()[:, None] * torch.tensor([[-0.02, 0.10]]))
        torch.testing.assert_close(height, expected, atol=3e-6, rtol=1e-5)
    gradient = torch.autograd.grad(sum(x.sum() for x in heights), position)[0]
    torch.testing.assert_close(gradient, torch.tensor([[0.8, 0.0, 0.0]]).expand_as(gradient), atol=1e-6, rtol=1e-5)


def test_native_rewards_use_latched_command_true_auxiliary_and_terminal_alive_flag():
    task, state, context = fixture()
    action, previous, velocity = torch.randn(2, 29), torch.zeros(2, 29), torch.tensor([[0.7, 0, 0.2]]).expand(2, -1)
    auxiliary = torch.cat((torch.full((2, 29), 2.0), torch.full((2, 29), 30.0)), -1).requires_grad_(True)
    forces = torch.zeros(2, 3, len(task.body_names), 3)
    forces[:, 0, task.foot_contact_ids[0], 2] = 2
    forces[:, 0, task.body_names.index("torso_link"), 2] = 5
    air, contact = torch.zeros(2, len(task.body_names)), torch.zeros(2, len(task.body_names))
    # Single stance is based on the timers, including when force history disagrees.
    air[:, task.foot_contact_ids[1]] = 0.10
    contact[:, task.foot_contact_ids[0]] = 0.08
    total, terms = task.reward(
        state,
        action,
        previous,
        velocity,
        context,
        auxiliary,
        forces,
        torch.zeros_like(forces),
        air,
        contact,
        torch.tensor([False, True]),
        task.sensor_context(),
    )
    assert len(terms) == 26
    torch.testing.assert_close(terms["is_alive"], torch.tensor([0.06, 0.0]))
    torch.testing.assert_close(terms["feet_air_time"], torch.full((2,), 0.08 * 0.5 * 0.02))
    torch.testing.assert_close(terms["undesired_contacts"], torch.zeros(2))
    torch.testing.assert_close(terms["dof_acc_l2"], torch.full((2,), 29 * 30**2 * -1.25e-7 * 0.02))
    grad = torch.autograd.grad(total.sum(), auxiliary)[0]
    assert torch.isfinite(grad).all() and grad[:, :29].norm() > 0 and grad[:, 29:].norm() > 0


def test_volume_points_use_link_origin_twists_and_inward_penetration_derivative():
    task, state, context = fixture()
    p = state[0].clone().requires_grad_(True)
    data = task.data((p, *state[1:]), context)
    original = task.volume_points(data)
    foot = data.body_link_pos_w.torch[0, task.vp_ids[0]].detach()
    # A vertical cylinder near the left foot, avoiding its singular centre.
    cylinder = torch.tensor([[foot[0] - 0.12, foot[1], -2.0, foot[0] - 0.12, foot[1], 2.0, 0.3]])
    task.vp_sensor._virtual_obstacles["test"] = CylinderSpatialGrid(cylinder, num_grid_cells=64, device="cpu")
    points = task.volume_points(data)
    torch.testing.assert_close(points.points_vel_w.torch, original.points_vel_w.torch)
    value = points.penetration_offset.torch.norm(dim=-1).sum()
    grad = torch.autograd.grad(value, p)[0]
    assert torch.isfinite(grad).all() and grad[0, 0] < 0
    values = []
    for sign in (-1, 1):
        shifted = p.detach().clone()
        shifted[:, 0] += sign * 1e-3
        predicted = task.volume_points(task.data((shifted, *state[1:]), context))
        values.append(predicted.penetration_offset.torch.norm(dim=-1).flatten(1).sum(-1))
    numerical = (values[1] - values[0]) / 0.002
    torch.testing.assert_close(grad[:, 0], numerical, atol=2e-4, rtol=2e-3)


def test_zero_tilt_reward_backward_is_finite_and_timer_modes_are_separate():
    task, state, context = fixture()
    zero = list(state)
    zero[4] = torch.zeros_like(zero[4], requires_grad=True)
    d = task.data(tuple(zero), context)
    forces = torch.zeros(2, 3, len(task.body_names), 3)
    forces[:, :, task.foot_contact_ids, 2] = 2
    scene = NativeTaskScene(robot=NS(data=d), contact_forces=NS(data=NS(net_forces_w_history=tensor_view(forces))))
    cfg = task.reward_cfg["feet_flat_ori"]
    value = cfg.func(NS(scene=scene), **cfg.params)
    value.sum().backward()
    assert torch.isfinite(zero[4].grad).all()
    timers = torch.tensor([[0.0, 0.1, 0.08, 0.0], [0.0, 0.0, 0.0, 0.0]])
    torch.testing.assert_close(single_stance_time(timers), torch.tensor([0.08, 0.0]))
    advanced = task.advance_feet_timers(timers, torch.tensor([[0.0, 1.0], [1.0, 1.0]]))
    torch.testing.assert_close(advanced, torch.tensor([[0.02, 0.0, 0.0, 0.02], [0.0, 0.0, 0.02, 0.02]]))


def test_pose_command_matches_native_rule_including_standing_random_and_near_goal():
    from instinctlab.tasks.parkour.mdp.commands.pose_velocity_command import PoseVelocityCommand

    task, state, context = fixture()
    data = task.data(state, context)
    term = NS(
        cfg=task.command_term.cfg,
        robot=NS(data=data),
        pos_command_w=state[0].clone(),
        progress_start_pos_w=state[0].clone(),
        max_command_b=torch.tensor([[0.7, 0.2, 0.4], [0.8, 0.1, 0.4]]),
        is_standing_env=torch.tensor([False, True]),
        random_velocity_indices=torch.tensor([False, False]),
        random_lin_vel_x=torch.tensor([0.3, 0.4]),
        random_lin_vel_y=torch.tensor([0.0, 0.1]),
        random_ang_vel_z=torch.tensor([-0.6, 0.0]),
        vel_command_b=torch.zeros(2, 3),
        pos_command_b=torch.zeros(2, 3),
    )
    term.pos_command_w[:, :2] += torch.tensor([[2.0, 1.0], [-2.0, -1.0]])
    for near, random in ((False, False), (True, False), (False, True)):
        if near:
            term.pos_command_w = state[0] + torch.tensor([0.1, 0.0, 0.0])
        term.random_velocity_indices[:] = random
        PoseVelocityCommand._update_command(term)
        command = dict(
            target=term.pos_command_w,
            maximum=term.max_command_b,
            standing=term.is_standing_env,
            random=term.random_velocity_indices,
            random_velocity=torch.stack((term.random_lin_vel_x, term.random_lin_vel_y, term.random_ang_vel_z), -1),
        )
        torch.testing.assert_close(task.velocity_command(state, command), term.vel_command_b)


def test_dense_progress_is_per_transition_telescopes_and_has_correct_gradient():
    from instinctlab.tasks.parkour.supermimic.native_rewards import goal_progress

    previous = torch.tensor([[0.0, 0.0, 0.9]])
    current = torch.tensor([[0.01, 0.0, 0.9]], requires_grad=True)
    command = NS(
        pos_command_w=torch.tensor([[2.0, 0.0, 0.9]]),
        progress_start_pos_w=previous,
        is_standing_env=torch.tensor([False]),
        random_velocity_indices=torch.tensor([False]),
    )
    env = NS(
        scene={"robot": NS(data=NS(root_pos_w=tensor_view(current)))},
        command_manager=NS(get_term=lambda name: command),
        step_dt=0.02,
    )
    rate = goal_progress(env, "base_velocity")
    torch.testing.assert_close(rate, torch.tensor([0.5]))
    gradient = torch.autograd.grad(rate.sum(), current)[0]
    torch.testing.assert_close(gradient, torch.tensor([[50.0, 0.0, 0.0]]))
    total = 0.0
    for x in (0.01, 0.02, 0.03):
        env.scene["robot"].data.root_pos_w = tensor_view(torch.tensor([[x, 0.0, 0.9]]))
        total += float(goal_progress(env, "base_velocity") * env.step_dt)
        command.progress_start_pos_w = torch.tensor([[x, 0.0, 0.9]])
    assert abs(total - 0.03) < 1e-6
    # An autoreset/resampled target establishes a new pre-step snapshot.
    command.pos_command_w = torch.tensor([[12.0, 0.0, 0.9]])
    command.progress_start_pos_w = torch.tensor([[10.0, 0.0, 0.9]])
    env.scene["robot"].data.root_pos_w = tensor_view(torch.tensor([[9.99, 0.0, 0.9]]))
    assert goal_progress(env, "base_velocity") < 0
    command.is_standing_env[:] = True
    assert goal_progress(env, "base_velocity") == 0
    command.is_standing_env[:] = False
    command.random_velocity_indices[:] = True
    assert goal_progress(env, "base_velocity") == 0


def test_smooth_wait_has_finite_forward_speed_gradient_and_no_standing_pressure():
    from instinctlab.tasks.parkour.supermimic.native_rewards import smooth_dont_wait

    velocity = torch.tensor([[0.0, 0, 0], [0.1, 0, 0], [0.3, 0, 0]], requires_grad=True)
    command = torch.tensor([[0.5, 0, 0]]).expand(3, -1).clone()
    env = NS(
        scene={"robot": NS(data=NS(root_lin_vel_b=tensor_view(velocity)))},
        command_manager=NS(get_command=lambda name: command),
    )
    value = smooth_dont_wait(env, "base_velocity")
    assert value[0] > value[1] > value[2]
    gradient = torch.autograd.grad((-0.5 * value).sum(), velocity)[0]
    assert torch.isfinite(gradient).all() and (gradient[:, 0] > 0).all()
    eps = 1e-4
    with torch.no_grad():
        velocity[:, 0] += eps
        plus = smooth_dont_wait(env, "base_velocity")
        velocity[:, 0] -= 2 * eps
        minus = smooth_dont_wait(env, "base_velocity")
    torch.testing.assert_close(gradient[:, 0], -0.5 * (plus - minus) / (2 * eps), atol=2e-3, rtol=2e-3)
    command.zero_()
    assert (smooth_dont_wait(env, "base_velocity") == 0).all()


def test_imagined_progress_uses_each_predicted_previous_position():
    from isaaclab.managers import RewardTermCfg

    from instinctlab.tasks.parkour.supermimic.native_rewards import goal_progress

    task, actor, perception, seed = imagination_fixture()
    task.reward_cfg["goal_progress"] = RewardTermCfg(
        func=goal_progress, weight=4.0, params=dict(command_name="base_velocity", target_radius=0.4)
    )
    seed.task_context.update(
        goal_target=seed.command["target"],
        progress_start_position=seed.state[:, :3],
        goal_standing=seed.command["standing"],
        goal_random=seed.command["random"],
    )
    from instinctlab.tasks.parkour.supermimic.native_imagination import imagine_native

    result = imagine_native(task, actor, perception, seed, horizon=8, add_noise=False)
    distance = (result["states"][..., :2] - seed.command["target"][:, None, :2]).norm(dim=-1)
    potential = (distance - 0.4).clamp_min(0)
    expected = 4.0 * (potential[:, :-1] - potential[:, 1:])
    torch.testing.assert_close(result["reward_terms"]["goal_progress"], expected)
    torch.testing.assert_close(expected.sum(-1), 4.0 * (potential[:, 0] - potential[:, -1]))
    gradient = torch.autograd.grad(result["reward_terms"]["goal_progress"].sum(), tuple(actor.parameters()))
    assert all(torch.isfinite(g).all() for g in gradient)
    assert sum(float(g.norm()) for g in gradient) > 0


def test_soft_contact_rewards_match_every_native_term_at_hard_modes_and_keep_gradients():
    task, state, context = fixture()
    bodies = len(task.body_names)
    total = torch.zeros(2, bodies)
    total[:, task.foot_contact_ids[0]] = 1
    normal = total.clone()
    normal[:, task.body_names.index("torso_link")] = 1
    contacts = torch.cat((total, normal, torch.tensor([[1.0, 0.0]]).expand(2, -1)), -1).requires_grad_(True)
    zeros = torch.zeros_like(total)
    forces = torch.stack((zeros, zeros, 2 * total), -1)[:, None]
    normal_forces = torch.stack((zeros, zeros, 2 * normal), -1)[:, None]
    timers = torch.tensor([[0.0, 0.1, 0.08, 0.0]]).expand(2, -1)
    air, contact = torch.zeros_like(total), torch.zeros_like(total)
    air[:, task.foot_contact_ids] = timers[:, :2]
    contact[:, task.foot_contact_ids] = timers[:, 2:]
    action, previous = torch.zeros(2, 29), torch.zeros(2, 29)
    velocity, auxiliary = torch.tensor([[0.7, 0, 0.2]]).expand(2, -1), torch.ones(2, 58, requires_grad=True)
    terminated = torch.tensor([False, True])
    hard, hard_terms = task.reward(
        state,
        action,
        previous,
        velocity,
        context,
        auxiliary,
        forces,
        normal_forces,
        air,
        contact,
        terminated,
        task.sensor_context(),
    )
    soft, soft_terms = task.soft_reward(
        state,
        action,
        previous,
        velocity,
        context,
        auxiliary,
        contacts,
        timers,
        terminated.float(),
        task.sensor_context(),
    )
    for name in hard_terms:
        torch.testing.assert_close(soft_terms[name], hard_terms[name], msg=lambda msg: f"{name}: {msg}")
    torch.testing.assert_close(soft, hard)
    contact_grad, aux_grad = torch.autograd.grad(soft.sum(), (contacts, auxiliary))
    assert torch.isfinite(contact_grad).all() and torch.isfinite(aux_grad).all()
    assert contact_grad[:, :bodies].norm() > 0 and contact_grad[:, bodies : 2 * bodies].norm() > 0
    assert aux_grad.norm() > 0


def imagination_fixture():
    from instinctlab_perception.policy import PerceptivePolicy
    from instinctlab_perception.rssm import PerceptionCfg, PerceptionModel
    from instinctlab_supermimic.networks import WorldModel
    from instinctlab_supermimic.physical import flatten_state

    from instinctlab.tasks.parkour.supermimic.native_imagination import NativeImaginationStart

    task, state, context = fixture()
    task.physics.model = WorldModel(3, 2 * len(task.body_names) + 2, 30, hidden=(32, 32), aux_dim=58)
    task.physics.scan = lambda s: s[0][..., 2:3].expand(-1, 3)
    task.physics.model_scan = lambda s: task.physics.scan(s)
    task.physics.scan_with_validity = lambda s: (task.physics.scan(s), torch.ones(len(s[0]), dtype=torch.bool))
    with torch.no_grad():
        task.physics.model.net[-1].weight.normal_(std=0.003)
    perception = PerceptionModel(PerceptionCfg(deter=16, stoch=4, categories=4, embed=16))
    actor = PerceptivePolicy(torch.nn.Linear(896, 29), feature_dim=perception.cfg.feature_dim)
    previous, velocity = torch.zeros(2, 29), torch.tensor([[0.7, 0, 0.2]]).expand(2, -1)
    command = dict(
        target=state[0] + torch.tensor([5.0, 0.0, 0.0]),
        maximum=torch.tensor([[0.7, 0, 1.0]]).expand(2, -1),
        standing=torch.zeros(2, dtype=torch.bool),
        random=torch.zeros(2, dtype=torch.bool),
        random_velocity=torch.zeros(2, 3),
        velocity=velocity,
        time_left=torch.full((2,), 10.0),
        origin=torch.zeros(2, 3),
    )
    seed = NativeImaginationStart(
        flatten_state(state),
        previous,
        torch.zeros(2, 4),
        torch.zeros(2, 30),
        context,
        command,
        task.sensor_context(),
        task.proprio(state, previous, velocity, context)[:, None].expand(-1, 8, -1),
        task.amp_frame(state, context)[:, None].expand(-1, 10, -1),
        perception.initial(2, sample=False),
        torch.zeros(2, dtype=torch.long),
    )
    return task, actor, perception, seed


def test_native_imagination_physical_bptt_and_partial_perception_interval():
    from instinctlab.tasks.parkour.supermimic.native_imagination import frozen_parameters, imagine_native

    task, actor, perception, seed = imagination_fixture()
    actor_outputs = []
    hook = actor.register_forward_hook(lambda module, inputs, output: actor_outputs.append(output))
    try:
        with frozen_parameters(task.physics.model, perception):
            imagined = imagine_native(task, actor, perception, seed, horizon=7, add_noise=False)
            assert imagined["states"].shape == (2, 8, 71)
            assert imagined["amp_windows"].shape == (2, 7, 670)
            assert imagined["perception_prior_ticks"] == 1
            assert imagined["phases"][0].tolist() == [0, 1, 2, 3, 4, 0, 1, 2]
            torch.testing.assert_close(imagined["pending"][:, -1, :2], imagined["actions"][:, 5:7])
            torch.testing.assert_close(imagined["pending"][:, -1, 2:], torch.zeros(2, 3, 29))
            assert not imagined["beliefs"].requires_grad
            torch.testing.assert_close(imagined["beliefs"][:, 4], imagined["beliefs"][:, 0])
            assert not torch.equal(imagined["beliefs"][:, 5], imagined["beliefs"][:, 0])
            # The bootstrap sees pending action slots and the final phase.
            extra = imagined["value_features"][..., -150:]
            torch.testing.assert_close(extra[:, -1, :145], imagined["pending"][:, -1].flatten(1))
            torch.testing.assert_close(extra[:, -1, 145:], torch.tensor([[0.0, 0, 1, 0, 0]]).expand(2, -1))
            derivative = torch.autograd.grad(
                imagined["states"][:, -1, 13:42].square().sum(), actor_outputs[0], retain_graph=True
            )[0]
            assert torch.isfinite(derivative).all() and derivative.norm() > 0
            (-imagined["task_rewards"].sum()).backward()
            assert all(torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None)
            assert sum(p.grad.norm() for p in actor.parameters() if p.grad is not None) > 0
            assert all(p.grad is None for p in task.physics.model.parameters())
            assert all(p.grad is None for p in perception.parameters())
    finally:
        hook.remove()
    assert all(p.requires_grad for p in task.physics.model.parameters())


def test_unsupported_imagination_absorbs_only_invalid_rows_and_keeps_valid_gradients():
    from instinctlab.tasks.parkour.supermimic.native_imagination import imagine_native

    task, actor, perception, seed = imagination_fixture()
    baseline = imagine_native(task, actor, perception, seed, horizon=4, add_noise=False)
    original_step = task.physics.model.step_with_aux
    original_scan = task.physics.scan

    def scan(state):
        assert (state[0][:, 0].abs() < 100).all(), "Unsupported state reached a strict terrain query."
        return original_scan(state)

    def step(*args):
        state, logits, auxiliary_logits, auxiliary = original_step(*args)
        offset = torch.zeros_like(state[0])
        offset[0, 0] = 1000
        return (state[0] + offset, *state[1:]), logits, auxiliary_logits, auxiliary

    task.physics.scan = scan
    task.physics.scan_with_validity = lambda s: (original_scan(s), s[0][:, 0].abs() < 100)
    task.physics.model.step_with_aux = step
    result = imagine_native(task, actor, perception, seed, horizon=4, add_noise=False)
    assert not result["terrain_supported"][0].any()
    assert result["terrain_supported"][1].all()
    torch.testing.assert_close(result["states"][0], seed.state[0].expand(5, -1))
    torch.testing.assert_close(result["states"][1], baseline["states"][1])
    assert (result["terminations"][0] == 1).all()
    assert (result["task_rewards"][0] == 0).all()
    for terms in result["reward_terms"].values():
        assert (terms[0] == 0).all()
    gradient = torch.autograd.grad(result["task_rewards"][1].sum(), tuple(actor.parameters()))
    assert all(torch.isfinite(g).all() for g in gradient)
    assert sum(float(g.norm()) for g in gradient) > 0


def test_critic_input_distinguishes_policy_and_amp_history():
    from dataclasses import replace

    from instinctlab.tasks.parkour.supermimic.native_imagination import imagine_native

    task, actor, perception, seed = imagination_fixture()
    with torch.no_grad():
        original = imagine_native(task, actor, perception, seed, horizon=1, add_noise=False)
        for name, offset in (("policy_history", 0), ("amp_history", 768)):
            history = getattr(seed, name).clone()
            history[:, 1, 0] += 0.05
            changed = imagine_native(
                task, actor, perception, replace(seed, **{name: history}), horizon=1, add_noise=False
            )
            width = 768 if name == "policy_history" else 670
            physical_dim = original["physical_features"].shape[-1]
            start = physical_dim + offset
            assert not torch.equal(original["value_features"][:, 0], changed["value_features"][:, 0])
            expected = flatten_history(history, POLICY_WIDTHS if name == "policy_history" else AMP_WIDTHS)
            torch.testing.assert_close(changed["value_features"][:, 0, start : start + width], expected)
            torch.testing.assert_close(original["physical_features"][:, 0], changed["physical_features"][:, 0])


@pytest.mark.parametrize("use_amp", [False, True])
def test_native_actor_critic_updates_do_not_update_p_f_style_or_failure(use_amp):
    from instinctlab_supermimic.networks import Critic, FailureHead

    from instinctlab.tasks.parkour.supermimic.native_imagination import frozen_parameters, imagine_native
    from instinctlab.tasks.parkour.supermimic.native_policy import NativePolicyUpdateCfg, update_native_policy

    task, actor, perception, seed = imagination_fixture()
    with torch.no_grad():
        example = imagine_native(task, actor, perception, seed, horizon=1, add_noise=False)
    critic = Critic(example["value_features"].shape[-1])
    failure = FailureHead(example["physical_features"].shape[-1])
    style = NS(discriminator=torch.nn.Linear(670, 1))
    style.reward = lambda windows: 1 - 0.25 * (style.discriminator(windows).squeeze(-1) - 1).square()
    modules = dict(
        policy=actor,
        critic=critic,
        perception=perception,
        physical=task.physics.model,
        style=style.discriminator,
        failure=failure,
    )
    before = {name: [p.detach().clone() for p in module.parameters()] for name, module in modules.items()}
    actor_opt, critic_opt = torch.optim.Adam(actor.parameters(), lr=1e-4), torch.optim.Adam(
        critic.parameters(), lr=1e-4
    )
    updates = dict(policy=0, critic=0)

    def count(name):
        def hook(*args):
            updates[name] += 1

        return hook

    actor_opt.register_step_post_hook(count("policy"))
    critic_opt.register_step_post_hook(count("critic"))
    report = update_native_policy(
        task,
        actor,
        perception,
        critic,
        style if use_amp else None,
        seed,
        actor_opt,
        critic_opt,
        NativePolicyUpdateCfg(horizon=7, critic_epochs=2, critic_minibatches=3, use_failure_head=True),
        failure,
    )
    assert updates == dict(policy=1, critic=6)
    assert report["amp_enabled"] == use_amp
    if not use_amp:
        assert report["style_reward"] == 0.0
    assert report["policy_updates"] == 1 and report["critic_updates"] == 6
    assert report["final_perception_phase"] == 2 and report["perception_prior_ticks"] == 1
    for name, module in modules.items():
        changed = any(not torch.equal(p, original) for p, original in zip(module.parameters(), before[name]))
        assert changed == (name in ("policy", "critic")), name
        assert all(torch.isfinite(p).all() for p in module.parameters())
    # Failure of an imagination call must also restore originally frozen flags.
    parameter = next(task.physics.model.parameters())
    parameter.requires_grad_(False)
    try:
        with frozen_parameters(task.physics.model):
            raise RuntimeError("audit")
    except RuntimeError:
        pass
    assert not parameter.requires_grad


def test_native_termination_uses_normal_contact_and_clamped_origin_height():
    task, state, _ = fixture()
    bodies = len(task.body_names)
    torso = task.body_names.index("torso_link")
    contacts = torch.zeros(2, 2 * bodies + 2)
    contacts[:, torso] = 1  # Total force alone does not drive illegal_contact.
    command = dict(origin=torch.tensor([[0.0, 0, 20.0], [0.0, 0, -1.0]]))
    probability = task.soft_termination(state, command, contacts)
    assert (probability < 1e-4).all()
    contacts[:, bodies + torso] = 1
    torch.testing.assert_close(task.soft_termination(state, command, contacts), torch.ones(2))
    contacts.zero_()
    position = state[0].clone()
    position[:, 2] = 0.4
    position.requires_grad_(True)
    rotation = state[1].clone().requires_grad_(True)
    probability = task.soft_termination((position, rotation, *state[2:]), command, contacts)
    assert probability[0] > 0.99 and probability[1] < 1e-4
    gradients = torch.autograd.grad(probability.sum(), (position, rotation))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert gradients[0][0, 2] < 0


def test_native_hiking_scanners_respect_ground_mesh_selection_after_backend_resolution():
    from isaaclab.sensors import RayCasterCfg

    from instinctlab.tasks.parkour.config.g1.g1_parkour_target_amp_cfg import G1ParkourEnvCfg
    from instinctlab.tasks.utils.backends import select_backend

    for backend in ("newton", "physx"):
        cfg = select_backend(G1ParkourEnvCfg(), backend)
        for scanner in (cfg.scene.left_height_scanner, cfg.scene.right_height_scanner):
            assert scanner.mesh_prim_paths == ["/World/ground"]
            assert scanner.offset.pos == (0.04, 0.0, 20.0)
            if backend == "newton":
                assert scanner.class_type == "isaaclab_newton.sensors.ray_caster.legacy_ray_caster:LegacyRayCaster"
            else:
                assert scanner.class_type == RayCasterCfg().class_type


@pytest.mark.parametrize("real_critic_only", [False, True])
def test_full_native_update_on_recorded_fixture_steps_all_six_optimizers(tmp_path, real_critic_only):
    from instinctlab_perception.events import PerceptionRuntime
    from instinctlab_perception.replay import PerceptionReplay
    from instinctlab_supermimic.networks import Critic, FailureHead
    from instinctlab_supermimic.style import StyleReward

    from instinctlab.tasks.parkour.supermimic.native_physics import NativePhysics
    from instinctlab.tasks.parkour.supermimic.native_policy import NativePolicyUpdateCfg
    from instinctlab.tasks.parkour.supermimic.native_runner import NativeDualRunner, NativeRunnerCfg
    from instinctlab.tasks.parkour.supermimic.native_training_replay import NativeTrainingReplay

    task, actor, perception, seed = imagination_fixture()
    physics = task.physics
    physics.foot_loss_weight = 1.0
    physics.foot_kinematics = NS(
        chain=task.kin.chain,
        order=task.kin.order,
        tracked_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
    )
    physics.foot_points = lambda state: NativePhysics.foot_points(physics, state)
    physics.error_floor, physics.auxiliary_floor = torch.full((70,), 1e-3), torch.ones(58)
    physics.window_loss = lambda **batch: NativePhysics.window_loss(physics, **batch)
    physics.recorder = NS(state=lambda: seed.state)
    task.context, task.command = lambda: seed.task_context, lambda: seed.command
    task.env.action_manager = NS(action=seed.previous_action.clone())
    zeros = torch.zeros(2, len(task.body_names))
    task.env.scene["contact_forces"] = NS(
        data=NS(current_air_time=tensor_view(zeros), current_contact_time=tensor_view(zeros))
    )
    runner = NativeDualRunner.__new__(NativeDualRunner)
    runner.env, runner.task, runner.physics = task.env, task, physics
    runner.contract = dict(num_envs=2, terrain_sha256="fixture_plane")
    runner.actor, runner.perception = actor, perception
    runner.cfg = NativeRunnerCfg(
        physical_updates=1,
        style_updates=1,
        model_only_iterations=0,
        perception_batch=2,
        perception_sequence=3,
        physical_batch=4,
        physical_horizon=3,
        physical_foot_weight=1.0,
        actor_batch=2,
        real_batch=4,
        real_value_updates=2 if real_critic_only else 0,
        real_value_horizon=5,
        real_value_batch=2,
        policy=NativePolicyUpdateCfg(
            horizon=5 if real_critic_only else 7,
            critic_epochs=0 if real_critic_only else 2,
            critic_minibatches=3,
            use_failure_head=True,
        ),
    )
    runner.runtime = PerceptionRuntime(perception, 2)
    runner.perception_replay, runner.replay = PerceptionReplay(2, holdout_fraction=0.5), NativeTrainingReplay(2)
    physical_dim = task.critic_features(
        unflatten_state(seed.state), seed.command, seed.timers, seed.physical_context
    ).shape[-1]
    runner.critic = Critic(physical_dim + 768 + 670 + perception.cfg.feature_dim + 150)
    runner.failure = FailureHead(physical_dim)
    runner.style = StyleReward(
        {"amp": (670,)},
        "cpu",
        dict(hidden_sizes=[32, 16], nonlinearity="ReLU"),
        optimizer_kwargs=dict(lr=1e-4),
        gradient_penalty_coef=5.0,
    )
    runner.modules = dict(
        perception=perception,
        physical=physics.model,
        policy=actor,
        critic=runner.critic,
        failure=runner.failure,
        style=runner.style.discriminator,
    )
    runner.optimizers = {
        name: torch.optim.Adam(module.parameters(), lr=1e-4)
        for name, module in runner.modules.items()
        if name != "style"
    }
    runner.optimizers["style"] = runner.style.optimizer
    runner.update_counts, runner.iteration = dict.fromkeys(runner.modules, 0), 0
    for name, opt in runner.optimizers.items():
        opt.register_step_post_hook(lambda optimizer, args, kwargs, key=name: runner._count(key))
    for step in range(11):
        _, tick = runner.runtime.before_action(
            torch.zeros(2, 1, 18, 32), torch.zeros(2, 64), runner.runtime.control_step.remainder(5) == 0
        )
        if tick is not None:
            runner.perception_replay.add(tick)
        previous = task.env.action_manager.action
        context = torch.cat((previous, torch.zeros(2, 1)), -1)
        runner.replay.add_starts(
            tick,
            task,
            context,
            flatten_history(seed.policy_history, POLICY_WIDTHS),
            flatten_history(seed.amp_history, AMP_WIDTHS),
        )
        action = torch.randn(2, 29) * 0.1
        following = seed.state.clone()
        following[:, 13:] += 1e-3
        transition = NS(
            state=seed.state,
            next_state=following,
            action=action,
            episode_step=runner.runtime.control_step.clone(),
            terminated=torch.zeros(2, dtype=torch.bool),
            truncated=torch.zeros(2, dtype=torch.bool),
        )
        next_context = torch.cat((action, context[:, 29:]), -1)
        learning = dict(
            policy_window=flatten_history(seed.amp_history, AMP_WIDTHS),
            reference_window=torch.randn(2, 670) * 0.1,
            command=seed.command,
            next_timers=seed.timers,
            next_features=task.critic_features(
                unflatten_state(following), seed.command, seed.timers, next_context
            ).detach(),
        )
        runner.replay.add_transition(
            transition,
            context,
            runner.runtime.episode_id,
            torch.zeros(2, physics.model.num_contact),
            torch.zeros(2, 58),
            learning,
            task_reward=torch.ones(2) * 0.1,
        )
        runner.runtime.after_action(action)
        task.env.action_manager.action = action
    before = {name: [p.detach().clone() for p in module.parameters()] for name, module in runner.modules.items()}
    report = runner.update()
    assert report["actual_updates"] == dict(
        perception=1, physical=1, style=1, failure=1, policy=1, critic=2 if real_critic_only else 6
    )
    assert (report["real_value"] is not None) == real_critic_only
    for name, module in runner.modules.items():
        assert any(not torch.equal(p, old) for p, old in zip(module.parameters(), before[name])), name
        assert all(torch.isfinite(p).all() for p in module.parameters())
    assert report["policy"]["final_perception_phase"] == (0 if real_critic_only else 2)
    assert report["iteration"] == 1
    assert report["evaluation"]["train"]["environments"] == 1
    assert report["evaluation"]["holdout"]["environments"] == 1
    # Evaluation must not leak held-out data into normalizers, modify any
    # parameter/buffer, take optimizer steps, or perturb future training RNG.
    states = {name: {k: v.clone() for k, v in module.state_dict().items()} for name, module in runner.modules.items()}
    counts = runner.update_counts.copy()
    rng = torch.get_rng_state().clone()
    evaluation = runner.evaluate_models()
    assert torch.equal(torch.get_rng_state(), rng)
    assert runner.update_counts == counts
    for name, module in runner.modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, states[name][key], rtol=0, atol=0)
    for split in ("train", "holdout"):
        assert evaluation[split]["failure_samples"] == runner.cfg.real_batch
        assert evaluation[split]["perception"]["valid_ticks"] > 0
        assert evaluation[split]["perception"]["observed_ticks"] > 0
        assert all(torch.isfinite(torch.tensor(value)) for value in evaluation[split]["perception"].values())
        assert evaluation[split]["actual_failure_rate"] == 0
        assert evaluation[split]["physical_valid_transitions"] > 0
        for branch in evaluation[split]["failure"].values():
            assert 0 <= branch["predicted_rate"] <= 1
            assert 0 <= branch["brier"] <= 1
    checkpoint = tmp_path / "native_dual.pt"
    runner.save(checkpoint)
    saved_actor = [p.detach().clone() for p in actor.parameters()]
    with torch.no_grad():
        next(actor.parameters()).add_(1.0)
    runner.replay = NativeTrainingReplay(2)
    runner.runtime = PerceptionRuntime(perception, 2)
    runner.update_counts = dict.fromkeys(runner.modules, 999)
    runner.contract["terrain_sha256"] = "different_plane"
    import pytest

    with pytest.raises(ValueError, match="contract"):
        runner.load(checkpoint)
    assert runner.update_counts["policy"] == 999
    assert not torch.equal(next(actor.parameters()), saved_actor[0])
    runner.contract["terrain_sha256"] = "fixture_plane"
    runner.load(checkpoint)
    assert runner.update_counts == report["actual_updates"]
    assert runner.replay.segment == 1 and runner.replay.controls == 11
    assert runner.runtime.episode_id.tolist() == [1, 1]
    assert runner.runtime.control_step.tolist() == [0, 0]
    assert runner.perception_replay.active == [None, None]
    for parameter, saved in zip(actor.parameters(), saved_actor):
        torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
    assert all(opt.state for opt in runner.optimizers.values())
    from dataclasses import replace

    runner.cfg = replace(runner.cfg, model_only_iterations=3, physical_updates=2, style_updates=2)
    actor_before = [p.detach().clone() for p in actor.parameters()]
    critic_before = [p.detach().clone() for p in runner.critic.parameters()]
    warmup = runner.update()
    assert warmup["actual_updates"] == dict(
        perception=1, physical=2, failure=2, style=2, policy=0, critic=2 if real_critic_only else 0
    )
    assert warmup["model_only"] and warmup["policy"] is None
    assert warmup["evaluation"] is None
    unchanged = [(actor, actor_before)]
    if not real_critic_only:
        unchanged.append((runner.critic, critic_before))
    for module, parameters in unchanged:
        for parameter, original in zip(module.parameters(), parameters):
            torch.testing.assert_close(parameter, original, rtol=0, atol=0)
