"""Reference tracking must preserve frames, commands and dataset boundaries."""

import numpy as np
import torch
from pathlib import Path

import pytest
from instinctlab_supermimic.reference import REFERENCE_INPUT_DIM, ReferencePlayer, tracking_reward

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.tasks.parkour.supermimic.lafan_reference import FEET
from instinctlab.tasks.parkour.supermimic.motion_tracking_reference import ReferenceWindows, motion_tracking_targets
from instinctlab.tasks.parkour.supermimic.stair_kinematics import NativeLinkKinematics

URDF = Path(__file__).resolve().parents[1] / "instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"


@pytest.fixture(scope="module")
def kinematics():
    return NativeLinkKinematics(URDF, "cpu", tracked_bodies=FEET)


def motion(frames=60):
    return dict(
        framerate=np.asarray(50),
        joint_names=np.asarray(G1_29DOF_POLICY_JOINT_ORDER_V1),
        joint_pos=np.zeros((frames, 29), np.float32),
        base_pos_w=np.tile(np.array([0, 0, 0.8], np.float32), (frames, 1)),
        base_quat_w=np.tile(np.array([1, 0, 0, 0], np.float32), (frames, 1)),
    )


def test_reference_velocity_in_heading_frame_and_no_quaternion_sign_spikes(kinematics):
    data = motion()
    data["base_pos_w"][:, 1] = np.arange(60) * 0.02 * 0.5
    data["base_quat_w"][:] = [2**-0.5, 0, 0, 2**-0.5]
    data["base_quat_w"][1::2] *= -1
    targets = motion_tracking_targets(data, kinematics)
    torch.testing.assert_close(targets["command"], torch.tensor([0.5, 0, 0]).expand(60, 3), atol=2e-6, rtol=0)
    assert not targets["contact"].any()  # Translating feet cannot be labelled as planted.
    assert not targets["joint_vel"].any()


def test_contact_labels_use_sole_height_and_world_speed(kinematics):
    data = motion()
    first = motion_tracking_targets(data, kinematics)
    data["base_pos_w"][:, 2] -= float(first["sole_height"].min())
    planted = motion_tracking_targets(data, kinematics)
    assert planted["contact"].all()
    data["base_pos_w"][:, 2] += 0.1
    lifted = motion_tracking_targets(data, kinematics)
    assert not lifted["contact"].any()
    # Relative foot targets are invariant to world translation.
    torch.testing.assert_close(planted["foot_pos"], lifted["foot_pos"])


def test_reference_rejects_wrong_joint_order_or_frame_rate(kinematics):
    data = motion()
    data["joint_names"] = data["joint_names"][::-1]
    with pytest.raises(ValueError, match="joint order"):
        motion_tracking_targets(data, kinematics)
    data = motion()
    data["framerate"] = np.asarray(30)
    with pytest.raises(ValueError, match="frame rate"):
        motion_tracking_targets(data, kinematics)


def clip(velocity, joint=0.0, frames=12):
    return dict(
        command=torch.tensor(velocity).float().expand(frames, 3).clone(),
        joint_pos=torch.full((frames, 29), joint),
        timestamp=torch.arange(frames),
    )


def test_matching_never_crosses_clips_or_lets_pose_override_command():
    bank = ReferenceWindows([clip([0.5, 0, 0], 0.0), clip([-0.5, 0, 0], 0.7)], horizon=5, stride=1)
    start, valid = bank.select(
        torch.tensor([[0.5, 0, 0], [-0.5, 0, 0]]), torch.tensor([0.7, 0.0])[:, None].expand(2, 29)
    )
    assert valid.all()
    assert start[0] < 12 and start[1] >= 12
    windows = bank.targets(start)
    torch.testing.assert_close(windows["command"][:, :, 0], torch.tensor([0.5, -0.5])[:, None].expand(2, 6))
    assert (windows["timestamp"].diff(dim=1) == 1).all()
    # Unsupported sideways commands must not inherit a forward target.
    _, valid = bank.select(torch.tensor([[0.0, 1.0, 0.0]]), torch.zeros(1, 29))
    assert not valid.any()


def test_matching_checks_whole_window_not_average():
    turning = clip([0, 0, 0])
    turning["command"][:, 2] = torch.tensor([-1.0, 1.0]).repeat(6)
    bank = ReferenceWindows([turning], horizon=5, stride=1)
    _, valid = bank.select(torch.zeros(1, 3), torch.zeros(1, 29))
    assert not valid.any()


def test_joint_continuity_only_selects_among_supported_candidates():
    bank = ReferenceWindows([clip([0.5, 0, 0], 0.0), clip([0.5, 0, 0], 0.7)], horizon=5)
    start, valid = bank.select(torch.tensor([[0.5, 0, 0]]), torch.full((1, 29), 0.7))
    assert valid.all() and start[0] >= 12


def playback_clip():
    data = clip([0.5, 0, 0], frames=12)
    data.update(foot_pos=torch.zeros(12, 2, 3), contact=torch.ones(12, 2), joint_vel=torch.zeros(12, 29))
    data["joint_pos"][:, 0] = torch.arange(12) / 100
    return data


def test_playback_cursor_survives_replay_and_resets_without_crossing_clips():
    bank = ReferenceWindows([playback_clip()], horizon=3, stride=1)
    player = ReferencePlayer(bank, 2)
    command, q = torch.tensor([[0.5, 0, 0]]).expand(2, 3), torch.zeros(2, 29)
    features, target, valid = player.prepare(command, q)
    assert features.shape == (2, REFERENCE_INPUT_DIM) and valid.all()
    assert (target["timestamp"] == 1).all()  # t input commands t+1 reward.
    snapshot = {name: value.clone() for name, value in player.state.items()}
    imagined = ReferencePlayer(bank, 2, snapshot)
    for _ in range(10):  # multiple segment boundaries
        actual = player.prepare(command, q)
        expected = imagined.prepare(command, q)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        player.advance()
        imagined.advance()
    assert (snapshot["cursor"] == 0).all()
    # A reset invalidates only its own cursor.
    player.state["remaining"][0] = 0
    other = player.state["cursor"][1].clone()
    player.prepare(command, q)
    assert player.state["cursor"][0] == 0 and player.state["cursor"][1] == other
    conflict, _, valid = player.prepare(-command, q)
    assert not valid.any() and not conflict.any()


def test_tracking_reward_state_gradients_and_no_dataset_gradient():
    q = torch.full((2, 29), 0.1, requires_grad=True)
    qd = torch.ones(2, 29, requires_grad=True)
    feet = torch.full((2, 2, 3), 0.02, requires_grad=True)
    contact = torch.full((2, 2), 0.6, requires_grad=True)
    target = dict(
        joint_pos=torch.zeros_like(q, requires_grad=True),
        joint_vel=torch.zeros_like(qd),
        foot_pos=torch.zeros_like(feet),
        contact=torch.ones_like(contact),
    )
    reward, _ = tracking_reward(q, qd, feet, contact, target, torch.tensor([True, False]), 0.02)
    reward.sum().backward()
    assert reward[0] > 0 and reward[1] == 0
    assert (q.grad[0] < 0).all() and (contact.grad[0] > 0).all()
    assert feet.grad[0].norm() > 0 and qd.grad[0].norm() > 0
    assert not q.grad[1].any() and target["joint_pos"].grad is None


def test_reference_controller_embeds_library_and_matches_training_actor(tmp_path):
    from instinctlab_supermimic.depth import DEPTH_SHAPE, DepthAutoencoder, DepthLatentPolicy, load_depth_controller

    bank = ReferenceWindows([playback_clip()], horizon=3)
    encoder, actor = DepthAutoencoder(), DepthLatentPolicy(REFERENCE_INPUT_DIM)
    path = tmp_path / "controller.pt"
    torch.save(
        dict(
            format="hiking_depth_reference_controller_v1",
            depth=encoder.encoder.state_dict(),
            actor=actor.state_dict(),
            depth_shape=DEPTH_SHAPE,
            depth_stride=5,
            reference_bank=bank.state_dict(),
        ),
        path,
    )
    controller = load_depth_controller(path)
    player = ReferencePlayer(bank, 2)
    q, command = torch.zeros(2, 29), torch.tensor([[0.5, 0, 0]]).expand(2, 3)
    proprio, image = torch.randn(2, 768), torch.rand(2, 18, 32)
    features, _, _ = player.prepare(command, q)
    expected = actor(proprio, encoder.encode(image[:, None].expand(-1, 8, -1, -1)), features)
    actual = controller(proprio, image, command=command, joint_pos=q)
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    with pytest.raises(ValueError, match="needs velocity"):
        controller(proprio, image)


def test_reference_reset_reanchors_xy_yaw_and_origin_velocity(kinematics):
    from types import SimpleNamespace as NS

    from instinctlab.tasks.parkour.supermimic.motion_tracking_reference import reset_flat_tracking_reference

    torch.manual_seed(19)
    data = motion()
    data["base_pos_w"][:, 1] = 2 + np.arange(60) * 0.01
    data["base_quat_w"][:] = [2**-0.5, 0, 0, 2**-0.5]
    bank = ReferenceWindows([motion_tracking_targets(data, kinematics)], horizon=3)

    class Robot:
        def __init__(self):
            root = torch.zeros(64, 13)
            root[:, 2], root[:, 6] = 0.9, 1.0
            self.data = NS(default_root_state=NS(torch=root), default_joint_pos=NS(torch=torch.zeros(64, 29)))

        def find_joints(self, names, preserve_order):
            return list(range(29)), names

        def write_joint_state_to_sim_index(self, **kwargs):
            self.joints = kwargs

        def write_root_link_pose_to_sim_index(self, **kwargs):
            self.pose = kwargs["root_pose"]

        def write_root_link_velocity_to_sim_index(self, **kwargs):
            self.velocity = kwargs["root_velocity"]

    class Scene(dict):
        env_origins = torch.tensor([10.0, 20.0, 0.0]).expand(64, 3)

    robot = Robot()
    env = NS(
        num_envs=64,
        device="cpu",
        scene=Scene(robot=robot),
        _tracking_reference_bank=bank,
        _tracking_reference_manifest={},
        _tracking_reset_cursor=torch.zeros(64, dtype=torch.long),
    )
    reset_flat_tracking_reference(env, slice(None), "unused_cached_library", 3)
    reference = env._tracking_reset_cursor >= 0
    assert reference.any() and (~reference).any()
    torch.testing.assert_close(robot.pose[:, :2], torch.tensor([10.0, 20.0]).expand(64, 2))
    torch.testing.assert_close(robot.pose[:, 3:7], torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(64, 4), atol=1e-6, rtol=0)
    torch.testing.assert_close(
        robot.velocity[reference, :3], torch.tensor([0.5, 0, 0]).expand(int(reference.sum()), 3), atol=1e-5, rtol=0
    )
    assert not robot.velocity[~reference].any()
    torch.testing.assert_close(robot.pose[reference, 2], torch.full((int(reference.sum()),), 0.82))
    torch.testing.assert_close(robot.pose[~reference, 2], torch.full((int((~reference).sum()),), 0.9))


def test_reference_imagination_reaches_actor_through_frozen_world_model():
    from dataclasses import replace

    from instinctlab_supermimic.depth import DepthAutoencoder, DepthLatentPolicy
    from instinctlab_supermimic.networks import WorldModel
    from test_hiking_native_task import imagination_fixture

    from instinctlab.tasks.parkour.supermimic.native_imagination import frozen_parameters, imagine_native

    task, _, _, seed = imagination_fixture()
    task.physics.model = WorldModel(3, 2 * len(task.body_names) + 2, 30, z_dim=32, hidden=(32, 32), aux_dim=58)
    with torch.no_grad():
        task.physics.model.net[-1].weight.normal_(std=0.003)
    reference = playback_clip()
    reference["command"][:] = seed.command["velocity"][0]
    bank = ReferenceWindows([reference], horizon=3, tolerance=(10, 10, 10))
    player = ReferencePlayer(bank, 2)
    player.prepare(seed.command["velocity"], seed.state[:, 13:42])
    seed = replace(seed, perception_state=None, visual_latent=torch.randn(2, 32), reference_state=player.state)
    actor, encoder = DepthLatentPolicy(REFERENCE_INPUT_DIM), DepthAutoencoder()
    with frozen_parameters(task.physics.model, encoder):
        imagined = imagine_native(task, actor, encoder, seed, horizon=3, add_noise=False, reference_bank=bank)
        imitation = sum(
            value.sum()
            for name, value in imagined["reward_terms"].items()
            if name.startswith("reference_") and name != "reference_active"
        )
        gradients = torch.autograd.grad(imitation, tuple(actor.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients) and sum(g.norm() for g in gradients) > 0
    assert all(p.grad is None for p in task.physics.model.parameters())
    assert (
        imagined["value_features"].shape[-1]
        == imagined["physical_features"].shape[-1] + 768 + 670 + 32 + REFERENCE_INPUT_DIM
    )
