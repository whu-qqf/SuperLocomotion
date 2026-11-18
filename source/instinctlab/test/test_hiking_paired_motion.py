"""Paired reference clips must retain terrain identity and physical timestamps."""

import json
import numpy as np
import torch
import yaml
from pathlib import Path
from types import SimpleNamespace

import pytest
from instinctlab_supermimic.depth import DepthAutoencoder, DepthHistory, DepthLatentPolicy, load_depth_controller
from instinctlab_supermimic.paired_reference import (
    PAIRED_REFERENCE_INPUT_DIM,
    PairedReferenceTargets,
    paired_reference_features,
    paired_tracking_reward,
)
from instinctlab_supermimic.rotations import quat_to_mat

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.tasks.parkour.supermimic.lafan_reference import FEET
from instinctlab.tasks.parkour.supermimic.native_physics import NativePhysics
from instinctlab.tasks.parkour.supermimic.paired_environment import paired_scan_out_of_bounds
from instinctlab.tasks.parkour.supermimic.paired_motion import (
    PairedMotionLibrary,
    PairedMotionPlayer,
    file_hash,
    next_touchdown_targets,
)
from instinctlab.tasks.parkour.supermimic.terrain import TerrainHeightField

URDF = Path(__file__).resolve().parents[1] / "instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"


@pytest.fixture
def dataset(tmp_path):
    source, directory = tmp_path / "source", tmp_path / "dataset"
    source.mkdir()
    directory.mkdir()
    frames, batch = 5, 2
    names = list(reversed(G1_29DOF_POLICY_JOINT_ORDER_V1))
    q = torch.arange(29).float()[None, None].expand(frames, batch, -1) / 100
    position = torch.zeros(frames, batch, 3)
    position[:, :, 2] = 0.8
    position[:, 0, 0] = torch.arange(frames) * 0.01
    position[:, 1, 0] = 20 + torch.arange(frames) * 0.01
    quaternion = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(frames, batch, 4)
    velocity = torch.tensor([0.5, 0, 0]).expand(frames, batch, 3)
    force = torch.zeros(frames, batch, 2, 3)
    force[:, :, 0, 2] = torch.arange(frames)[:, None] * 50 + 50
    trace = dict(
        joint_names=names,
        dt=0.02,
        joint_pos=q,
        joint_vel=torch.zeros_like(q),
        root_pos_w=position,
        root_quat_w=quaternion,
        root_lin_vel_w=velocity,
        root_ang_vel_w=torch.zeros_like(velocity),
        post_step_foot_force_w=force,
        post_step_contact_valid=torch.ones(frames, batch, dtype=torch.bool),
    )
    torch.save(trace, source / "physical_trace.pt")
    (source / "result.json").write_text(json.dumps(dict(robot_urdf_sha256=file_hash(URDF))))
    np.savez(source / "grid.npz", data=np.zeros((3, 3)))
    metadata, audit = dict(terrains=[], motion_files=[]), dict(
        quaternion_npz="wxyz", quaternion_runtime="xyzw", terrains=[]
    )
    for env in range(batch):
        mesh = directory / f"terrain{env}.stl"
        mesh.write_bytes(f"hashed collision surface {env}".encode())
        origin = [20.0 * env, 0.0, 0.0]
        path = directory / f"seed1_env{env}_retargetted.npz"
        np.savez(
            path,
            joint_names=names,
            framerate=50,
            joint_pos=q[:, env].numpy(),
            joint_vel=trace["joint_vel"][:, env].numpy(),
            base_pos_w=position[:, env].numpy() - origin,
            base_quat_w=quaternion[:, env, [3, 0, 1, 2]].numpy(),
            base_lin_vel_w=velocity[:, env].numpy(),
            base_ang_vel_w=trace["root_ang_vel_w"][:, env].numpy(),
        )
        forces = np.zeros((frames, 2, 3), dtype=np.float32)
        forces[1:] = force[:-1, env].numpy()
        np.savez(
            path.with_suffix(".foot_contacts.npz"),
            foot_names=FEET,
            force_w=forces,
            valid=np.array([False, True, True, True, True]),
            contact=np.linalg.norm(forces, axis=-1) > 1,
        )
        metadata["terrains"].append(dict(terrain_id=env, terrain_file=mesh.name))
        metadata["motion_files"].append(dict(terrain_id=env, motion_file=path.name))
        audit["terrains"].append(
            dict(
                terrain_id=env,
                source=str(source),
                mesh_source="recorded_collision_heightfield",
                mesh_sha256=file_hash(mesh),
                origin_w=origin,
                source_trace_sha256=file_hash(source / "physical_trace.pt"),
                collision_heightfield=dict(file="grid.npz", sha256=file_hash(source / "grid.npz")),
                motions=[dict(file=path.name, sha256=file_hash(path))],
            )
        )
    (directory / "metadata.yaml").write_text(yaml.safe_dump(metadata))
    (directory / "audit.json").write_text(json.dumps(audit))
    return directory


def test_query_preserves_terrain_origin_joint_order_and_clip_end(dataset):
    library = PairedMotionLibrary(dataset, URDF)
    frames = torch.tensor([[0, 1, 4, 5, -1], [0, 1, 4, 5, -1]])
    origin = torch.tensor([[10.0, 20, 3], [30.0, 40, 5]])
    targets = library.query(torch.tensor([0, 1]), frames, origin, torch.tensor([0, 1]))
    assert targets["valid"].tolist() == [[True, True, True, False, False]] * 2
    assert targets["contact_valid"].tolist() == [[False, True, True, False, False]] * 2
    torch.testing.assert_close(targets["root_pos"][:, 0], origin + torch.tensor([0, 0, 0.8]))
    torch.testing.assert_close(targets["root_velocity"][:, 0], torch.tensor([0.5, 0, 0]).expand(2, 3))
    torch.testing.assert_close(targets["joint_pos"][0, 0], torch.arange(28, -1, -1).float() / 100)
    assert targets["force"][0, 1, 0, 2] == 50  # post-step 0 belongs to pose 1.
    # Padding stays on this clip's endpoint; it cannot leak the next clip's frame.
    torch.testing.assert_close(targets["root_pos"][:, 2], targets["root_pos"][:, 3])
    baseline = library.query(torch.tensor([0, 1]), frames, torch.zeros_like(origin), torch.tensor([0, 1]))
    torch.testing.assert_close(targets["foot_pos"] - baseline["foot_pos"], origin[:, None, None].expand(2, 5, 2, 3))
    assert all(not value.requires_grad for value in targets.values())
    with pytest.raises(ValueError, match="terrain"):
        library.query(torch.tensor([0, 1]), frames, origin, torch.tensor([1, 0]))


def test_contact_timestamp_corruption_is_rejected_without_trusting_export_hash(dataset):
    path = dataset / "seed1_env0_retargetted.foot_contacts.npz"
    with np.load(path) as data:
        values = {name: data[name] for name in data.files}
    values["force_w"][1:, :, 2] += 10
    np.savez(path, **values)
    with pytest.raises(ValueError, match="timestamps"):
        PairedMotionLibrary(dataset, URDF)


def test_mesh_or_robot_asset_changes_are_rejected(dataset, tmp_path):
    replacement = tmp_path / "robot.urdf"
    replacement.write_text(URDF.read_text() + "\n")
    with pytest.raises(ValueError, match="robot assets"):
        PairedMotionLibrary(dataset, replacement)
    (dataset / "terrain0.stl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="terrain hash"):
        PairedMotionLibrary(dataset, URDF)


def test_touchdowns_skip_current_stance_and_do_not_bridge_unknown_contacts():
    contacts = torch.tensor([0, 1, 1, 0, 0, 1, 1, 0, 1]).bool()[:, None].expand(-1, 2)
    feet = torch.arange(9).float()[:, None, None].expand(-1, 2, 3)
    target = dict(contact=contacts, contact_valid=torch.ones(9, dtype=torch.bool), foot_pos=feet)
    labels = next_touchdown_targets(target)
    assert labels["touchdown_pos"][:5, 0, 0].tolist() == [1, 5, 5, 5, 5]
    assert labels["touchdown_steps"][:5, 0].tolist() == [1, 4, 3, 2, 1]
    assert not labels["touchdown_valid"][-1].any()
    target["contact_valid"][4] = False
    labels = next_touchdown_targets(target)
    assert not labels["touchdown_valid"][1:5].any()
    assert labels["touchdown_valid"][5:8].all()


def test_player_cannot_rematch_terrain_or_loop_at_clip_end(dataset):
    library = PairedMotionLibrary(dataset, URDF)
    terrain_ids, origins = torch.tensor([0, 1]), torch.zeros(2, 3)
    player = PairedMotionPlayer(library, terrain_ids, origins)
    with pytest.raises(ValueError, match="reset"):
        player.window(32)
    player.reset(torch.arange(2))
    initial = player.window(4)
    assert initial["valid"].all()
    player.advance()
    restored = PairedMotionPlayer(library, terrain_ids, origins, player.state)
    for key, value in restored.window(32).items():
        torch.testing.assert_close(value, player.window(32)[key])
    for _ in range(10):
        player.advance()
    assert player.ended.all()
    assert player.state["clip_id"].tolist() == [0, 1]
    assert player.window(32)["valid"].sum() == 2  # Endpoint only, no wrapping.
    with pytest.raises(ValueError, match="terrain assignments"):
        PairedMotionPlayer(library, terrain_ids.flip(0), origins, player.state)
    old_frame = player.state["frame"][1].clone()
    player.reset(torch.tensor([0]), random_phase=True, minimum_remaining=3)
    assert player.state["frame"][0] <= 1
    assert player.state["frame"][1] == old_frame


def test_paired_conditions_and_rewards_use_identical_frames_and_detach_labels(dataset):
    library = PairedMotionLibrary(dataset, URDF)
    targets = library.query(torch.tensor([0]), torch.tensor([[0, 1, 4, 5]]), torch.zeros(1, 3), torch.tensor([0]))
    target = {name: value[0].clone() for name, value in targets.items()}
    target["joint_pos"].requires_grad_()
    position = (target["root_pos"] + 0.01).detach().requires_grad_()
    rotation = quat_to_mat(target["root_quat"])
    q = (target["joint_pos"] + 0.02).detach().requires_grad_()
    contact = torch.full((4, 2), 0.7, requires_grad=True)
    state = (position, rotation, target["root_velocity"], target["root_angular_velocity"], q, target["joint_vel"])
    features = paired_reference_features(position, rotation, target, library.dt)
    assert features.shape == (4, PAIRED_REFERENCE_INPUT_DIM)
    assert not features[-1].any()
    reward, terms = paired_tracking_reward(state, target["foot_pos"], contact, target, library.dt)
    assert reward[-1] == 0
    assert terms["paired_contact"][0] == 0  # No measurement at initial pose.
    detached = tuple(value.detach() for value in state)
    real, _ = paired_tracking_reward(detached, target["foot_pos"], contact.detach(), target, library.dt)
    torch.testing.assert_close(real, reward.detach())
    reward.sum().backward()
    assert position.grad[:3].abs().sum() > 0
    assert q.grad[:3].abs().sum() > 0
    assert contact.grad[1:3].abs().sum() > 0
    assert not contact.grad[[0, 3]].any()
    assert target["joint_pos"].grad is None
    assert not position.grad[-1].any()
    assert not q.grad[-1].any()


def test_exported_paired_controller_uses_embedded_targets_and_explicit_cursor(dataset, tmp_path):
    library = PairedMotionLibrary(dataset, URDF)
    restored = PairedReferenceTargets.from_state_dict(library.state_dict())
    clips, terrain, origin = torch.tensor([0, 1]), torch.tensor([0, 1]), torch.zeros(2, 3)
    target = library.query(clips, torch.ones(2, 1, dtype=torch.long), origin, terrain)
    for key, value in restored.query(clips, torch.ones(2, 1, dtype=torch.long), origin, terrain).items():
        torch.testing.assert_close(value, target[key])
    target = {key: value[:, 0] for key, value in target.items()}
    actor, depth = DepthLatentPolicy(PAIRED_REFERENCE_INPUT_DIM), DepthAutoencoder()
    path = tmp_path / "controller.pt"
    torch.save(
        dict(
            format="hiking_depth_paired_controller_v1",
            depth_shape=(8, 18, 32),
            depth_stride=5,
            depth=depth.encoder.state_dict(),
            actor=actor.state_dict(),
            paired_reference=library.state_dict(),
        ),
        path,
    )
    controller = load_depth_controller(path)
    image, proprio = torch.rand(2, 18, 32), torch.rand(2, 768)
    position, rotation = target["root_pos"], quat_to_mat(target["root_quat"])
    features = paired_reference_features(position, rotation, target, library.dt)
    expected = actor(proprio, depth.encode(DepthHistory(image).stack()), features)
    actual = controller(proprio, image, position, rotation, clips, torch.zeros(2, dtype=torch.long), origin, terrain)
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="endpoint"):
        controller(proprio, image, position, rotation, clips, torch.full((2,), 4), origin, terrain)
    with pytest.raises(ValueError, match="terrain"):
        controller(proprio, image, position, rotation, clips, torch.zeros(2, dtype=torch.long), origin, terrain.flip(0))


def test_reference_reward_gradient_matches_finite_difference_through_real_urdf_fk(dataset):
    library = PairedMotionLibrary(dataset, URDF)
    target = library.query(torch.tensor([0, 1]), torch.tensor([[1], [2]]), torch.zeros(2, 3), torch.tensor([0, 1]))
    target = {key: value[:, 0] for key, value in target.items()}
    rotation = quat_to_mat(target["root_quat"])
    inputs = torch.cat((target["root_pos"] + 0.03, target["joint_pos"] + 0.07, torch.full((2, 2), 0.6)), -1)
    inputs.requires_grad_()

    def reward(values):
        position, q, contact = values[:, :3], values[:, 3:32], values[:, 32:]
        state = (position, rotation, target["root_velocity"], target["root_angular_velocity"], q, target["joint_vel"])
        links = library.kinematics.links(
            dict(
                root_pos_w=position,
                root_quat_w=target["root_quat"][:, [1, 2, 3, 0]],
                root_lin_vel_w=state[2],
                root_ang_vel_w=state[3],
                joint_pos=q,
                joint_vel=state[5],
            ),
            root_com=torch.zeros(3),
        )
        return paired_tracking_reward(state, links["body_link_pos_w"], contact, target, library.dt)[0]

    direction = torch.linspace(-1, 1, 34).expand_as(inputs)
    direction = direction / direction.norm(dim=-1, keepdim=True)
    derivative = (torch.autograd.grad(reward(inputs).sum(), inputs)[0] * direction).sum(-1)
    assert derivative.abs().min() > 1e-4
    for epsilon in (0.001, 0.0003):
        finite_difference = (reward(inputs + epsilon * direction) - reward(inputs - epsilon * direction)) / (
            2 * epsilon
        )
        torch.testing.assert_close(derivative, finite_difference, atol=3e-5, rtol=0.003)


def test_real_termination_checks_full_model_scan_before_root_leaves_terrain():
    field = TerrainHeightField(
        torch.zeros(41, 41),
        torch.ones(41, 41, dtype=torch.bool),
        [-1, -1],
        0.05,
        [[-0.8, -0.5], [-0.8, 0.5], [0.8, -0.5], [0.8, 0.5]],
    )
    position = torch.tensor([[0.0, 0.0, 0.8], [0.25, 0.0, 0.8]])
    quaternion = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, -1)

    class Scene(dict):
        terrain = SimpleNamespace(native_height_field=field)

    robot = SimpleNamespace(
        data=SimpleNamespace(
            root_link_pos_w=SimpleNamespace(torch=position), root_link_quat_w=SimpleNamespace(torch=quaternion)
        )
    )
    env = SimpleNamespace(scene=Scene(robot=robot))
    # Both roots are on the mesh, but only the first entire footprint fits.
    assert field.query(position[:, :2])[1].all()
    ended = paired_scan_out_of_bounds(env)
    assert ended.tolist() == [False, True]
    physics = NativePhysics.__new__(NativePhysics)
    physics.terrain = field
    _, supported = physics.scan_with_validity((position, quat_to_mat(quaternion[:, [3, 0, 1, 2]])))
    torch.testing.assert_close(ended, ~supported)
