"""Command-compatible motion targets, independent of the AMP discriminator.

Positions use the converted torso root and canonical policy joint order. Foot
contacts are kinematic labels derived from sole height and world foot speed;
they are not measured forces. Windows never cross clip boundaries.
"""

import hashlib
import json
import numpy as np
import torch
from pathlib import Path

from instinctlab_supermimic.reference import ReferenceWindows  # noqa: F401
from instinctlab_supermimic.rotations import mat_to_quat, mv, quat_to_mat, so3_log, yaw_mat

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1

from .lafan_reference import FEET
from .stair_kinematics import NativeLinkKinematics


def motion_tracking_targets(motion, kinematics, dt=0.02):
    """Read native NPZ arrays; compute heading-local targets at each frame.

    Central differences are used inside the clip and one-sided differences at
    endpoints. No differences connect unrelated clips. WXYZ quaternion signs
    do not affect the SO(3) angular velocity calculation.
    """
    if list(motion["joint_names"]) != list(G1_29DOF_POLICY_JOINT_ORDER_V1):
        raise ValueError("Motion tracking requires canonical named joint order.")
    if abs(float(motion["framerate"]) * dt - 1) > 1e-6:
        raise ValueError("Motion tracking and control must use the same frame rate.")
    device = kinematics.order.device
    q, p, quat = (
        torch.as_tensor(motion[name], dtype=torch.float32, device=device)
        for name in ("joint_pos", "base_pos_w", "base_quat_w")
    )
    if len(q) < 3 or p.shape != (len(q), 3) or quat.shape != (len(q), 4) or q.shape != (len(q), 29):
        raise ValueError("Motion tracking needs at least three complete native frames.")
    if not all(torch.isfinite(x).all() for x in (q, p, quat)) or (quat.norm(dim=-1) - 1).abs().max() > 1e-3:
        raise ValueError("Invalid motion tracking coordinates or quaternions.")
    rotation = quat_to_mat(quat)
    heading_inverse = yaw_mat(rotation).transpose(-1, -2)
    velocity = torch.gradient(p, spacing=dt, dim=0)[0]
    qd = torch.gradient(q, spacing=dt, dim=0)[0]
    # Relative rotations expressed in the world frame. Average adjacent
    # interval rates to place them at the same timestamp as the pose.
    interval_w = so3_log(rotation[1:] @ rotation[:-1].transpose(-1, -2)) / dt
    angular = torch.cat((interval_w[:1], (interval_w[:-1] + interval_w[1:]) / 2, interval_w[-1:]))
    links = kinematics.links(
        dict(
            root_pos_w=p,
            root_quat_w=quat[:, [1, 2, 3, 0]],
            root_lin_vel_w=velocity,
            root_ang_vel_w=angular,
            joint_pos=q,
            joint_vel=qd,
        ),
        # Input velocities are at the root link origin, not its COM.
        root_com=torch.zeros(3, device=device),
    )
    foot_ids = [kinematics.tracked_bodies.index(name) for name in FEET]
    feet = links["body_link_pos_w"][:, foot_ids]
    foot_rotation = quat_to_mat(links["body_link_quat_w"][:, foot_ids][..., [3, 0, 1, 2]])
    corners = p.new_tensor([[x, y, -0.058] for x in (0.132, -0.054) for y in (-0.018, 0.018)])
    sole = torch.einsum("bfij,pj->bfpi", foot_rotation, corners) + feet[:, :, None]
    sole_height = sole[..., 2].amin(-1)
    foot_velocity = torch.gradient(feet, spacing=dt, dim=0)[0]
    speed = foot_velocity.norm(dim=-1)
    contact = (sole_height < 0.04) & (speed < 0.2)
    local_velocity = mv(heading_inverse, velocity)
    return dict(
        joint_pos=q,
        joint_vel=qd,
        root_pos=p,
        root_quat=quat,
        root_velocity=velocity,
        root_angular_velocity=angular,
        command=torch.cat((local_velocity[:, :2], angular[:, 2:3]), -1),
        foot_pos=mv(heading_inverse[:, None], feet - p[:, None]),
        sole_height=sole_height,
        foot_speed=speed,
        contact=contact.float(),
    )


def load_motion_tracking(directory, urdf, device="cpu"):
    """Load only the hashed training split from the converted LAFAN library."""
    directory, urdf = Path(directory), Path(urdf)
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["split"] != "train" or manifest["output_quaternion_order"] != "wxyz":
        raise ValueError("Expected WXYZ training motions, excluding validation/test clips.")
    if hashlib.sha256(urdf.read_bytes()).hexdigest() != manifest["urdf_sha256"]:
        raise ValueError("Motion tracking reference uses a different robot asset.")
    kinematics = NativeLinkKinematics(urdf, device, tracked_bodies=FEET)
    clips = []
    for record in manifest["clips"]:
        path = directory / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Motion tracking reference hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as motion:
            clips.append(motion_tracking_targets(motion, kinematics))
    return clips, manifest


def tracking_bank_for_env(env, directory, horizon):
    """Reset events and learner share the same verified, immutable targets."""
    if not hasattr(env, "_tracking_reference_bank"):
        clips, manifest = load_motion_tracking(directory, Path(env.cfg.scene.robot.spawn.asset_path), env.device)
        env._tracking_reference_bank = ReferenceWindows(clips, horizon=horizon)
        env._tracking_reference_manifest = manifest
        env._tracking_reset_cursor = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env._tracking_reference_bank, env._tracking_reference_manifest


def reset_flat_tracking_reference(env, env_ids, directory, horizon):
    """Reference-state initialization for the flat forward stage.

    Choose only windows compatible with 0.5 m/s forward. Remove reference
    world XY/yaw, retain pose and world twists in the new heading. No
    dynamics labels are fabricated: F still trains on simulator transitions.
    """
    bank, _ = tracking_bank_for_env(env, directory, horizon)
    command = bank.mean_command.new_tensor([0.5, 0.0, 0.0])
    error = torch.maximum((command - bank.command_min).abs(), (command - bank.command_max).abs())
    eligible = bank.starts[(error <= bank.tolerance).all(-1)]
    if not len(eligible):
        raise ValueError("Reference library has no complete window for flat 0.5 m/s forward initialization.")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    elif isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    indices = eligible[torch.randint(len(eligible), (len(env_ids),), device=env.device)]
    frame = {name: value[indices] for name, value in bank.frames.items()}
    rotation = quat_to_mat(frame["root_quat"])
    inverse_heading = yaw_mat(rotation).transpose(-1, -2)
    new_quat = mat_to_quat(inverse_heading @ rotation)[:, [1, 2, 3, 0]]  # native backend XYZW
    robot = env.scene["robot"]
    default_root = robot.data.default_root_state.torch[env_ids]
    position = default_root[:, :3].clone() + env.scene.env_origins[env_ids]
    position[:, 2] = env.scene.env_origins[env_ids, 2] + frame["root_pos"][:, 2] + 0.02
    joint_ids, _ = robot.find_joints(list(G1_29DOF_POLICY_JOINT_ORDER_V1), preserve_order=True)
    velocity = torch.cat(
        (mv(inverse_heading, frame["root_velocity"]), mv(inverse_heading, frame["root_angular_velocity"])), -1
    )
    # Keep ordinary standing starts in the real replay as well: the exported
    # controller must learn to enter a gait without a simulator pose reset.
    standing = torch.rand(len(env_ids), device=env.device) < 0.2
    position[standing] = default_root[standing, :3] + env.scene.env_origins[env_ids[standing]]
    new_quat[standing] = default_root[standing, 3:7]
    velocity[standing] = 0
    frame["joint_pos"][standing] = robot.data.default_joint_pos.torch[env_ids[standing]][:, joint_ids]
    frame["joint_vel"][standing] = 0
    robot.write_joint_state_to_sim_index(
        position=frame["joint_pos"], velocity=frame["joint_vel"], joint_ids=joint_ids, env_ids=env_ids
    )
    robot.write_root_link_pose_to_sim_index(root_pose=torch.cat((position, new_quat), -1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=env_ids)
    indices[standing] = -1  # Re-select by the measured standing pose on first control.
    env._tracking_reset_cursor[env_ids] = indices


def configure_tracking_resets(cfg, directory, horizon):
    from isaaclab.managers import EventTermCfg

    cfg.events.reset_base = None
    cfg.events.reset_robot_joints = None
    cfg.events.reset_tracking_reference = EventTermCfg(
        func=reset_flat_tracking_reference, mode="reset", params=dict(directory=directory, horizon=horizon)
    )
