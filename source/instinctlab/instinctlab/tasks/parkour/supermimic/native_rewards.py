"""Dense progress and smooth velocity penalties for native Hiking BPTT."""

import re
import torch

from isaaclab.managers import RewardTermCfg


def configure_imagined_depth_rewards(cfg):
    """Replace overlapping Hiking gait terms, retaining its tracking/regularizers.

    Use the spatial support field as the sole support penalty; the local gap
    readout still determines uncertain touchdowns but earns no second penalty.
    """
    from instinctlab_supermimic.locomotion import LOCOMOTION_WEIGHTS

    for name in ("feet_air_time", "feet_slide", "feet_flat_ori", "feet_at_plane"):
        getattr(cfg.rewards.rewards, name).weight = 0.0
    cfg.locomotion_reward_weights = dict(LOCOMOTION_WEIGHTS, locomotion_support=0.0)
    cfg.native_tracking_reward_version = "imagined_depth_no_duplicate_gait_v1"
    return cfg


def smooth_dont_wait(env, command_name, width=0.05):
    """Relax Hiking's three speed thresholds; keep its forward-command gate."""
    if width <= 0:
        raise ValueError("Velocity threshold width must be positive.")
    command = env.command_manager.get_command(command_name)
    velocity = env.scene["robot"].data.root_lin_vel_b.torch[:, 0]
    thresholds = velocity.new_tensor([0.15, 0.0, -0.15])
    return (command[:, 0] > 0.3) * torch.sigmoid((thresholds - velocity[:, None]) / width).sum(-1)


def goal_progress(env, command_name, target_radius=0.4):
    """Reduction of distance outside Hiking's arrival radius, per second.

    RewardManager multiplies by dt. A transition therefore earns exactly
    weight * (previous potential - current potential), including terminal
    transitions before autoreset. Standing commands earn no goal progress.
    """
    command = env.command_manager.get_term(command_name)
    target = command.pos_command_w[:, :2]
    previous = command.progress_start_pos_w[:, :2]
    current = env.scene["robot"].data.root_pos_w.torch[:, :2]
    before = ((target - previous).norm(dim=-1) - target_radius).clamp_min(0)
    after = ((target - current).norm(dim=-1) - target_radius).clamp_min(0)
    tracking_target = ~command.is_standing_env & ~command.random_velocity_indices
    return (before - after) / env.step_dt * tracking_target


def configure_native_tracking_rewards(cfg):
    """Retain Hiking commands/weights; add explicit progress and relax waiting."""
    cfg.rewards.rewards.goal_progress = RewardTermCfg(
        func=goal_progress,
        weight=4.0,
        params=dict(command_name="base_velocity", target_radius=cfg.commands.base_velocity.target_dis_threshold),
    )
    cfg.rewards.rewards.dont_wait.func = smooth_dont_wait
    cfg.rewards.rewards.dont_wait.params = dict(command_name="base_velocity", width=0.05)
    cfg.native_tracking_reward_version = "goal_progress4_smooth_wait005_v1"
    return cfg


_TGI_POSE_STD = {
    "walking": {
        r".*hip_pitch.*": 0.3,
        r".*hip_roll.*": 0.15,
        r".*hip_yaw.*": 0.15,
        r".*knee.*": 0.35,
        r".*ankle_pitch.*": 0.25,
        r".*ankle_roll.*": 0.1,
        r".*waist_yaw.*": 0.2,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.1,
        r".*shoulder_pitch.*": 0.15,
        r".*shoulder_roll.*": 0.15,
        r".*shoulder_yaw.*": 0.1,
        r".*elbow.*": 0.15,
        r".*wrist.*": 0.3,
    },
    "running": {
        r".*hip_pitch.*": 0.5,
        r".*hip_roll.*": 0.2,
        r".*hip_yaw.*": 0.2,
        r".*knee.*": 0.6,
        r".*ankle_pitch.*": 0.35,
        r".*ankle_roll.*": 0.15,
        r".*waist_yaw.*": 0.3,
        r".*waist_roll.*": 0.08,
        r".*waist_pitch.*": 0.2,
        r".*shoulder_pitch.*": 0.5,
        r".*shoulder_roll.*": 0.2,
        r".*shoulder_yaw.*": 0.15,
        r".*elbow.*": 0.35,
        r".*wrist.*": 0.3,
    },
}


def _tgi_pose_std(names, mode, device, dtype):
    values = []
    patterns = _TGI_POSE_STD.get(mode, {})
    for name in names:
        value = 0.05
        for pattern, candidate in patterns.items():
            if re.fullmatch(pattern, name):
                value = candidate
                break
        values.append(value)
    return torch.tensor(values, device=device, dtype=dtype)


def tgi_core_reward(task, state, action, previous_action, velocity, context):
    """TGI tracking/pose/control terms in Hiking's native root and joint frames.

    Terrain/contact terms are supplied once by locomotion_readout. Real replay
    and imagination call this same function; native RewardManager logs remain
    separately labelled diagnostics of the original task.
    """
    data = task.data(state, context)
    linear, angular = data.root_lin_vel_b.torch, data.root_ang_vel_b.torch
    gravity = data.projected_gravity_b.torch
    joints = state[4]
    default = context["default_joint_pos"][:, task.joint_ids]
    limits = context["soft_joint_pos_limits"][:, task.joint_ids]
    deviation = (joints - default).abs()
    names = task.physics.recorder.joint_names
    waist = joints.new_tensor([name.startswith("waist_") for name in names])
    hip = joints.new_tensor(["_hip_roll_" in name or "_hip_yaw_" in name for name in names])
    arms = joints.new_tensor([any(part in name for part in ("shoulder", "elbow", "wrist")) for name in names])
    speed = velocity[:, :2].norm(dim=-1) + velocity[:, 2].abs()
    standing = (speed < 0.05).float()
    running = (speed >= 1.5).float()
    walking = 1 - standing - running
    std = (
        _tgi_pose_std(names, "walking", joints.device, joints.dtype)[None] * walking[:, None]
        + _tgi_pose_std(names, "running", joints.device, joints.dtype)[None] * running[:, None]
        + _tgi_pose_std(names, "standing", joints.device, joints.dtype)[None] * standing[:, None]
    )
    terms = dict(
        tgi_track_linear=2 * torch.exp(-(velocity[:, :2] - linear[:, :2]).square().sum(-1) / 0.25),
        tgi_track_angular=2 * torch.exp(-(velocity[:, 2] - angular[:, 2]).square() / 0.5),
        tgi_upright=0.25 * torch.exp(-gravity[:, :2].square().sum(-1) / 0.2),
        tgi_standing_pose=2.5 * torch.exp(-((joints - default) / std).square().mean(-1)) * standing,
        tgi_joint_limits=-(torch.relu(joints - limits[..., 1]) + torch.relu(limits[..., 0] - joints)).sum(-1),
        tgi_action_rate=-0.1 * (action - previous_action).square().sum(-1),
        tgi_body_angular=-0.05 * state[3][:, :2].square().sum(-1),
        tgi_joint_deviation=-(deviation * (waist + hip + 0.1 * arms)).sum(-1),
        tgi_vertical_velocity=-0.5 * linear[:, 2].square(),
        tgi_angular_xy=-0.05 * angular[:, :2].square().sum(-1),
    )
    terms = {name: value * task.dt for name, value in terms.items()}
    return sum(terms.values()), terms
