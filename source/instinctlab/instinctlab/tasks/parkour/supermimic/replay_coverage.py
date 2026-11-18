"""Coverage checks for saved native Hiking ring buffers, without simulation."""

import math
import torch

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1


def replay_rows(replay):
    """Return filled rows in time order, excluding unwritten ring capacity."""
    filled, capacity, cursor = (int(replay[name]) for name in ("filled", "capacity_per_env", "cursor"))
    if not 0 < filled <= capacity or not 0 <= cursor < capacity:
        raise ValueError("invalid native replay ring metadata")
    return (torch.arange(filled) + cursor - filled) % capacity


def native_replay_coverage(replay, expected_schema):
    """Report recorded-state coverage, not natural gait or stair completion.

    Joint order follows the native collector's fixed policy joint order. A
    matching buffer layout alone does not prove the physical/terrain contract.
    """
    rows = replay_rows(replay)
    differences = {}
    for name, shape in (expected_schema["state_format"] | expected_schema["context_format"]).items():
        value = replay.get("buffer/" + name)
        if value is None or value.shape[-1] != math.prod(shape):
            differences[name] = {
                "expected_size": math.prod(shape),
                "actual_size": None if value is None else value.shape[-1],
            }

    def field(name):
        return replay["buffer/" + name][rows]

    joint = field("joint_pos")
    knee_ids = [list(G1_29DOF_POLICY_JOINT_ORDER_V1).index(n) for n in ("left_knee_joint", "right_knee_joint")]
    knees = joint[..., knee_ids]
    quaternion = field("root_quat_w")
    tilt = torch.acos((1 - 2 * quaternion[..., :2].square().sum(-1)).clamp(-1, 1))
    speed = field("root_lin_vel_w")[..., :2].norm(dim=-1)
    age = field("episode_time").squeeze(-1)
    torso = field("illegal_torso_contact").squeeze(-1)
    # A descriptive state filter; this is deliberately not a success label.
    moving = (tilt < math.pi / 6) & (speed > 0.2) & (age >= 1) & (torso < 0.5)
    contacts = field("contact_state")[..., 4:6] >= 0.5
    episode = field("episode_step").reshape(len(rows), -1)
    continuous = episode[1:] == episode[:-1] + 1
    changed = (contacts[1:] != contacts[:-1]) & continuous[..., None]

    def summary(value):
        return {"min": float(value.min()), "median": float(value.median()), "max": float(value.max())}

    return {
        "filled_rows": len(rows),
        "environments": int(replay["num_envs"]),
        "recorded_states": int(speed.numel()),
        "layout_differences": differences,
        "joint_order_source": "native collector fixed G1_29DOF_POLICY_JOINT_ORDER_V1; no joint names in ring file",
        "knee_angles_rad": {name: summary(knees[..., i]) for i, name in enumerate(("left", "right"))},
        "either_knee_above_rad": {
            str(threshold): int((knees.amax(-1) > threshold).sum()) for threshold in (1.2, 1.5, 2.0)
        },
        "upright_noncontact_moving_states": int(moving.sum()),
        "upright_noncontact_moving_high_knee_states": int((moving & (knees.amax(-1) > 1.2)).sum()),
        "upright_noncontact_moving_knee_above_rad": {
            str(threshold): int((moving & (knees.amax(-1) > threshold)).sum()) for threshold in (1.2, 1.5, 2.0)
        },
        "moving_filter": "tilt<30deg, planar COM speed>0.2m/s, episode age>=1s, no recorded torso contact",
        "planar_com_speed_m_s": summary(speed),
        "tilt_degrees": summary(tilt * 180 / math.pi),
        "continuous_support_changes_left_right": changed.sum((0, 1)).tolist(),
        "samples_at_episode_start": int((episode == 0).sum()),
        "scope": "Recorded-state coverage only. No terrain identity, stair-contact or stable-controller certification.",
    }
