"""Replay coverage must exclude unwritten rows and reset contact changes."""

import torch

import pytest

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.tasks.parkour.supermimic.replay_coverage import native_replay_coverage, replay_rows


def test_wrapped_replay_excludes_unwritten_capacity_and_reset_contact_changes():
    replay = {"filled": 3, "capacity_per_env": 5, "cursor": 1, "num_envs": 2}
    for name, size in (
        ("joint_pos", 29),
        ("root_quat_w", 4),
        ("root_lin_vel_w", 3),
        ("episode_time", 1),
        ("illegal_torso_contact", 1),
        ("contact_state", 21),
    ):
        replay["buffer/" + name] = torch.zeros(5, 2, size)
    replay["buffer/root_quat_w"][..., 3] = 1
    replay["buffer/root_lin_vel_w"][..., 0] = 0.3
    replay["buffer/episode_time"][:] = 2
    replay["buffer/joint_pos"][:] = 0.6
    replay["buffer/joint_pos"][1:3] = 100  # Not written in this ring.
    knee = list(G1_29DOF_POLICY_JOINT_ORDER_V1).index("left_knee_joint")
    replay["buffer/joint_pos"][0, 0, knee] = 1.6
    replay["buffer/episode_step"] = torch.zeros(5, 2, dtype=torch.long)
    replay["buffer/episode_step"][3] = torch.tensor([5, 5])
    replay["buffer/episode_step"][4] = torch.tensor([6, 0])  # Right environment reset.
    replay["buffer/episode_step"][0] = torch.tensor([7, 1])
    replay["buffer/contact_state"][4, :, 4:6] = 1
    schema = {
        "state_format": {"joint_pos": (29,), "contact_state": (21,)},
        "context_format": {"shape_friction": (33,)},
    }
    assert replay_rows(replay).tolist() == [3, 4, 0]
    report = native_replay_coverage(replay, schema)
    assert report["recorded_states"] == 6
    assert report["either_knee_above_rad"] == {"1.2": 1, "1.5": 1, "2.0": 0}
    assert report["upright_noncontact_moving_high_knee_states"] == 1
    assert report["upright_noncontact_moving_knee_above_rad"] == {"1.2": 1, "1.5": 1, "2.0": 0}
    assert report["continuous_support_changes_left_right"] == [3, 3]
    assert report["layout_differences"] == {"shape_friction": {"expected_size": 33, "actual_size": None}}


def test_invalid_replay_metadata_is_rejected():
    with pytest.raises(ValueError, match="ring metadata"):
        replay_rows({"filled": 6, "capacity_per_env": 5, "cursor": 1})
