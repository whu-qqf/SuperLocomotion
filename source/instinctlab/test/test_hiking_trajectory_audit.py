"""Trajectory review must preserve terminal failures and censoring boundaries."""

import hashlib
import json
import numpy as np
import torch

import pytest

from instinctlab.tasks.parkour.supermimic.trajectory_audit import audit_motions, audit_replay, episode_segments


def test_episode_segments_distinguish_censored_and_complete_intervals():
    done = torch.tensor([[True], [False], [False], [True], [False]])
    segments = episode_segments(done)
    assert [(s["start"], s["end"]) for s in segments] == [(0, 0), (1, 3), (4, 4)]
    assert segments[0]["left_censored"]
    assert not segments[1]["left_censored"] and not segments[1]["right_censored"]
    assert segments[2]["right_censored"]


def test_terminal_failure_union_and_reset_safe_contact_switches():
    replay = {"filled": 3, "capacity_per_env": 3, "cursor": 0}
    root = torch.tensor([[[0.0, 0.0, 0.8]], [[0.0, 0.0, 0.4]], [[0.0, 0.0, 0.8]]])
    quat = torch.tensor([[[0.0, 0.0, 1.0, 0.0]], [[1.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 1.0]]])
    contact = torch.zeros(3, 1, 21)
    contact[:2, :, 4:12] = 1
    contact[:, :, -9:] = 1
    velocity = torch.tensor([[[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]]] * 3)
    for name, tensor in {
        "root_pos_w": root,
        "root_quat_w": quat,
        "root_lin_vel_w": torch.zeros(3, 1, 3),
        "contact_state": contact,
        "body_link_lin_vel_w": velocity,
        "illegal_torso_contact": torch.tensor([[[0.0]], [[1.0]], [[0.0]]]),
        "task_reward_terms": torch.full((3, 1, 1), -0.016),
    }.items():
        replay["buffer/" + name + "@final"] = tensor
    replay["buffer/done"] = torch.tensor([[[0.0]], [[1.0]], [[0.0]]])
    replay["buffer/env_origin"] = torch.zeros(3, 1, 3)
    # Reset observations conceal the failure; only @final values are valid here.
    replay["buffer/root_pos_w"] = torch.full((3, 1, 3), 1.0)
    schema = {"reward_names": ["feet_slide"], "reward_weights": {"feet_slide": -0.4}}
    results, _ = audit_replay(replay, schema, ["left_ankle_roll_link", "right_ankle_roll_link"])
    assert results["done_events"] == 1
    assert results["terminal_flags"] == {
        "torso_contact": 1,
        "tilt_gt_1rad": 1,
        "height_lt_0.5m": 1,
        "physical_union": 1,
        "no_physical_flag": 0,
    }
    assert results["reset_safe_between_control_step_clock_switches"] == 0
    assert results["link_proxy_vs_native_slide_rmse_mps"] < 1e-6


def test_reference_pose_audit_rejects_modified_joints_even_with_new_clip_hash(tmp_path):
    source = tmp_path / "source.npz"
    arrays = {
        "framerate": np.array(50.0),
        "joint_names": np.array(["joint"]),
        "joint_pos": np.zeros((4, 1), dtype=np.float32),
        "base_pos_w": np.zeros((4, 3), dtype=np.float32),
        "base_quat_w": np.tile([1.0, 0.0, 0.0, 0.0], (4, 1)).astype(np.float32),
    }
    np.savez(source, **arrays)
    clip = tmp_path / "walk_train_retargetted.npz"
    translated = {**arrays, "base_pos_w": arrays["base_pos_w"] + np.array([1, 0, 0], dtype=np.float32)}
    np.savez(clip, **translated)
    entry = {
        "name": "walk_train",
        "frames": [0, 4],
        "translation_m": [1, 0, 0],
        "floor_p95_minus_p05_m": 0.0,
        "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
    }
    manifest = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "clips": [entry],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "train.yaml").write_text("selected_files:\n  - walk_train_retargetted.npz\n")
    assert audit_motions(tmp_path)["selected_clips"][0]["hash_and_source_pose_parity"]
    translated["joint_pos"] = np.ones((4, 1), dtype=np.float32)
    np.savez(clip, **translated)
    entry["sha256"] = hashlib.sha256(clip.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        audit_motions(tmp_path)
