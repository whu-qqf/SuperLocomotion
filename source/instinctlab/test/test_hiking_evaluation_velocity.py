"""Evaluation velocity matches Hiking's full-body root COM convention."""

import math
import torch

from instinctlab.tasks.parkour.supermimic.native_latent_evaluation import (
    root_com_body_velocity,
    swing_clearance_summary,
)


def test_tilted_root_velocity_includes_com_shift_and_full_body_rotation():
    state = torch.zeros(1, 71)
    state[:, 3] = math.cos(math.pi / 4)
    state[:, 5] = math.sin(math.pi / 4)  # 90 degrees about world Y
    state[:, 7] = 1  # root-link origin velocity along world X
    state[:, 11] = 1  # angular velocity about world Y
    measured = root_com_body_velocity(state, torch.tensor([[0.1, 0.0, 0.0]]))
    # COM offset points down in world coordinates: omega x offset reduces X
    # velocity by .1; the full frame transform places it on body Z, not X.
    torch.testing.assert_close(measured, torch.tensor([[0.0, 0.0, 0.9]]), atol=1e-6, rtol=1e-6)


def test_upright_com_velocity_differs_from_link_origin_under_rotation():
    state = torch.zeros(1, 71)
    state[:, 3] = 1
    state[:, 7] = 0.5
    state[:, 12] = 2
    measured = root_com_body_velocity(state, torch.tensor([[0.1, 0.0, 0.0]]))
    torch.testing.assert_close(measured, torch.tensor([[0.5, 0.2, 0.0]]))
    torch.testing.assert_close(root_com_body_velocity(state, torch.zeros(1, 3)), state[:, 7:10])


def test_swing_clearance_excludes_jumps_standing_resets_and_invalid_terrain():
    clearance = torch.tensor([[[0.0, 0.02]], [[0.12, 0.0]], [[0.5, 0.5]], [[0.0, 0.9]], [[0.0, 0.9]], [[0.0, 0.9]]])
    contact = torch.tensor([[[1, 0]], [[0, 1]], [[0, 0]], [[1, 0]], [[1, 0]], [[1, 0]]], dtype=torch.bool)
    alive = torch.tensor([[1], [1], [1], [1], [0], [1]], dtype=torch.bool)
    command = torch.zeros(6, 1, 3)
    command[..., 0] = 0.5
    command[3] = 0  # Standing must not count as walking swing.
    valid = torch.ones_like(contact)
    valid[5, 0, 1] = False
    report = swing_clearance_summary(clearance, valid, contact, alive, command)
    assert report["sample_count"] == 2
    assert report["invalid_terrain_sample_count"] == 1
    assert report["per_foot_sample_count"] == [1, 1]
    assert abs(report["mean_m"] - 0.07) < 1e-6
    assert report["below_8cm_fraction"] == 0.5
    assert abs(report["p10_p50_p90_m"][1] - 0.07) < 1e-6


def test_no_single_support_reports_missing_clearance_rather_than_success():
    report = swing_clearance_summary(
        torch.full((2, 1, 2), 0.5),
        torch.ones(2, 1, 2, dtype=torch.bool),
        torch.zeros(2, 1, 2, dtype=torch.bool),  # Both feet airborne.
        torch.ones(2, 1, dtype=torch.bool),
        torch.tensor([[[0.5, 0.0, 0.0]], [[0.5, 0.0, 0.0]]]),
    )
    assert report["sample_count"] == 0
    assert report["mean_m"] is None
    assert report["below_8cm_fraction"] is None
