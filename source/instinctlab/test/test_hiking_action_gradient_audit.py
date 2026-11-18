"""A gradient check must reject wrong signs and inconclusive simulator noise."""

import torch

import pytest

from instinctlab.tasks.parkour.supermimic.action_gradient_audit import (
    compare_probe_derivatives,
    first_action_derivative,
)


def prediction(actions):
    return actions.square().cumsum(1)[..., :2]


def fixture():
    actions = torch.full((2, 3, 2), 0.5)
    direction = torch.tensor([[1.0, -0.5], [-0.5, 1.0]])
    outcomes = {}
    for name, epsilon in (("zero_a", 0), ("zero_0", 0), ("plus_0.05", 0.05), ("minus_0.05", -0.05)):
        applied = actions.clone()
        applied[:, 0] += epsilon * direction
        outcomes[name] = dict(states=prediction(applied), done=torch.zeros(2, 3, dtype=torch.bool))
    return actions, direction, dict(outcomes=outcomes, restore_max_state_errors=torch.zeros(4, 2))


def test_directional_derivative_agrees_with_physical_centered_difference_and_exposes_wrong_sign():
    actions, direction, probes = fixture()
    derivative = first_action_derivative(prediction, actions, direction)
    torch.testing.assert_close(derivative, direction[:, None].expand(-1, 3, -1))
    result = compare_probe_derivatives(probes, lambda states: states, derivative, epsilons=(0.05,))
    report = result["perturbations"][0]
    assert report["resolved_components"] == 12
    assert report["sign_agreement"] == 1
    assert report["relative_error"] < 1e-5
    wrong = compare_probe_derivatives(probes, lambda states: states, -derivative, epsilons=(0.05,))
    assert wrong["perturbations"][0]["sign_agreement"] == 0
    assert wrong["perturbations"][0]["cosine"] == pytest.approx(-1, abs=1e-6)
    assert not actions.requires_grad


def test_replay_noise_makes_small_signal_inconclusive():
    actions, direction, probes = fixture()
    probes["outcomes"]["zero_0"]["states"] += 1
    derivative = first_action_derivative(prediction, actions, direction)
    report = compare_probe_derivatives(probes, lambda states: states, derivative, epsilons=(0.05,))["perturbations"][0]
    assert report["resolved_components"] == 0
    assert report["status"] == "inconclusive"
    assert report["cosine"] is None and report["sign_agreement"] is None


def test_only_targets_after_a_termination_are_excluded():
    actions, direction, probes = fixture()
    probes["outcomes"]["plus_0.05"]["done"][0, 0] = True
    derivative = first_action_derivative(prediction, actions, direction)
    report = compare_probe_derivatives(probes, lambda states: states, derivative, epsilons=(0.05,))["perturbations"][0]
    assert report["valid_components"] == 8  # Env 0: terminal target only; env 1: all three.
    assert report["resolved_components"] == 8


def test_missing_repeats_and_failed_restore_cannot_pass():
    actions, direction, probes = fixture()
    derivative = first_action_derivative(prediction, actions, direction)
    probes["restore_max_state_errors"][0, 0] = 1e-5
    with pytest.raises(ValueError, match="restored exactly"):
        compare_probe_derivatives(probes, lambda states: states, derivative, epsilons=(0.05,))
    probes["restore_max_state_errors"].zero_()
    del probes["outcomes"]["zero_0"]
    with pytest.raises(ValueError, match="two unperturbed"):
        compare_probe_derivatives(probes, lambda states: states, derivative, epsilons=(0.05,))


def test_contact_strata_exclude_other_components_and_report_missing_samples():
    actions, direction, probes = fixture()
    derivative = first_action_derivative(prediction, actions, direction)
    selected = torch.zeros_like(derivative, dtype=torch.bool)
    selected[0, :2, 1] = True
    report = compare_probe_derivatives(
        probes, lambda states: states, derivative, epsilons=(0.05,), component_mask=selected
    )["perturbations"][0]
    assert report["valid_components"] == report["resolved_components"] == 2
    assert report["sign_agreement"] == 1
    empty = compare_probe_derivatives(
        probes, lambda states: states, derivative, epsilons=(0.05,), component_mask=torch.zeros_like(selected)
    )["perturbations"][0]
    assert empty["valid_components"] == empty["resolved_components"] == 0
    assert empty["status"] == "inconclusive" and empty["cosine"] is None
