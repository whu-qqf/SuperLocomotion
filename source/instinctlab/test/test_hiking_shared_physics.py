"""Shared F preserves the verified MJ equations and physical checkpoint layout."""

import io
import torch

import pytest
from instinctlab_supermimic.networks import WorldModel
from instinctlab_supermimic.physical import flatten_state, invariant_features, unflatten_state
from instinctlab_supermimic.rotations import quat_to_mat


def physical_state(batch=3):
    torch.manual_seed(19)
    return (
        torch.randn(batch, 3),
        quat_to_mat(torch.randn(batch, 4)),
        torch.randn(batch, 3),
        torch.randn(batch, 3),
        torch.randn(batch, 29),
        torch.randn(batch, 29),
    )


def test_physical_state_roundtrip_preserves_world_twists_and_joint_order():
    state = physical_state()
    packed = flatten_state(state)
    assert packed.shape == (3, 71)
    recovered = unflatten_state(packed)
    for actual, expected in zip(recovered, state):
        torch.testing.assert_close(actual, expected)
    assert invariant_features(state).shape == (3, 70)


def test_f_has_no_visual_input_or_output_in_dual_route():
    model = WorldModel(scan_dim=9, num_contact=5, xi_dim=59, hidden=(32, 32))
    state = physical_state()
    actions, context, scan = torch.randn(3, 29), torch.randn(3, 59), torch.randn(3, 9)
    model.in_norm.update(model.inputs(state, actions, context, scan))
    next_state, contacts, latent = model.step(state, actions, context, scan)
    assert model.z_dim == 0 and latent is None
    assert contacts.shape == (3, 5)
    assert model.net[-1].out_features == 70 + 5
    assert model.in_dim == 70 + 29 + 59 + 9
    assert all(torch.isfinite(value).all() for value in next_state)


def test_auxiliary_physical_readouts_use_their_own_units_and_action_gradients():
    model = WorldModel(9, 5, 59, hidden=(32, 32), aux_dim=58)
    state = physical_state()
    actions = torch.randn(3, 29, requires_grad=True)
    context, scan = torch.randn(3, 59), torch.randn(3, 9)
    labels = torch.arange(174.0).reshape(3, 58)
    model.aux_norm.update(labels)
    _, contacts, latent, auxiliary = model.step_with_aux(state, actions, context, scan)
    torch.testing.assert_close(auxiliary, labels.mean(0).expand_as(labels))
    assert contacts.shape == (3, 5) and latent is None
    with torch.no_grad():
        model.net[-1].weight[75:].normal_(std=0.01)
    auxiliary = model.step_with_aux(state, actions, context, scan)[3]
    action_gradient = torch.autograd.grad(auxiliary.square().mean(), actions)[0]
    assert torch.isfinite(action_gradient).all() and action_gradient.norm() > 0
    # Adding readouts must not add a visual channel to the physical state.
    assert model.z_dim == 0 and model.in_dim == 70 + 29 + 59 + 9


def test_shared_f_matches_archived_mj_values_checkpoint_and_action_gradients():
    original = pytest.importorskip("instinct_mj.rl.supermimic.networks")
    old = original.WorldModel(9, 5, 59, hidden=(32, 32))
    with torch.no_grad():
        old.net[-1].weight.normal_(std=0.01)
        old.out_norm.mean.normal_(std=0.01)
    serialized = io.BytesIO()
    torch.save(old.state_dict(), serialized)
    serialized.seek(0)
    new = WorldModel(9, 5, 59, hidden=(32, 32))
    new.load_state_dict(torch.load(serialized, weights_only=True), strict=True)
    state = physical_state()
    actions = torch.randn(3, 29, requires_grad=True)
    context, scan = torch.randn(3, 59), torch.randn(3, 9)
    before, after = old.step(state, actions, context, scan), new.step(state, actions, context, scan)
    for left, right in zip(before[0], after[0]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(before[1], after[1], rtol=0, atol=0)
    grad_old = torch.autograd.grad(before[0][4].square().sum(), actions)[0]
    grad_new = torch.autograd.grad(after[0][4].square().sum(), actions)[0]
    assert grad_new.norm() > 0
    torch.testing.assert_close(grad_new, grad_old, rtol=0, atol=0)
