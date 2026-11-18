"""The published task boundary and shared sensor rotations stay independent of imitation."""

import gymnasium as gym
import torch

import pytest
from instinctlab_supermimic.quaternion_xyzw import multiply, rotate, rotate_inverse
from instinctlab_supermimic.rotations import mv, quat_to_mat

import instinctlab.tasks  # noqa: F401
from instinctlab.tasks.parkour.supermimic.native_latent import NativeLatentCfg


def test_only_hiking_tasks_are_registered():
    tasks = [name for name in gym.registry if name.startswith("Instinct-")]
    assert tasks
    assert all(name.startswith("Instinct-Parkour-Target-Amp-G1") for name in tasks)


@pytest.mark.parametrize(
    "kwargs", [{"use_amp": True}, {"tracking_reference_dir": "unused"}, {"paired_reference_dir": "unused"}]
)
def test_hiking_configuration_rejects_imitation_modes(kwargs):
    with pytest.raises(ValueError, match="reference-free Hiking"):
        NativeLatentCfg(**kwargs)


def test_sensor_rotations_match_matrix_values_and_gradients():
    generator = torch.Generator().manual_seed(42)
    raw = torch.randn(7, 4, generator=generator, dtype=torch.float64, requires_grad=True)
    q = torch.nn.functional.normalize(raw, dim=-1)
    vector = torch.randn(7, 3, generator=generator, dtype=torch.float64)
    actual = rotate(q, vector)
    expected = mv(quat_to_mat(q[:, [3, 0, 1, 2]]), vector)
    torch.testing.assert_close(actual, expected)
    grad_actual = torch.autograd.grad(actual.sum(), raw, retain_graph=True)[0]
    grad_expected = torch.autograd.grad(expected.sum(), raw)[0]
    torch.testing.assert_close(grad_actual, grad_expected)
    torch.testing.assert_close(rotate_inverse(q, actual), vector)
    other = torch.nn.functional.normalize(torch.randn(7, 4, generator=generator, dtype=torch.float64), dim=-1)
    torch.testing.assert_close(rotate(multiply(q, other), vector), rotate(q, rotate(other, vector)))
