"""Measured return labels stop at terminations and remain detached."""

import torch
from dataclasses import replace

import pytest

from instinctlab.tasks.parkour.supermimic.native_policy import NativePolicyUpdateCfg
from instinctlab.tasks.parkour.supermimic.native_runner import NativeRunnerCfg
from instinctlab.tasks.parkour.supermimic.native_value import real_value_loss


def test_real_value_terminal_target_and_bootstrap_gradient():
    weight = torch.tensor(2.0, requires_grad=True)
    critic = lambda x: x.sum(-1) * weight
    next_features = torch.tensor([[3.0]], requires_grad=True)
    returns = torch.tensor([4.0, 5.0], requires_grad=True)
    batch = dict(
        features=torch.tensor([[1.0], [2.0]]),
        returns=returns,
        bootstrap_indices=torch.tensor([1]),
        next_features=next_features,
        lengths=torch.tensor([1, 5]),
    )
    loss, metrics = real_value_loss(critic, batch, 0.5)
    target = torch.tensor([4.0, 5.0 + 0.5**5 * 6.0])
    expected = ((torch.tensor([1.0, 2.0]) * weight - target) ** 2).mean()
    torch.testing.assert_close(loss, expected)
    expected_grad = torch.autograd.grad(expected, weight)[0]
    loss.backward()
    torch.testing.assert_close(weight.grad, expected_grad)
    assert next_features.grad is None and returns.grad is None
    assert metrics["physical_terminal_fraction"] == 0.5


def test_real_only_critic_requires_tick_aligned_actor_and_real_update():
    policy = NativePolicyUpdateCfg(horizon=5, critic_epochs=0)
    with pytest.raises(ValueError, match="measured critic updates"):
        NativeRunnerCfg(policy=policy)
    cfg = NativeRunnerCfg(real_value_updates=2, policy=policy)
    assert cfg.real_value_horizon == 30
    with pytest.raises(ValueError, match="tick-aligned actor"):
        replace(cfg, policy=replace(policy, horizon=4))
    with pytest.raises(ValueError, match="perception tick"):
        replace(cfg, real_value_horizon=31)
