"""Reward audits must distinguish true derivatives from cut/surrogate paths."""

import torch

from instinctlab.tasks.parkour.supermimic.native_reward_audit import directional_reward_audit


def test_reward_audit_matches_analytic_derivatives_per_environment():
    actions = torch.linspace(-0.4, 0.7, 24, dtype=torch.float64).reshape(2, 3, 4)

    def objective(x):
        quadratic = x.square().sum((1, 2))
        linear = 0.3 * x.sum((1, 2))
        return dict(quadratic=quadratic, linear=linear, total=quadratic + linear)

    report = directional_reward_audit(objective, actions)
    torch.testing.assert_close(report["gradients"][:, 0], 2 * actions)
    torch.testing.assert_close(report["gradients"][:, 1], torch.full_like(actions, 0.3))
    for index in range(2):
        torch.testing.assert_close(report["finite_difference"][:, index], report["automatic"], rtol=1e-9, atol=1e-9)


def test_reward_audit_exposes_detached_reward_and_surrogate_backward():
    actions = torch.full((2, 3, 4), 0.2, dtype=torch.float64)

    def objective(x):
        detached = x.detach().sum((1, 2))
        surrogate = (2 * x - x.detach()).sum((1, 2))
        return dict(detached=detached, surrogate=surrogate, total=detached + surrogate)

    report = directional_reward_audit(objective, actions)
    assert report["summary"]["detached"]["gradient_rms"] == 0
    assert report["summary"]["detached"]["finite_difference"][0]["finite_difference_rms"] > 0
    assert report["summary"]["surrogate"]["finite_difference"][0]["absolute_error_rms"] > 0


def test_core_reward_gradients_through_native_readouts_match_finite_difference():
    from types import SimpleNamespace

    from test_hiking_native_task import fixture

    from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
    from instinctlab.tasks.parkour.supermimic.native_rewards import tgi_core_reward

    task, state, context = fixture()
    task.physics.recorder = SimpleNamespace(joint_names=list(G1_29DOF_POLICY_JOINT_ORDER_V1))
    command = torch.tensor([[0.4, 0.1, 0.2], [0.2, -0.1, 0.3]])
    previous = torch.full((2, 29), 0.1)
    actions = torch.linspace(-0.2, 0.3, 58).reshape(2, 1, 29)

    def objective(controls):
        action = controls[:, 0]
        # Known differentiable state response isolates the native reward readout
        # from world-model accuracy, which the runtime audit measures separately.
        response = (
            *state[:2],
            state[2] + 0.2 * action[:, :3],
            state[3] + 0.1 * action[:, 3:6],
            state[4] + 0.01 * action,
            state[5],
        )
        total, terms = tgi_core_reward(task, response, action, previous, command, context)
        return dict(**terms, total=total)

    report = directional_reward_audit(objective, actions)
    for name in ("tgi_track_linear", "tgi_track_angular", "tgi_action_rate", "total"):
        term = report["summary"][name]
        assert term["gradient_rms"] > 0
        assert term["finite_difference"][0]["absolute_error_rms"] < 5e-5
