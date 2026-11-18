"""Behavioral and derivative checks for reference-free terrain locomotion."""

import torch

from instinctlab_supermimic.locomotion import event_gated_return, locomotion_terms, touchdown_events


def readout(contact, ground=None, points=None, velocity=None):
    points = torch.zeros(1, 2, 4, 3) if points is None else points
    ground = torch.zeros(1, 2, 4) if ground is None else ground
    return locomotion_terms(
        torch.tensor([contact], dtype=torch.float32),
        torch.ones(1, 2, 3),
        points,
        ground,
        torch.ones_like(ground, dtype=torch.bool),
        torch.tensor([[0.5, 0, 0]]) if velocity is None else velocity,
        0.02,
    )


def test_single_support_and_flight_and_standing():
    stance, _ = readout([1, 0])
    flight, _ = readout([0, 0])
    double, _ = readout([1, 1])
    assert stance["locomotion_single_support"] > double["locomotion_single_support"]
    assert flight["locomotion_flight"] < stance["locomotion_flight"]
    standing, _ = readout([1, 1], velocity=torch.zeros(1, 3))
    assert standing["locomotion_single_support"] == 0
    assert standing["locomotion_stand_contact"] == 0


def test_slope_support_and_edge_and_clearance_saturation():
    slope = torch.tensor([[[0.0, 0.02, 0.04, 0.06], [0.0, 0.02, 0.04, 0.06]]])
    points = torch.zeros(1, 2, 4, 3)
    points[..., 2] = slope
    matched, support = readout([1, 0], ground=slope, points=points)
    _, flat_support = readout([1, 0])
    torch.testing.assert_close(support, flat_support)
    edge = slope.clone()
    edge[..., 2:] -= 0.2
    poor, _ = readout([1, 0], ground=edge, points=points)
    assert poor["locomotion_support"] < matched["locomotion_support"]
    points[..., 2] = 0.1
    high, _ = readout([1, 0], points=points)
    points[..., 2] = 0.3
    higher, _ = readout([1, 0], points=points)
    assert high["locomotion_clearance"] == higher["locomotion_clearance"] == 0


def test_tgi_clearance_uses_foot_center_while_legacy_uses_lowest_sole_point():
    points = torch.zeros(1, 2, 15, 3)
    ground = torch.zeros(1, 2, 15)
    valid = torch.ones_like(ground, dtype=torch.bool)
    points[..., 7, 2] = 0.2
    points[..., 0, 2] = -0.04
    common = dict(
        contact=torch.zeros(1, 2),
        foot_velocity=torch.ones(1, 2, 3),
        sole_points=points,
        ground=ground,
        valid=valid,
        velocity=torch.tensor([[0.5, 0.0, 0.0]]),
        dt=0.02,
    )
    tgi, _ = locomotion_terms(**common, tgi_recipe=True)
    legacy, _ = locomotion_terms(**common, tgi_recipe=False)
    assert tgi["locomotion_clearance"].item() == 0
    assert legacy["locomotion_clearance"].item() < 0


def test_touchdown_events_ignore_existing_stance_and_detach():
    contact = torch.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    events = touchdown_events(torch.tensor([[1.0, 0.0], [0.0, 0.0]]), contact, torch.ones_like(contact), 0.3)
    assert events.tolist() == [False, True]
    assert not events.requires_grad


def test_event_return_cuts_future_reward_but_keeps_event_value_derivative():
    reward = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    value = torch.tensor([[10.0, 20.0, 30.0]], requires_grad=True)
    event = torch.tensor([[False, True, False]])
    result = event_gated_return(reward, torch.zeros_like(reward), value, event, 0.5)
    torch.testing.assert_close(result, torch.tensor([7.0]))
    result.sum().backward()
    torch.testing.assert_close(reward.grad, torch.tensor([[1.0, 0.5, 0.0]]))
    torch.testing.assert_close(value.grad, torch.tensor([[0.0, 0.25, 0.0]]))
    plain = event_gated_return(reward.detach(), torch.zeros_like(reward), value.detach(), torch.zeros_like(event), 0.5)
    torch.testing.assert_close(plain, torch.tensor([6.5]))


def test_exact_terrain_forward_preserves_edge_and_bilinear_gradient():
    from instinctlab.tasks.parkour.supermimic.terrain import TerrainHeightField

    field = TerrainHeightField(
        torch.tensor([[0.0, 0.2], [0.0, 0.2]]), torch.ones(2, 2, dtype=torch.bool), [0.0, 0.0], 1.0, [[0.0, 0.0]]
    )
    xy = torch.tensor([[0.25, 0.5]], requires_grad=True)
    smooth, _ = field.query(xy)
    exact, valid = field.query(xy, exact=True)
    assert valid.all() and exact.item() == 0
    torch.testing.assert_close(torch.autograd.grad(exact.sum(), xy)[0], torch.autograd.grad(smooth.sum(), xy)[0])


def test_support_field_pulls_foot_toward_stone_and_rejects_invalid_ground():
    from instinctlab.tasks.parkour.supermimic.terrain import TerrainHeightField

    heights = torch.full((41, 41), -0.5)
    heights[15:26, 15:26] = 0
    valid = torch.ones_like(heights, dtype=torch.bool)
    field = TerrainHeightField(heights, valid, [-1.0, -1.0], 0.05, [[0.0, 0.0]])
    field.build_support_field()
    xy = torch.tensor([[0.23, 0.0]], requires_grad=True)
    support = field.query_support(xy)
    gradient = torch.autograd.grad(support.sum(), xy)[0]
    assert gradient[0, 0] < 0  # ascent points toward the stone centre
    assert field.query_support(torch.zeros(1, 2)) > support
    assert field.query_support(torch.tensor([[3.0, 0.0]])).item() == 0
    field.valid[:] = False
    field.build_support_field()
    assert field.query_support(torch.zeros(1, 2)).item() == 0


def test_event_terminal_cancels_value_bootstrap():
    reward = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    value = torch.tensor([[10.0, 20.0, 30.0]], requires_grad=True)
    result = event_gated_return(reward, torch.tensor([[0.0, 1.0, 0.0]]), value, torch.ones(1, 3, dtype=torch.bool), 0.5)
    # Earliest event replaces the suffix with V(s1), independently of later steps.
    torch.testing.assert_close(result, torch.tensor([6.0]))
    terminal = event_gated_return(
        reward[:, 1:2], torch.ones(1, 1), value[:, 1:2], torch.ones(1, 1, dtype=torch.bool), 0.5
    )
    torch.testing.assert_close(terminal, torch.tensor([2.0]))
    gradient = torch.autograd.grad(terminal.sum(), value)[0]
    assert gradient.count_nonzero() == 0


def test_policy_update_uses_event_return_and_keeps_frozen_models_unchanged(monkeypatch):
    from types import SimpleNamespace

    from instinctlab.tasks.parkour.supermimic import native_policy

    class ScalarCritic(torch.nn.Linear):
        def forward(self, value):
            return super().forward(value).squeeze(-1)

    actor = torch.nn.Linear(1, 1, bias=False)
    critic = ScalarCritic(1, 1, bias=False)
    physics_model = torch.nn.Linear(1, 1)
    perception = torch.nn.Linear(1, 1)
    with torch.no_grad():
        actor.weight.fill_(1)
        critic.weight.fill_(1)
    task = SimpleNamespace(
        env=SimpleNamespace(cfg=SimpleNamespace(terrain_locomotion=True)),
        physics=SimpleNamespace(model=physics_model),
    )

    def imagined(*args, **kwargs):
        weight = actor.weight.reshape(1)
        rewards = torch.stack((weight, 10 * weight), 1)
        return dict(
            task_rewards=rewards,
            terrain_supported=torch.ones(1, 2, dtype=torch.bool),
            terminations=torch.zeros(1, 2),
            touchdown_events=torch.tensor([[True, False]]),
            value_features=torch.stack((0 * weight, weight, 20 * weight), 1)[..., None],
            reward_terms={"probe": rewards},
            perception_prior_ticks=0,
            phases=torch.zeros(1, 3),
        )

    monkeypatch.setattr(native_policy, "imagine_native", imagined)
    report = native_policy.update_native_policy(
        task,
        actor,
        perception,
        critic,
        None,
        SimpleNamespace(state=torch.zeros(1, 71)),
        torch.optim.SGD(actor.parameters(), lr=0.01),
        torch.optim.SGD(critic.parameters(), lr=0.01),
        native_policy.NativePolicyUpdateCfg(
            horizon=2, gamma=0.5, critic_epochs=0, critic_minibatches=1, touchdown_cut_threshold=0.3
        ),
    )
    # J = r0 + gamma V(s1) = 1.5*w; the later reward and terminal value have no influence.
    assert abs(report["actor_grad_norm"] - 1.5) < 1e-6
    assert abs(report["actor_loss"] + 1.5) < 1e-6
    assert report["touchdown_cut_fraction"] == 0.5
    for module in (critic, physics_model, perception):
        assert all(parameter.grad is None and parameter.requires_grad for parameter in module.parameters())
    torch.testing.assert_close(critic.weight, torch.ones_like(critic.weight))


def test_native_four_corner_clearance_queries_center_and_keeps_gradient():
    points = torch.zeros(1, 2, 4, 3)
    points[..., 2] = 0.2
    points.requires_grad_()
    ground = torch.full((1, 2, 4), 0.2)
    valid = torch.ones_like(ground, dtype=torch.bool)
    center_ground = torch.full((1, 2), 0.15, requires_grad=True)
    terms, _ = locomotion_terms(
        torch.ones(1, 2),
        torch.ones(1, 2, 3),
        points,
        ground,
        valid,
        torch.tensor([[0.5, 0.0, 0.0]]),
        0.02,
        tgi_recipe=True,
        clearance_ground=center_ground,
        clearance_valid=torch.ones(1, 2, dtype=torch.bool),
    )
    terms["locomotion_clearance"].sum().backward()
    assert (points.grad[..., 2] > 0).all()
    assert (center_ground.grad < 0).all()
