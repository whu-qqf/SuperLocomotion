"""Perception contracts: temporal targets, real action order, reset and gradients."""

import io
import torch
from dataclasses import replace

import pytest
from instinctlab_perception.events import PerceptionRuntime
from instinctlab_perception.policy import PerceptivePolicy
from instinctlab_perception.replay import PerceptionReplay, collate_ticks, reconstruct_prefixes
from instinctlab_perception.rssm import PerceptionCfg, PerceptionModel


@pytest.fixture
def model():
    torch.manual_seed(42)
    return PerceptionModel(PerceptionCfg(deter=32, stoch=4, categories=4, embed=32, overshoot_steps=3))


def drive(model, controls=11, delay=0):
    runtime = PerceptionRuntime(model, 2, delay_steps=delay)
    replay = PerceptionReplay(2, max_ticks_per_env=20, holdout_fraction=0.5)
    features = []
    for step in range(controls):
        depth = torch.full((2, 1, 18, 32), step / 20)
        proprio = torch.full((2, 64), step / 100)
        captured = runtime.control_step.remainder(5) == 0
        feat, tick = runtime.before_action(depth, proprio, captured)
        if tick is not None:
            replay.add(tick)
        features.append(feat)
        runtime.after_action(torch.full((2, 29), float(step + 1)))
    return runtime, replay, features


def test_tick_contains_five_real_actions_in_order(model):
    runtime, replay, features = drive(model)
    history = replay.episodes[0][0]
    assert [int(r["control_step"]) for r in history] == [0, 5, 10]
    torch.testing.assert_close(history[1]["actions"][:, 0], torch.arange(1.0, 6.0))
    torch.testing.assert_close(history[2]["actions"][:, 0], torch.arange(6.0, 11.0))
    assert int(runtime.count[0]) == 1
    for step in (1, 2, 3, 4):
        torch.testing.assert_close(features[0], features[step], atol=0, rtol=0)
    assert not torch.equal(features[0], features[5])


def test_delayed_camera_keeps_capture_delivery_and_assimilation_distinct(model):
    _, replay, _ = drive(model, delay=1)
    history = replay.episodes[0][0]
    assert not history[0]["observed"]
    assert history[0]["depth"] is None
    assert history[1]["capture_step"] == 0
    assert history[1]["delivery_step"] == 1
    assert history[1]["control_step"] == 5
    assert history[1]["frame_id"] == 0
    assert float(history[1]["frame_age"]) == pytest.approx(0.1)
    assert history[2]["capture_step"] == 5
    assert history[2]["delivery_step"] == 6


def test_partial_reset_discards_pending_old_images_without_touching_other_env(model):
    runtime, replay, _ = drive(model, controls=3, delay=5)
    other = runtime.state.index(torch.tensor([1]))
    runtime.reset(torch.tensor([0]))
    assert not runtime.camera.occupied[0].any()
    torch.testing.assert_close(runtime.state.deter[1], other.deter[0], rtol=0, atol=0)
    assert runtime.control_step.tolist() == [0, 3]
    assert runtime.count.tolist() == [0, 3]
    feat, tick = runtime.before_action(torch.ones(2, 1, 18, 32), torch.zeros(2, 64), torch.tensor([True, False]))
    assert tick.env.tolist() == [0]
    assert tick.episode_id.tolist() == [1]
    assert tick.is_first.tolist() == [True]
    assert not tick.observed.any()  # Old episode's delayed frame cannot return.
    assert runtime.episode_id.tolist() == [1, 0]
    assert torch.isfinite(feat).all()


def test_prefix_replay_matches_live_state_and_recomputes_after_parameter_change(model):
    runtime, replay, _ = drive(model)
    prefixes = [replay.prefix(i, 0, 10) for i in range(2)]
    state = reconstruct_prefixes(model, prefixes)
    torch.testing.assert_close(state.deter, runtime.state.deter, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(state.logits, runtime.state.logits, atol=1e-6, rtol=1e-6)
    with torch.no_grad():
        model.posterior_head[-1].bias[0] += 1
    fresh = reconstruct_prefixes(model, prefixes)
    assert not torch.equal(state.logits, fresh.logits)
    assert not fresh.deter.requires_grad


def test_checkpoint_roundtrip_mid_interval_with_pending_delivery(model):
    runtime, replay, _ = drive(model, controls=3, delay=5)
    serialized = io.BytesIO()
    torch.save(dict(runtime=runtime.state_dict(), replay=replay.state_dict()), serialized)
    serialized.seek(0)
    checkpoint = torch.load(serialized, weights_only=False)
    restored = PerceptionRuntime(model, 2, delay_steps=5)
    restored.load_state_dict(checkpoint["runtime"])
    restored_replay = PerceptionReplay(2, max_ticks_per_env=20, holdout_fraction=0.5)
    restored_replay.load_state_dict(checkpoint["replay"])
    for _ in range(4):
        captured = runtime.control_step.remainder(5) == 0
        args = (torch.ones(2, 1, 18, 32), torch.zeros(2, 64), captured)
        f0, t0 = runtime.before_action(*args)
        f1, t1 = restored.before_action(*args)
        torch.testing.assert_close(f0, f1, atol=0, rtol=0)
        if t0 is not None:
            for name in vars(t0):
                torch.testing.assert_close(getattr(t0, name), getattr(t1, name), atol=0, rtol=0)
        action = torch.full((2, 29), 0.25)
        runtime.after_action(action)
        restored.after_action(action)
    assert len(restored_replay.episodes[0][0]) == 1


def test_current_image_changes_current_policy_feature(model):
    torch.manual_seed(12)
    initial = model.initial(2, sample=False)
    actions = torch.zeros(2, 5, 29)
    proprio = torch.zeros(2, 64)
    depth = torch.stack((torch.zeros(1, 18, 32), torch.ones(1, 18, 32)))
    post, _ = model.observe(
        initial,
        actions,
        depth,
        proprio,
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        sample=False,
    )
    assert (model.feature(post)[0] - model.feature(post)[1]).abs().max() > 1e-5
    torch.testing.assert_close(post.deter[0], post.deter[1], atol=1e-6, rtol=1e-6)


def test_missing_image_uses_prior_not_zero_image_posterior(model):
    initial = model.initial(1, sample=False)
    args = (initial, torch.zeros(1, 5, 29), torch.zeros(1, 1, 18, 32), torch.zeros(1, 64), torch.zeros(1))
    absent, prior = model.observe(*args, torch.tensor([False]), torch.tensor([False]), sample=False)
    seen, _ = model.observe(*args, torch.tensor([True]), torch.tensor([False]), sample=False)
    torch.testing.assert_close(absent.logits, prior.logits, rtol=0, atol=0)
    assert not torch.equal(absent.logits, seen.logits)


def test_perception_training_updates_encoder_prior_and_posterior(model):
    _, replay, _ = drive(model, controls=21)
    records = list(replay.episodes[0][0])
    batch = collate_ticks([records], model.cfg, "cpu")
    # Disable free-nats only for this gradient diagnostic to expose both KL paths.
    model.cfg = replace(model.cfg, free_nats=0)
    loss, metrics, _ = model.sequence_loss(batch, sample=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    before = {k: p.detach().clone() for k, p in model.named_parameters()}
    optimizer.zero_grad()
    loss.backward()
    for module in (model.image_encoder, model.transition_input, model.gru, model.prior_head, model.posterior_head):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(g.isfinite().all() for g in grads)
        assert sum(g.abs().sum() for g in grads) > 0
    assert metrics["overshoot"] > 0
    optimizer.step()
    assert any(not torch.equal(p, before[k]) for k, p in model.named_parameters())


def test_policy_input_is_detached_from_perception_but_projection_can_train(model):
    # Pass a feature with a graph: isolation is owned by the actual policy.
    feature = model.feature(model.initial(2, sample=False))
    policy = PerceptivePolicy(torch.nn.Linear(768 + 128, 29), model.cfg.feature_dim)
    policy(torch.zeros(2, 768), feature).square().sum().backward()
    assert policy.projection[0].weight.grad.abs().sum() > 0
    assert policy.actor.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())


def test_replay_never_samples_holdout_and_never_crosses_reset(model):
    runtime, replay, _ = drive(model, controls=6)
    replay.finish(torch.tensor([0]), torch.tensor([0]))
    runtime.reset(torch.tensor([0]))
    _, tick = runtime.before_action(torch.zeros(2, 1, 18, 32), torch.zeros(2, 64), torch.tensor([True, False]))
    replay.add(tick)
    samples = replay.sample(50, generator=torch.Generator().manual_seed(4))
    assert all(s.env == 0 for s in samples)
    for s in samples:
        assert all(int(r["episode_id"]) == s.episode_id for r in s.prefix + s.target)
        if s.prefix:
            assert int(s.prefix[0]["control_step"]) == 0
        else:
            assert s.target[0]["is_first"]
    assert all(s.env == 1 for s in replay.sample(5, holdout=True))


def test_replay_evicts_whole_episode_and_rejects_missing_prefix(model):
    replay = PerceptionReplay(2, max_ticks_per_env=1)
    # Read first tick by reconstructing from an independent fresh runtime.
    fresh = PerceptionRuntime(model, 2)
    args = (torch.zeros(2, 1, 18, 32), torch.zeros(2, 64), torch.ones(2, dtype=torch.bool))
    _, tick = fresh.before_action(*args)
    replay.add(tick)
    fresh.reset(torch.tensor([0, 1]))
    _, new_tick = fresh.before_action(*args)
    replay.add(new_tick)
    with pytest.raises(KeyError, match="prefix"):
        replay.prefix(0, 0, 0)
    new_tick.is_first[:] = False
    new_tick.episode_id[:] = 9
    with pytest.raises(ValueError, match="prefix"):
        replay.add(new_tick)


def test_padding_does_not_change_sequence_loss(model):
    _, replay, _ = drive(model, controls=11)
    records = replay.episodes[0][0]
    short = collate_ticks([records[:2]], model.cfg, "cpu")
    padded = collate_ticks([records[:2], records], model.cfg, "cpu")
    # Duplicate the short sequence and add one explicitly masked tick to each.
    padded = {name: value[:1] for name, value in padded.items()}
    plain_loss, _, _ = model.sequence_loss(short, sample=False)
    padded_loss, _, _ = model.sequence_loss(padded, sample=False)
    torch.testing.assert_close(plain_loss, padded_loss)
