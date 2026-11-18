"""Real history ownership, terminal targets and current-weight P seed replay."""

import io
import torch
from types import SimpleNamespace as NS

import pytest
from instinctlab_perception.events import PerceptionRuntime
from instinctlab_perception.replay import PerceptionReplay, reconstruct_prefixes
from instinctlab_perception.rssm import PerceptionCfg, PerceptionModel

from instinctlab.tasks.parkour.supermimic.native_task import AMP_WIDTHS, POLICY_WIDTHS, flatten_history
from instinctlab.tasks.parkour.supermimic.native_training_replay import NativeTrainingReplay, history_frames


def collect(controls=12, capacity=30):
    torch.manual_seed(17)
    model = PerceptionModel(PerceptionCfg(deter=16, stoch=2, categories=4, embed=16))
    runtime = PerceptionRuntime(model, 2)
    p_replay = PerceptionReplay(2, holdout_fraction=0.5)
    replay = NativeTrainingReplay(2, capacity_steps=capacity)
    sensor = NS(current_air_time=NS(torch=torch.zeros(2, 2)), current_contact_time=NS(torch=torch.ones(2, 2)))
    state = torch.zeros(2, 71)
    previous = torch.zeros(2, 29)
    command = dict(time_left=torch.full((2,), 5.0), velocity=torch.zeros(2, 3))
    task = NS(
        env=NS(scene={"contact_forces": NS(data=sensor)}, action_manager=NS(action=previous)),
        physics=NS(recorder=NS(state=lambda: state)),
        foot_contact_ids=[0, 1],
        context=lambda: {"default_joint_pos": torch.zeros(2, 29)},
        command=lambda: command,
        sensor_context=lambda: [{"drift": torch.zeros(2, 3)}],
    )
    histories = {}
    for step in range(controls):
        _, tick = runtime.before_action(
            torch.full((2, 1, 18, 32), step / 20),
            torch.zeros(2, 64),
            runtime.control_step.remainder(5) == 0,
        )
        if tick is not None:
            p_replay.add(tick)
        policy = torch.randn(2, 8, 96)  # Stand in for actual noisy manager histories.
        amp = torch.randn(2, 10, 67)
        context = torch.full((2, 4), float(step))
        state.fill_(step)
        if tick is not None:
            for index, env in enumerate(tick.env.tolist()):
                histories[env, int(tick.episode_id[index]), int(tick.control_step[index])] = policy[env].clone()
        replay.add_starts(tick, task, context, flatten_history(policy, POLICY_WIDTHS), flatten_history(amp, AMP_WIDTHS))
        # Mutating live buffers cannot change a retained start.
        policy.fill_(999)
        action = torch.full((2, 29), step + 0.5)
        terminated = torch.tensor([step == 3, False])
        episode = runtime.episode_id.clone()
        age = runtime.control_step.clone()
        transition = NS(
            state=state,
            next_state=state + 0.25,
            action=action,
            episode_step=age,
            terminated=terminated,
            truncated=torch.zeros(2, dtype=torch.bool),
        )
        replay.add_transition(
            transition,
            context,
            episode,
            torch.zeros(2, 62),
            torch.zeros(2, 58),
            task_reward=torch.full((2,), step + 0.125),
        )
        state.fill_(999)
        runtime.after_action(action)
        ids = terminated.nonzero(as_tuple=True)[0]
        if len(ids):
            p_replay.finish(ids, episode[ids])
            runtime.reset(ids)
        previous.copy_(action)
    return replay, p_replay, model, histories


def test_native_history_inverse_preserves_every_term_frame_and_noise():
    for widths, length in ((POLICY_WIDTHS, 8), (AMP_WIDTHS, 10)):
        frames = torch.randn(3, length, sum(widths))
        torch.testing.assert_close(
            history_frames(flatten_history(frames, widths), widths, length), frames, rtol=0, atol=0
        )
    with pytest.raises(ValueError, match="layout"):
        history_frames(torch.zeros(2, 96), POLICY_WIDTHS, 8)


def test_native_collection_schedule_matches_utd_and_is_resume_stable():
    from instinctlab.tasks.parkour.supermimic.native_runner import NativeDualRunner, NativeRunnerCfg

    runner = NativeDualRunner.__new__(NativeDualRunner)
    runner.cfg = NativeRunnerCfg()
    runner.replay = NativeTrainingReplay(2)
    runner.iteration = 0
    assert runner.collection_steps() == 64
    runner.replay.controls = 64
    counts = []
    for iteration in range(100):
        runner.iteration = iteration
        counts.append(runner.collection_steps())
    assert counts == [2, 3] * 50
    assert sum(counts) == 250  # 100 updates / 250 controls = 0.4 UTD.
    restored = NativeDualRunner.__new__(NativeDualRunner)
    restored.cfg, restored.replay, restored.iteration = runner.cfg, runner.replay, 73
    assert restored.collection_steps() == counts[73]


@pytest.mark.parametrize("num_envs", [1024, 2048, 4096])
def test_physical_replay_ratio_preserves_sample_budget_across_env_counts(num_envs):
    from instinctlab.tasks.parkour.supermimic.native_runner import NativeDualRunner, NativeRunnerCfg

    runner = NativeDualRunner.__new__(NativeDualRunner)
    runner.cfg = NativeRunnerCfg(physical_replay_ratio=16.0)
    runner.replay = NativeTrainingReplay(num_envs)
    runner.iteration = 0
    assert runner.collection_steps() == 64
    runner.replay.controls = 64
    counts = []
    for iteration in range(160):
        runner.iteration = iteration
        counts.append(runner.collection_steps())
    sampled = 160 * runner.cfg.physical_updates * runner.cfg.physical_batch * runner.cfg.physical_horizon
    assert sampled / (sum(counts) * num_envs) == 16.0
    assert 0 in counts  # Multiple updates can reuse the same collected controls.
    restored = NativeDualRunner.__new__(NativeDualRunner)
    restored.cfg, restored.replay = runner.cfg, runner.replay
    resumed = []
    for iteration in range(73, 160):
        restored.iteration = iteration
        resumed.append(restored.collection_steps())
    assert resumed == counts[73:]


@pytest.mark.parametrize("ratio", [0.0, -1.0, float("nan"), float("inf")])
def test_physical_replay_ratio_rejects_invalid_budgets(ratio):
    from instinctlab.tasks.parkour.supermimic.native_runner import NativeRunnerCfg

    with pytest.raises(ValueError, match="finite and positive"):
        NativeRunnerCfg(physical_replay_ratio=ratio)


def test_physical_replay_ratio_rejects_fixed_collection_override():
    from instinctlab.tasks.parkour.supermimic.native_runner import NativeRunnerCfg

    with pytest.raises(ValueError, match="mutually exclusive"):
        NativeRunnerCfg(physical_replay_ratio=16.0, collect_steps=2)


def test_zero_control_iteration_never_steps_simulation_or_invents_tracking():
    from instinctlab.tasks.parkour.supermimic.native_runner import NativeDualRunner

    runner = NativeDualRunner.__new__(NativeDualRunner)
    runner.replay = NS(controls=64)
    runner.episode_statistics = NS(summary=lambda: {"completed_episodes": 0})
    result = runner.collect(0)
    assert result == dict(
        resets=0, collected_controls=0, real_controls=64, real_tracking={}, episode={"completed_episodes": 0}
    )


def test_new_reward_initialization_keeps_fresh_critic_optimizers_and_replay(tmp_path):
    from instinctlab.tasks.parkour.supermimic.native_runner import NativeDualRunner

    names = ("perception", "physical", "policy", "failure", "style", "critic")
    runner = NativeDualRunner.__new__(NativeDualRunner)
    runner.iteration = 0
    runner.replay = NativeTrainingReplay(2)
    runner.update_counts = dict.fromkeys(names, 0)
    runner.contract = dict(physics="same", task_rewards="new")
    runner.modules = {name: torch.nn.Linear(2, 2) for name in names}
    runner.optimizers = {name: torch.optim.Adam(module.parameters()) for name, module in runner.modules.items()}
    runner.refresh_runtime = lambda: None
    critic = {k: v.clone() for k, v in runner.modules["critic"].state_dict().items()}
    incoming = {
        name: {k: torch.ones_like(v) for k, v in module.state_dict().items()} for name, module in runner.modules.items()
    }
    path = tmp_path / "old.pt"
    torch.save(
        dict(format="native_hiking_dual_v1", contract=dict(physics="same"), iteration=8300, modules=incoming), path
    )
    result = runner.initialize_modules(path)
    assert result["source_iteration"] == 8300
    assert runner.iteration == runner.replay.controls == 0
    assert not any(runner.update_counts.values())
    assert all(not opt.state for opt in runner.optimizers.values())
    for name, module in runner.modules.items():
        expected = critic if name == "critic" else incoming[name]
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[key])
    runner.contract["physics"] = "different"
    with pytest.raises(ValueError, match="contract differs"):
        runner.initialize_modules(path)


def test_uniform_failure_loss_is_stationary_at_true_prior_and_not_cost_weighted_prior():
    from instinctlab.tasks.parkour.supermimic.native_runner import native_failure_loss

    # A constant head has no predictive information. Its optimum must equal
    # the real termination frequency, since its sigmoid drives survival.
    labels = torch.zeros(1000, dtype=torch.bool)
    labels[:10] = True
    prior = labels.float().mean()
    logit = torch.logit(prior).detach().requires_grad_(True)
    loss = native_failure_loss(logit.expand(1000), logit.expand(1000), labels)
    derivative = torch.autograd.grad(loss, logit)[0]
    torch.testing.assert_close(derivative, torch.zeros_like(derivative), atol=1e-7, rtol=0)
    biased_prior = 10 * prior / (1 - prior + 10 * prior)
    logit = torch.logit(biased_prior).detach().requires_grad_(True)
    loss = native_failure_loss(logit.expand(1000), logit.expand(1000), labels)
    assert torch.autograd.grad(loss, logit)[0] > 0.1
    assert float((1 - biased_prior).pow(32)) < float((1 - prior).pow(32)) / 10


def test_physical_windows_keep_terminal_targets_and_never_cross_reset():
    replay, _, _, _ = collect()
    batch = replay.sample_physics(256, 8, (0,), "cpu", torch.Generator().manual_seed(5))
    assert batch["mask"][:, 0].all()
    assert not (batch["mask"][:, 1:] & ~batch["mask"][:, :-1]).any()
    initial = batch["state"][:, 0, 0].long()
    assert set(initial.tolist()) == set(range(12))
    for row, start in enumerate(initial.tolist()):
        count = min(4 - start if start <= 3 else 12 - start, 8)
        assert int(batch["mask"][row].sum()) == count
        expected = torch.arange(start, start + count).float() + 0.25
        torch.testing.assert_close(batch["next_state"][row, :count, 0], expected)
    # The terminal target is step 3.25; step 4 is the new episode, never its replacement.
    terminal = initial == 3
    assert batch["mask"][terminal, 0].all() and not batch["mask"][terminal, 1:].any()
    assert (batch["next_state"][terminal, 0, 0] == 3.25).all()


def test_pre_reset_rewards_survive_checkpoint_and_legacy_rewards_remain_unknown():
    replay, _, _, _ = collect()
    # Env zero terminates at control 3. The terminal reward belongs to that
    # transition, not the first control of the following episode.
    assert replay.rows[3]["terminated"][0]
    torch.testing.assert_close(replay.rows[3]["task_reward"], torch.full((2,), 3.125))
    assert replay.rows[3]["task_reward_valid"].all()
    buffer = io.BytesIO()
    torch.save(replay.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=False)
    restored = NativeTrainingReplay(2, capacity_steps=30)
    restored.load_state_dict(saved)
    torch.testing.assert_close(restored.rows[3]["task_reward"], replay.rows[3]["task_reward"])
    assert restored.rows[3]["task_reward_valid"].all()
    # Old checkpoints contain no measured returns; zero is only padding and
    # cannot be consumed as a zero-valued reward label.
    for row in saved["rows"]:
        del row["task_reward"]
        del row["task_reward_valid"]
    restored.load_state_dict(saved)
    assert all(not row["task_reward_valid"].any() for row in restored.rows)
    assert all("task_reward" not in row for row in saved["rows"])


def value_fixture():
    replay, p_replay, model, _ = collect(controls=24)
    for i, row in enumerate(replay.rows):
        row["learning"] = dict(policy_window=torch.full((2, 670), float(i)))
    return replay, p_replay, model


def test_real_value_windows_keep_terminal_rewards_and_exclude_missing_or_timeout_data():
    replay, p_replay, _ = value_fixture()
    windows = replay.sample_value_windows(p_replay, 64, 5, (0, 1), torch.Generator().manual_seed(5))
    terminal = [
        w
        for w in windows
        if w["start"]["env"] == 0 and w["start"]["control_step"] == 0 and w["start"]["collection_step"] == 0
    ]
    assert terminal and all(w["end"] is None for w in terminal)
    torch.testing.assert_close(terminal[0]["task_reward"], torch.arange(4.0) + 0.125)
    assert terminal[0]["terminal_state"][0] == 3.25
    for window in windows:
        if window["end"] is not None:
            assert window["end"]["episode_id"] == window["start"]["episode_id"]
            assert window["end"]["control_step"] - window["start"]["control_step"] == 5
    replay.rows[3]["truncated"][0] = True
    for row in replay.rows:
        row["task_reward_valid"][1] = False
    windows = replay.sample_value_windows(p_replay, 32, 5, (0, 1), torch.Generator().manual_seed(5))
    assert all(w["start"]["env"] == 0 and w["start"]["collection_step"] >= 4 for w in windows)
    for row in replay.rows:
        row["task_reward_valid"].zero_()
    with pytest.raises(RuntimeError, match="No measured"):
        replay.sample_value_windows(p_replay, 4, 5, (0, 1))


def test_real_value_batch_corrects_terminal_amp_and_only_trains_critic():
    from instinctlab_supermimic.networks import Critic

    from instinctlab.tasks.parkour.supermimic.native_value import real_value_batch, real_value_loss

    replay, p_replay, model = value_fixture()
    windows = replay.sample_value_windows(p_replay, 64, 5, (0, 1), torch.Generator().manual_seed(5))
    physical_scale = torch.nn.Parameter(torch.ones(()))
    style_scale = torch.nn.Parameter(torch.tensor(0.5))
    task = NS(
        critic_features=lambda state, command, timers, context: state[0] * physical_scale,
        amp_frame=lambda state, context: state[0][:, :1].expand(-1, 67),
    )
    style = NS(reward=lambda window: window.mean(-1) * style_scale)
    batch = real_value_batch(task, model, p_replay, style, windows, 0.9)
    assert not batch["features"].requires_grad and not batch["returns"].requires_grad
    for i, window in enumerate(windows):
        expected_history = flatten_history(window["start"]["values"]["policy_history"][None], POLICY_WIDTHS)[0]
        torch.testing.assert_close(batch["features"][i, 3:771], expected_history)
        if window["start"]["collection_step"] == 0 and window["start"]["env"] == 0:
            expected = sum(0.9**j * (j + 0.125 + (0.5 * j if j < 3 else 0.5 * (9 * 2 + 3.25) / 10)) for j in range(4))
            torch.testing.assert_close(batch["returns"][i], torch.tensor(expected))
    before = batch["features"].clone()
    with torch.no_grad():
        model.posterior_head[-1].bias[0] += 1.0
    changed = real_value_batch(task, model, p_replay, style, windows, 0.9)
    assert not torch.equal(before, changed["features"])
    critic = Critic(batch["features"].shape[-1])
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4)
    before = [p.detach().clone() for p in critic.parameters()]
    loss, metrics = real_value_loss(critic, batch, 0.9)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(a, b) for a, b in zip(before, critic.parameters()))
    assert all(p.grad is None for p in model.parameters())
    assert physical_scale.grad is None and style_scale.grad is None
    assert 0 < metrics["physical_terminal_fraction"] < 1


@pytest.mark.parametrize("reward", [torch.zeros(2, 1), torch.tensor([0.0, float("nan")])])
def test_real_reward_labels_reject_shape_or_nonfinite_values(reward):
    replay, _, _, _ = collect(controls=1)
    row = replay.rows[-1]
    transition = NS(
        state=row["state"],
        next_state=row["next_state"],
        action=row["actions"],
        episode_step=row["control_step"] + 1,
        terminated=row["terminated"],
        truncated=row["truncated"],
    )
    with pytest.raises(ValueError, match="finite per-environment scalars"):
        replay.add_transition(
            transition,
            row["context"],
            row["episode_id"],
            row["contacts"],
            row["auxiliary"],
            task_reward=reward,
        )


def test_starts_use_complete_current_p_prefix_and_exact_real_history():
    replay, p_replay, model, histories = collect()
    # Remove initial starts so every sample must incorporate five actual actions.
    replay.starts = type(replay.starts)(r for r in replay.starts if r["control_step"] == 5)
    seed = replay.sample_starts(p_replay, model, 4, 32, 0.02)
    assert seed.control_step.tolist() == [5] * 4
    expected = histories[0, 1, 5]
    torch.testing.assert_close(seed.policy_history, expected.expand(4, -1, -1), rtol=0, atol=0)
    prefix = p_replay.prefix(0, 1, 5)
    assert [int(r["control_step"]) for r in prefix] == [0, 5]
    rebuilt = reconstruct_prefixes(model, [prefix] * 4)
    torch.testing.assert_close(seed.perception_state.logits, rebuilt.logits)
    with torch.no_grad():
        model.posterior_head[-1].bias[0] += 1.0
    fresh = replay.sample_starts(p_replay, model, 4, 32, 0.02)
    assert not torch.equal(seed.perception_state.logits, fresh.perception_state.logits)
    assert not fresh.perception_state.deter.requires_grad
    assert (fresh.state[:, 0] == 9).all()  # Episode 1's step 5 was collection step 9.
    assert (fresh.physical_context == 9).all()


def test_seed_eviction_command_horizon_and_checkpoint_roundtrip():
    replay, p_replay, model, _ = collect(capacity=4)
    assert all(r["collection_step"] >= 8 for r in replay.starts)
    stream = io.BytesIO()
    torch.save(replay.state_dict(), stream)
    stream.seek(0)
    restored = NativeTrainingReplay(2, capacity_steps=4)
    restored.load_state_dict(torch.load(stream, weights_only=False))
    generator = lambda: torch.Generator().manual_seed(21)
    left = replay.sample_starts(p_replay, model, 3, 32, 0.02, generator())
    right = restored.sample_starts(p_replay, model, 3, 32, 0.02, generator())
    torch.testing.assert_close(left.policy_history, right.policy_history, rtol=0, atol=0)
    torch.testing.assert_close(left.perception_state.logits, right.perception_state.logits, rtol=0, atol=0)
    for record in restored.starts:
        record["values"]["command"]["time_left"] = torch.tensor(0.64)
    with pytest.raises(RuntimeError, match="command horizon"):
        restored.sample_starts(p_replay, model, 3, 32, 0.02)
    for record in restored.starts:
        record["values"]["command"]["time_left"] = torch.tensor(5.0)
    p_replay.episodes[0].clear()
    with pytest.raises(RuntimeError, match="complete P prefixes"):
        restored.sample_starts(p_replay, model, 3, 32, 0.02)


def test_physical_replay_rejects_discontinuous_episode_clock():
    replay, _, _, _ = collect()
    row = replay.rows[-1]
    transition = NS(
        state=row["state"],
        next_state=row["next_state"],
        action=row["actions"],
        episode_step=row["control_step"] + 2,
        terminated=row["terminated"],
        truncated=row["truncated"],
    )
    with pytest.raises(ValueError, match="clocks"):
        replay.add_transition(transition, row["context"], row["episode_id"], row["contacts"], row["auxiliary"])


def test_resume_segment_is_a_physical_window_boundary_without_fabricated_failure():
    replay, _, _, _ = collect(controls=3)
    replay.begin_collection_segment()
    old = replay.rows[-1]
    transition = NS(
        state=torch.full((2, 71), 100.0),
        next_state=torch.full((2, 71), 100.25),
        action=old["actions"],
        episode_step=torch.zeros(2, dtype=torch.long),
        terminated=torch.zeros(2, dtype=torch.bool),
        truncated=torch.zeros(2, dtype=torch.bool),
    )
    replay.add_transition(transition, old["context"], old["episode_id"] + 1, old["contacts"], old["auxiliary"])
    batch = replay.sample_physics(128, 4, (0,), "cpu", torch.Generator().manual_seed(8))
    last_old = batch["state"][:, 0, 0] == 2
    assert last_old.any() and (batch["mask"][last_old].sum(-1) == 1).all()
    assert not replay.rows[-2]["terminated"].any() and not replay.rows[-2]["truncated"].any()


def test_runner_refresh_rebuilds_live_belief_but_keeps_pending_actions_and_camera():
    from instinctlab.tasks.parkour.supermimic.native_runner import NativeDualRunner

    model = PerceptionModel(PerceptionCfg(deter=16, stoch=2, categories=4, embed=16))
    runtime = PerceptionRuntime(model, 2)
    replay = PerceptionReplay(2)
    for step in range(7):
        _, tick = runtime.before_action(
            torch.zeros(2, 1, 18, 32), torch.zeros(2, 64), runtime.control_step.remainder(5) == 0
        )
        if tick is not None:
            replay.add(tick)
        runtime.after_action(torch.full((2, 29), float(step)))
    runtime.reset(torch.tensor([0]))  # A fresh episode at the update boundary.
    clocks = {name: getattr(runtime, name).clone() for name in ("actions", "count", "control_step", "episode_id")}
    camera = runtime.camera.state_dict()
    old = runtime.state.logits.clone()
    with torch.no_grad():
        model.posterior_head[-1].bias[0] += 1.0
        model.initial_deter.add_(0.2)
    runner = NativeDualRunner.__new__(NativeDualRunner)
    runner.runtime, runner.perception_replay, runner.perception = runtime, replay, model
    runner.refresh_runtime()
    assert not torch.equal(runtime.state.logits[1], old[1])
    expected = reconstruct_prefixes(model, [replay.prefix(1, 0, 5)])
    torch.testing.assert_close(runtime.state.logits[1], expected.logits[0])
    torch.testing.assert_close(runtime.state.deter[0], model.initial(1, sample=False).deter[0])
    for name, value in clocks.items():
        torch.testing.assert_close(getattr(runtime, name), value, rtol=0, atol=0)
    for name, value in camera.items():
        torch.testing.assert_close(runtime.camera.state_dict()[name], value, rtol=0, atol=0)


def test_shared_style_fits_real_data_and_keeps_imagined_input_gradient():
    from instinctlab_supermimic.style import StyleReward

    from instinctlab.tasks.parkour.supermimic.native_imagination import frozen_parameters

    style = StyleReward(
        {"amp": (670,)},
        "cpu",
        dict(hidden_sizes=[32, 16], nonlinearity="ReLU"),
        optimizer_kwargs=dict(lr=1e-4),
        gradient_penalty_coef=5.0,
    )
    policy = torch.randn(8, 670, requires_grad=True)
    reference = torch.randn(8, 670, requires_grad=True)
    before = {name: p.detach().clone() for name, p in style.discriminator.named_parameters()}
    metrics = style.update(policy, reference)
    assert all(torch.isfinite(value).all() for value in metrics.values())
    assert policy.grad is None and reference.grad is None
    assert any(not torch.equal(before[name], p) for name, p in style.discriminator.named_parameters())
    style.discriminator.zero_grad(set_to_none=True)
    imagined = torch.randn(8, 670, requires_grad=True)
    with frozen_parameters(style.discriminator):
        style.reward(imagined).sum().backward()
    assert torch.isfinite(imagined.grad).all() and imagined.grad.abs().sum() > 0
    assert all(p.grad is None for p in style.discriminator.parameters())


@pytest.mark.parametrize("horizon", [1, 8, 20])
def test_grouped_physical_sampling_matches_scalar_windows(horizon):
    from instinctlab.tasks.parkour.supermimic.native_training_replay import select_rows, stack_rows

    replay, _, _, _ = collect()
    # Exercise arbitrary environment order, clock discontinuity,
    # collection boundaries, truncation and NaNs in invalid padding.
    for index, row in enumerate(replay.rows):
        row["depth"] = torch.full((2, 8, 2, 3), float(index))
    replay.rows[5]["truncated"][1] = True
    replay.rows[7]["segment"] += 1
    replay.rows[9]["control_step"] += 2
    replay.rows[-1]["depth"].fill_(float("nan"))
    envs = (1, 0)
    picks = torch.randint(len(replay.rows) * len(envs), (64,), generator=torch.Generator().manual_seed(37))
    expected, masks = [], []
    fields = ("state", "next_state", "actions", "context", "contacts", "auxiliary", "depth")
    for pick in picks.tolist():
        start, env = divmod(pick, len(envs))
        env = envs[env]
        first = replay.rows[start]
        window, mask, intact = [], [], True
        for offset in range(horizon):
            index = start + offset
            row = replay.rows[min(index, len(replay.rows) - 1)]
            intact = (
                intact
                and index < len(replay.rows)
                and bool(
                    row["episode_id"][env] == first["episode_id"][env]
                    and row["segment"][env] == first["segment"][env]
                    and row["control_step"][env] == first["control_step"][env] + offset
                )
            )
            window.append({name: select_rows(row[name], env) for name in fields})
            mask.append(intact)
            intact = intact and not bool(row["terminated"][env] | row["truncated"][env])
        expected.append(stack_rows(window, "cpu"))
        masks.append(mask)
    expected = stack_rows(expected, "cpu")
    expected["mask"] = torch.tensor(masks)
    actual = replay.sample_physics(64, horizon, envs, "cpu", torch.Generator().manual_seed(37), extra_fields=("depth",))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
