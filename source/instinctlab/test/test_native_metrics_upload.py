"""Check metric serialization and asynchronous JSONL producer boundaries."""

import json
import sys
from types import SimpleNamespace

import pytest
from instinctlab_supermimic import metrics_upload


def test_nested_metrics_preserve_steps_and_skip_non_scalars():
    assert metrics_upload.scalar_metrics(
        dict(iteration=701, model_only=False, policy=dict(loss=2.5), evaluation=None, scope="description")
    ) == {"iteration": 701, "model_only": 0, "policy/loss": 2.5}
    with pytest.raises(ValueError, match="Nonfinite"):
        metrics_upload.scalar_metrics(dict(loss=float("nan")))


def test_compact_world_model_log_keeps_key_metrics_and_preserves_all_curves():
    record = dict(
        iteration=5,
        model_only=True,
        policy=None,
        model=dict(physical=0.2, contact=0.1, latent=0.3, reconstruction_l1=0.05),
        collection=dict(episode=dict(mean_episode_length=40, metrics=dict(minor_reward=1))),
    )
    output = metrics_upload.format_iteration(record, 500, compact=True)
    assert output.startswith("Learning iteration 5/500 [world_model]")
    assert "model/physical=0.2" in output and "model/contact=0.1" in output
    assert "minor_reward" not in output and "\n" not in output
    assert metrics_upload.scalar_metrics(record)["collection/episode/metrics/minor_reward"] == 1


def test_drain_records_after_producer_exit(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"iteration": 701}\n{"iteration": 702}\n')
    assert list(metrics_upload.follow_metrics(path, lambda: False)) == [dict(iteration=701), dict(iteration=702)]


def test_partial_write_is_retried_without_losing_record(tmp_path, monkeypatch):
    path = tmp_path / "metrics.jsonl"
    path.write_text('{"iteration": ')
    running = True

    def complete_write(_):
        nonlocal running
        with path.open("a") as stream:
            stream.write("701}\n")
        running = False

    monkeypatch.setattr(metrics_upload.time, "sleep", complete_write)
    assert list(metrics_upload.follow_metrics(path, lambda: running)) == [dict(iteration=701)]


def test_missing_or_incomplete_final_log_fails(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with pytest.raises(RuntimeError, match="before creating"):
        list(metrics_upload.follow_metrics(path, lambda: False))
    path.write_text('{"iteration": ')
    with pytest.raises(ValueError, match="incomplete"):
        list(metrics_upload.follow_metrics(path, lambda: False))


@pytest.mark.parametrize("new_sdk", [False, True])
def test_resume_backfills_full_text_without_relogging_existing_curves(tmp_path, monkeypatch, capsys, new_sdk):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(start_iteration=700, target_iteration=703, additional_iterations=3)))
    records = [
        dict(iteration=i, policy=dict(actor_loss=-i / 100), evaluation=dict(holdout=dict(error=0.1)))
        for i in range(701, 704)
    ]
    (tmp_path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (tmp_path / "swanlab.json").write_text(json.dumps(dict(run_id="same-run", last_logged_iteration=702)))
    (tmp_path / "results.json").write_text(json.dumps(dict(iterations=3)))
    init_args, uploaded = [], []
    run = SimpleNamespace(id="same-run")
    if new_sdk:
        run.url = "https://example.test/same-run"
    sdk = SimpleNamespace(
        init=lambda **kwargs: init_args.append(kwargs) or run,
        log=lambda data, step: uploaded.append((step, data)),
        get_url=lambda: "https://example.test/same-run",
        finish=lambda: None,
    )
    if new_sdk:
        del sdk.get_url
    monkeypatch.setitem(sys.modules, "swanlab", sdk)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "upload",
            "--output",
            str(tmp_path),
            "--pid",
            "1",
            "--project",
            "test",
            "--experiment",
            "test",
            "--config",
            str(config),
            "--resume",
        ],
    )
    identities = iter(["original", None, None, None, None])
    monkeypatch.setattr(metrics_upload, "process_identity", lambda _: next(identities))
    metrics_upload.main()
    assert init_args[0]["id"] == "same-run" and init_args[0]["resume"] == "must"
    assert [step for step, _ in uploaded] == [703]
    output = capsys.readouterr().out
    for record in records:
        assert f"Learning iteration {record['iteration']}/703" in output
    assert "Actor objective loss (BPTT): -7.0100" in output
    assert output.count("evaluation/holdout/error: 0.1000") == 3
    status = json.loads((tmp_path / "swanlab.json").read_text())
    assert status["last_logged_iteration"] == status["last_text_iteration"] == 703
    assert status["text_records"] == 3


def test_iteration_block_uses_real_episode_values_and_measured_timings():
    record = dict(
        iteration=1301,
        collection=dict(
            real_controls=3315,
            episode=dict(
                mean_task_reward=2.5,
                mean_episode_length=42.0,
                metrics={"Episode_Termination/base_contact": 0.25, "Metrics/base_velocity/error_vel_xy": 0.4},
            ),
        ),
        policy=dict(actor_loss=-2.0, critic_loss=0.1, task_reward=0.05),
        performance=dict(
            transitions_per_second=128.0,
            collection_seconds=0.5,
            learning_seconds=1.5,
            iteration_seconds=2.0,
            session_total_seconds=20.0,
            eta_seconds=100.0,
            total_timesteps=424320,
            collection_action_noise_std=0.2,
        ),
    )
    output = metrics_upload.format_iteration(record, 14500)
    for expected in [
        "Learning iteration 1301/14500",
        "Mean task episode reward (last 100): 2.5000",
        "Imagined task reward: 0.0500",
        "Episode_Termination/base_contact: 0.2500",
        "128 steps/s (collection: 0.500s, learning: 1.500s)",
        "Total timesteps: 424320",
        "ETA: 100.00s",
    ]:
        assert expected in output
    assert "Surrogate loss" not in output and "strict_success_rate" not in output
