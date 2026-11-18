"""Real episode logs must exclude stale reset info and weight reset batches correctly."""

import torch

import pytest

from instinctlab.tasks.parkour.supermimic.native_statistics import NativeEpisodeStatistics


def test_episode_rewards_survive_collection_boundaries_and_ignore_stale_info():
    stats = NativeEpisodeStatistics(3, "cpu")
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    stats.observe(rewards, torch.zeros(3), {"log": {"Episode_Reward/task/sum": 999.0}}, {})
    summary = stats.summary()
    assert summary["completed_episodes"] == 0 and summary["mean_task_reward"] is None
    assert summary["metrics"] == {}
    stats.observe(
        rewards,
        torch.tensor([1, 0, 0]),
        {"log": {"Episode_Reward/task/sum": 2.0, "Episode_Termination/base_contact": 0.0}},
        {"base_contact": torch.tensor([True, False, False]), "time_out": torch.zeros(3, dtype=torch.bool)},
    )
    stats.observe(
        rewards,
        torch.tensor([0, 1, 1]),
        {"log": {"Episode_Reward/task/sum": 7.5, "Episode_Termination/base_contact": 0.0}},
        {"base_contact": torch.tensor([False, False, True]), "time_out": torch.tensor([False, True, False])},
    )
    summary = stats.summary()
    assert summary["completed_episodes"] == 3
    assert summary["mean_task_reward"] == pytest.approx(17 / 3)
    assert summary["mean_episode_length"] == pytest.approx(8 / 3)
    assert summary["metrics"]["Episode_Reward/task/sum"] == pytest.approx(17 / 3)
    assert summary["metrics"]["Episode_Termination/base_contact"] == pytest.approx(2 / 3)
    assert summary["metrics"]["Episode_Termination/time_out"] == pytest.approx(1 / 3)
    assert stats.summary()["metrics"] == {}


def test_reward_logging_does_not_change_reward_inputs():
    rewards = torch.tensor([[1.0, 2.0], [4.0, 5.0]])
    original = rewards.clone()
    stats = NativeEpisodeStatistics(2, "cpu")
    stats.observe(rewards, torch.ones(2), {}, {})
    assert torch.equal(rewards, original)
    assert stats.summary()["mean_task_reward"] == 6.0
