"""Real episode logging only; never part of a training objective or replay."""

import torch
from collections import deque


class NativeEpisodeStatistics:
    def __init__(self, num_envs, device):
        self.returns = torch.zeros(num_envs, device=device)
        self.lengths = torch.zeros(num_envs, device=device)
        self.completed_returns = deque(maxlen=100)
        self.completed_lengths = deque(maxlen=100)
        self.metric_sums = {}
        self.metric_counts = {}
        self.completed = 0

    def observe(self, rewards, done, info, termination_terms):
        self.returns += rewards.detach().sum(-1) if rewards.ndim == 2 else rewards.detach()
        self.lengths += 1
        ids = done.bool().nonzero(as_tuple=True)[0]
        count = len(ids)
        if not count:
            return
        self.completed += count
        self.completed_returns.extend(self.returns[ids].cpu().tolist())
        self.completed_lengths.extend(self.lengths[ids].cpu().tolist())
        self.returns[ids] = 0
        self.lengths[ids] = 0
        # extras['log'] persists between resets. Consume it only on an actual
        # reset and weight each native reset-batch mean by its episode count.
        for name, value in info.get("log", {}).items():
            if name.startswith("Episode_Termination/"):
                continue
            value = float(torch.as_tensor(value).float().mean())
            self.metric_sums[name] = self.metric_sums.get(name, 0.0) + value * count
            self.metric_counts[name] = self.metric_counts.get(name, 0) + count
        # Native reset logs average last-done flags over all environments.
        # Count the actual completed episodes instead; reasons may overlap.
        for name, values in termination_terms.items():
            key = "Episode_Termination/" + name
            self.metric_sums[key] = self.metric_sums.get(key, 0.0) + float(values[ids].sum())
            self.metric_counts[key] = self.metric_counts.get(key, 0) + count

    def summary(self):
        result = dict(
            completed_episodes=self.completed,
            mean_task_reward=(
                (sum(self.completed_returns) / len(self.completed_returns)) if self.completed_returns else None
            ),
            mean_episode_length=(
                (sum(self.completed_lengths) / len(self.completed_lengths)) if self.completed_lengths else None
            ),
            metrics={key: value / self.metric_counts[key] for key, value in self.metric_sums.items()},
        )
        self.metric_sums.clear()
        self.metric_counts.clear()
        self.completed = 0
        return result
