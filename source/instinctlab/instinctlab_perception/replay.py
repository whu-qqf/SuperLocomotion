"""Sparse perception ticks with whole episode prefixes, independent of physics replay.

Images are stored once per observed tick. Partial episodes are valid training
samples; a reset ends the sequence, never an implicit zero-filled continuation.
"""

import torch
from collections import OrderedDict
from dataclasses import dataclass

from .events import PerceptionTick
from .rssm import PerceptionModel, RSSMState


@dataclass
class PerceptionSequence:
    prefix: list[dict]
    target: list[dict]
    env: int
    episode_id: int


class PerceptionReplay:
    def __init__(self, num_envs, action_steps=5, max_ticks_per_env=800, holdout_fraction=0.1):
        if num_envs < 1 or max_ticks_per_env < 1 or not 0 <= holdout_fraction < 1:
            raise ValueError("Invalid perception replay dimensions or holdout fraction.")
        self.num_envs, self.action_steps, self.max_ticks_per_env = num_envs, action_steps, max_ticks_per_env
        self.train_envs = tuple(range(num_envs - int(num_envs * holdout_fraction)))
        self.holdout_envs = tuple(range(len(self.train_envs), num_envs))
        self.episodes = [OrderedDict() for _ in range(num_envs)]
        self.active = [None] * num_envs

    def add(self, tick: PerceptionTick):
        for i, env in enumerate(tick.env.tolist()):
            record = {name: value[i].detach().cpu().clone() for name, value in vars(tick).items() if name != "env"}
            ep = int(record["episode_id"])
            step = int(record["control_step"])
            episodes = self.episodes[env]
            if bool(record["is_first"]):
                if step != 0 or ep in episodes:
                    raise ValueError("Episode replay must begin once at control step zero.")
                episodes[ep] = []
                self.active[env] = ep
            if self.active[env] != ep or ep not in episodes:
                raise ValueError("Missing episode prefix; refusing to invent a hidden state.")
            history = episodes[ep]
            if history and step != int(history[-1]["control_step"]) + self.action_steps:
                raise ValueError("Perception ticks must contain exactly one ordered action interval.")
            if bool(record["observed"]):
                if not (0 <= int(record["capture_step"]) <= int(record["delivery_step"]) <= step):
                    raise ValueError("Camera capture/delivery/assimilation timestamps are out of order.")
                if any(bool(r["observed"]) and int(r["frame_id"]) >= int(record["frame_id"]) for r in history):
                    raise ValueError("A camera frame cannot be assimilated twice or out of order.")
            else:
                record["depth"] = None
            history.append(record)
            if len(history) > self.max_ticks_per_env:
                raise RuntimeError(
                    "Active episode exceeds perception replay capacity; increase capacity, do not truncate its prefix."
                )
            while sum(map(len, episodes.values())) > self.max_ticks_per_env and next(iter(episodes)) != ep:
                episodes.popitem(last=False)

    def finish(self, env_ids, episode_ids):
        for env, ep in zip(env_ids.tolist(), episode_ids.tolist()):
            if self.active[env] != ep:
                raise ValueError("Finishing the wrong perception episode.")
            self.active[env] = None

    def prefix(self, env, episode_id, control_step):
        if episode_id not in self.episodes[env]:
            raise KeyError("The complete perception prefix is no longer retained.")
        history = self.episodes[env][episode_id]
        index = next((i for i, r in enumerate(history) if int(r["control_step"]) == control_step), None)
        if index is None:
            raise KeyError("Policy starts must coincide with a retained perception tick.")
        return history[: index + 1]

    def sample(self, batch_size, sequence_length=64, generator=None, holdout=False):
        pool = self.holdout_envs if holdout else self.train_envs
        episodes = [(env, ep, len(history)) for env in pool for ep, history in self.episodes[env].items() if history]
        if not episodes:
            raise RuntimeError("No perception ticks in the requested environment split.")
        # Uniform over ticks without materializing up to N * capacity Python
        # tuples on every update. Episode prefixes themselves stay intact.
        cumulative = torch.tensor([length for _, _, length in episodes]).cumsum(0)
        picks = torch.randint(int(cumulative[-1]), (batch_size,), generator=generator)
        episode_indices = torch.searchsorted(cumulative, picks, right=True)
        offsets = picks - torch.nn.functional.pad(cumulative, (1, 0))[episode_indices]
        samples = []
        for index, start in zip(episode_indices.tolist(), offsets.tolist()):
            env, ep, _ = episodes[index]
            history = self.episodes[env][ep]
            samples.append(PerceptionSequence(history[:start], history[start : start + sequence_length], env, ep))
        return samples

    def state_dict(self):
        return dict(
            config=(self.num_envs, self.action_steps, self.max_ticks_per_env, self.train_envs, self.holdout_envs),
            episodes=self.episodes,
            active=self.active.copy(),
        )

    def load_state_dict(self, state):
        if state["config"] != self.state_dict()["config"]:
            raise ValueError("Incompatible perception replay checkpoint.")
        self.episodes, self.active = state["episodes"], state["active"].copy()


def collate_ticks(sequences, cfg, device):
    """Pad only the target batch; mask explicitly distinguishes padding from ticks."""
    b = len(sequences)
    t = max(map(len, sequences), default=0)
    if t == 0:
        raise ValueError("Cannot collate an empty perception sequence batch.")
    shapes = dict(
        actions=(cfg.action_steps, cfg.action_dim),
        depth=(1, cfg.image_height, cfg.image_width),
        proprio=(cfg.proprio_dim,),
        frame_age=(),
        observed=(),
        is_first=(),
        mask=(),
    )
    result = {
        name: torch.zeros(
            b,
            t,
            *shape,
            device=device,
            dtype=(
                torch.uint8
                if name == "depth"
                else torch.bool if name in ("observed", "is_first", "mask") else torch.float32
            )
        )
        for name, shape in shapes.items()
    }
    for i, sequence in enumerate(sequences):
        for j, record in enumerate(sequence):
            result["mask"][i, j] = True
            for name in shapes:
                if name != "mask" and record[name] is not None:
                    result[name][i, j] = record[name].to(device)
    return result


@torch.no_grad()
def reconstruct_prefixes(model: PerceptionModel, prefixes):
    """Recompute with current model parameters; no cached/old-parameter states."""
    device = next(model.parameters()).device
    state = model.initial(len(prefixes), sample=False)
    if not any(prefixes):
        return state.detach()
    batch = collate_ticks(prefixes, model.cfg, device)
    for j in range(batch["mask"].shape[1]):
        posterior, _ = model.observe(
            state,
            batch["actions"][:, j],
            batch["depth"][:, j],
            batch["proprio"][:, j],
            batch["frame_age"][:, j],
            batch["observed"][:, j],
            batch["is_first"][:, j],
            sample=False,
        )
        state = RSSMState.choose(batch["mask"][:, j], posterior, state)
    return state.detach()
