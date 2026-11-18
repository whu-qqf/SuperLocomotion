"""Sequential Hiking dynamics fitting with terrain and delayed-action context."""

from __future__ import annotations

import torch

from .terrain import TerrainHeightField


def window_indices(done: torch.Tensor, horizon: int) -> torch.Tensor:
    """Return (start time, env) windows; allow termination only on the last step.

    The terminal next-state target must be the observation before auto-reset.
    No window may use an action or target from a subsequent episode.
    """
    if done.ndim != 2 or not 1 <= horizon <= len(done):
        raise ValueError("done must be (time, env) and horizon within recorded time")
    windows = done.bool().unfold(0, horizon, 1)
    return (~windows[..., :-1].any(dim=-1)).nonzero()


class HikingDynamicsContext:
    """Re-query terrain and advance history on every predicted transition.

    Static fields are realized randomized physics and actuator lags. Actuator
    ``age`` counts buffered physics substeps since reset, capped at queue capacity.
    Raw actions have the policy joint order; state joints use the recorded layout.
    """

    def __init__(self, terrain: TerrainHeightField, parameterization, layout: dict):
        self.terrain = terrain
        self.parameterization = parameterization
        self.layout = layout
        self.slices = {}
        offset = 0
        for name, size in layout["sizes"].items():
            self.slices[name] = slice(offset, offset + size)
            offset += size
        self.size = offset

    def refresh(self, state, context):
        parts = self.parameterization._unpack(state)
        scan, valid = self.terrain.scan(parts["root_pos_w"], parts["root_quat_w"])
        return torch.cat((scan, context[..., self.slices["terrain"].stop :]), dim=-1), valid

    def advance(self, context, action):
        history = context[..., self.slices["action_history"]].reshape(
            *action.shape[:-1], self.layout["history_length"], action.shape[-1]
        )
        history = torch.cat((history[..., 1:, :], action.unsqueeze(-2)), dim=-2).flatten(-2)
        age = context[..., self.slices["actuator_age"]]
        caps = age.new_tensor(self.layout["age_caps"])
        age = torch.minimum(age + self.layout["decimation"], caps)
        return torch.cat(
            (
                context[..., : self.slices["actuator_age"].start],
                age,
                history,
            ),
            dim=-1,
        )


def rollout(model, context_builder, state, actions, context):
    """Open-loop predictions; never use future logged terrain or action history."""
    states, support = [], []
    valid_so_far = torch.ones(state.shape[0], dtype=torch.bool, device=state.device)
    for action in actions.unbind(1):
        context, valid = context_builder.refresh(state, context)
        valid_so_far = valid_so_far & valid
        state = model(state, action, context)
        states.append(state)
        support.append(valid_so_far)
        context = context_builder.advance(context, action)
    return torch.stack(states, 1), torch.stack(support, 1)


def normalized_rollout_loss(model, predictions, targets, valid):
    error = model.parameterization.error(predictions, targets) / model._error_scale()
    per_step = error.square().mean(-1)
    return (per_step * valid).sum() / valid.sum().clamp_min(1)
