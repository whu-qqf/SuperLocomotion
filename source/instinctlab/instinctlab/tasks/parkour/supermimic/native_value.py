"""Critic supervision from measured Hiking returns, independent of F rollout."""

import torch

from instinctlab_perception.replay import reconstruct_prefixes
from instinctlab_supermimic.physical import unflatten_state

from .native_imagination import native_value_features
from .native_task import AMP_WIDTHS, flatten_history
from .native_training_replay import stack_rows


@torch.no_grad()
def real_value_features(task, perception, perception_replay, records):
    """Reconstruct current-P posteriors at measured tick-aligned states."""
    device = next(perception.parameters()).device
    values = stack_rows([r["values"] for r in records], device)
    prefixes = [perception_replay.prefix(r["env"], r["episode_id"], r["control_step"]) for r in records]
    memory = reconstruct_prefixes(perception, prefixes)
    belief = perception.feature(memory)
    physical = task.critic_features(
        unflatten_state(values["state"]), values["command"], values["timers"], values["physical_context"]
    )
    batch, period = len(records), perception.cfg.action_steps
    return native_value_features(
        physical,
        values["policy_history"],
        values["amp_history"],
        belief,
        belief.new_zeros(batch, period, 29),
        torch.zeros(batch, dtype=torch.long, device=device),
    )


@torch.no_grad()
def real_value_batch(task, perception, perception_replay, style, windows, gamma):
    """Measured task+AMP returns; terminal AMP frames precede autoreset.

    All labels and P reconstructions are detached. Unknown task rewards and
    timeouts have already been excluded by the replay sampler. Endpoints use
    the actual P posterior, not a fabricated image or a stale cached belief.
    """
    if not windows or not 0 < gamma <= 1:
        raise ValueError("Real value batches need measured windows and a valid discount.")
    device = next(perception.parameters()).device
    starts = [w["start"] for w in windows]
    features = real_value_features(task, perception, perception_replay, starts)
    lengths = torch.tensor([len(w["task_reward"]) for w in windows], device=device)
    amp = [w["amp_windows"].to(device).clone() for w in windows]
    terminals = [i for i, w in enumerate(windows) if w["end"] is None]
    if terminals:
        state = torch.stack([windows[i]["terminal_state"] for i in terminals]).to(device)
        context = stack_rows([windows[i]["start"]["values"]["task_context"] for i in terminals], device)
        previous = torch.stack([windows[i]["terminal_previous_amp"] for i in terminals]).to(device)
        frame = task.amp_frame(unflatten_state(state), context)
        corrected = flatten_history(torch.cat((previous[:, 1:], frame[:, None]), 1), AMP_WIDTHS)
        for row, index in enumerate(terminals):
            amp[index][-1] = corrected[row]
    style_reward = style.reward(torch.cat(amp))
    task_reward = torch.cat([w["task_reward"] for w in windows]).to(device)
    discount = torch.cat([gamma ** torch.arange(int(n), device=device) for n in lengths.tolist()])
    owners = torch.repeat_interleave(torch.arange(len(windows), device=device), lengths)
    returns = features.new_zeros(len(windows)).index_add(0, owners, discount * (task_reward + style_reward))
    alive = [i for i, w in enumerate(windows) if w["end"] is not None]
    next_features = (
        real_value_features(task, perception, perception_replay, [windows[i]["end"] for i in alive])
        if alive
        else features.new_empty((0, features.shape[-1]))
    )
    return dict(
        features=features,
        returns=returns,
        lengths=lengths,
        next_features=next_features,
        bootstrap_indices=torch.tensor(alive, device=device, dtype=torch.long),
    )


def real_value_loss(critic, batch, gamma):
    """Real n-step TD: a physical termination has no value bootstrap."""
    with torch.no_grad():
        target = batch["returns"].clone()
        alive = batch["bootstrap_indices"]
        if len(alive):
            target[alive] += gamma ** batch["lengths"][alive] * critic(batch["next_features"])
    prediction = critic(batch["features"])
    gap = prediction - target
    loss = gap.square().mean()
    return loss, dict(
        loss=loss.detach(),
        mean_signed_gap=gap.detach().mean(),
        rmse=loss.detach().sqrt(),
        mean_value=prediction.detach().mean(),
        mean_target=target.mean(),
        physical_terminal_fraction=1 - len(alive) / len(target),
        mean_return_steps=batch["lengths"].float().mean(),
    )
