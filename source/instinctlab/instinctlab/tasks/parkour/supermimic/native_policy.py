"""Native SuperMimic BPTT actor step and detached lambda-return critic fit.

Follows the discounted-survival objective of the MIT SuperMimic reference
(Chong Zhang / ETH Zurich). P and F have separate real-data optimizers.
"""

import torch
from dataclasses import dataclass

from instinctlab_supermimic.locomotion import event_gated_return

from .native_imagination import frozen_parameters, imagine_native


@dataclass(frozen=True)
class NativePolicyUpdateCfg:
    horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_failure_head: bool = False
    failure_clip: float = 0.95
    touchdown_cut_threshold: float = 0.0


def update_native_policy(
    task,
    actor,
    perception,
    critic,
    style,
    start,
    actor_optimizer,
    critic_optimizer,
    cfg=NativePolicyUpdateCfg(),
    failure_head=None,
    reference_bank=None,
    paired_library=None,
    imagined_camera=None,
    depth_optimizer=None,
):
    """One actor update, followed by the configured number of critic steps.

    This accepts a current-P reconstructed real seed. Collection/replay and
    learning P, F, style and failure are owned by the native runner. Neither a
    PPO loss nor a baseline/teacher controller participates in this update.
    """
    batch = len(start.state)
    if cfg.critic_epochs < 0 or not 1 <= cfg.critic_minibatches <= batch * cfg.horizon:
        raise ValueError("Native critic batches must be nonempty and epochs nonnegative.")
    if not (0 < cfg.gamma <= 1 and 0 <= cfg.lam <= 1 and 0 <= cfg.failure_clip < 1):
        raise ValueError("Invalid native return or failure probability settings.")
    if not 0 <= cfg.touchdown_cut_threshold < 1:
        raise ValueError("Touchdown cut threshold must be in [0, 1).")
    if cfg.touchdown_cut_threshold > 0 and not getattr(task.env.cfg, "terrain_locomotion", False):
        raise ValueError("Touchdown cuts require terrain locomotion geometry.")
    if cfg.use_failure_head and failure_head is None:
        raise ValueError("Native failure prediction was enabled without a trained head.")
    frozen = [task.physics.model, critic]
    if depth_optimizer is None:
        frozen.append(perception)
    if style is not None:
        frozen.append(style.discriminator)
    if failure_head is not None:
        frozen.append(failure_head)
    with frozen_parameters(*frozen):
        imagined = imagine_native(
            task,
            actor,
            perception,
            start,
            cfg.horizon,
            reference_bank=reference_bank,
            paired_library=paired_library,
            touchdown_cut_threshold=cfg.touchdown_cut_threshold,
            imagined_camera=imagined_camera,
        )
        style_reward = (
            torch.zeros_like(imagined["task_rewards"])
            if style is None
            else style.reward(imagined["amp_windows"].flatten(0, 1)).view(batch, cfg.horizon)
        )
        style_reward = torch.where(imagined["terrain_supported"], style_reward, torch.zeros_like(style_reward))
        reward = imagined["task_rewards"] + style_reward
        terminated = imagined["terminations"]
        physical_hazard = terminated.detach()
        learned_hazard = torch.zeros_like(physical_hazard)
        if cfg.use_failure_head:
            prediction = failure_head(imagined["physical_features"][:, 1:]).sigmoid().clamp(max=cfg.failure_clip)
            learned_hazard = prediction.detach()
            terminated = 1 - (1 - terminated) * (1 - prediction)
        survival = (1 - terminated).cumprod(1)
        before = torch.cat((torch.ones_like(survival[:, :1]), survival[:, :-1]), 1)
        discount = cfg.gamma ** torch.arange(cfg.horizon, device=reward.device, dtype=reward.dtype)
        returns = (discount * before * reward).sum(1)
        returns = returns + cfg.gamma**cfg.horizon * survival[:, -1] * critic(imagined["value_features"][:, -1])
        event_fraction = 0.0
        if cfg.touchdown_cut_threshold > 0:
            events = imagined["touchdown_events"]
            returns = event_gated_return(
                reward, terminated, critic(imagined["value_features"][:, 1:]), events, cfg.gamma
            )
            event_fraction = float(events.float().mean())
        loss = -returns.mean()
        actor_optimizer.zero_grad(set_to_none=True)
        if depth_optimizer is not None:
            depth_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        torch.nn.utils.clip_grad_value_(actor.parameters(), 0.1)
        if not torch.isfinite(actor_grad):
            raise FloatingPointError("Nonfinite native actor gradient; optimizer not stepped.")
        depth_grad = loss.new_zeros(())
        if depth_optimizer is not None:
            depth_grad = torch.nn.utils.clip_grad_norm_(perception.parameters(), 1.0)
            if not torch.isfinite(depth_grad):
                raise FloatingPointError("Nonfinite task encoder gradient; optimizers not stepped.")
        actor_optimizer.step()
        if depth_optimizer is not None:
            depth_optimizer.step()
    # No physical or policy graph participates in critic fitting. Changing
    # normalizer statistics happens after the actor graph has been consumed.
    critic_steps = 0
    critic_loss = loss.detach().new_zeros(())
    if cfg.critic_epochs:
        features = imagined["value_features"].detach()
        with torch.no_grad():
            critic.norm.update(features)
            value = critic(features)
            following = value[:, -1]
            targets = []
            for index in range(cfg.horizon - 1, -1, -1):
                following = reward[:, index].detach() + cfg.gamma * (1 - terminated[:, index].detach()) * (
                    (1 - cfg.lam) * value[:, index + 1] + cfg.lam * following
                )
                targets.append(following)
            targets = torch.stack(targets[::-1], 1).flatten()
        inputs = features[:, :-1].flatten(0, 1)
        if "reference_valid" in imagined:
            valid = imagined["reference_valid"].flatten()
            inputs, targets = inputs[valid], targets[valid]
            if len(inputs) < cfg.critic_minibatches:
                raise ValueError("Paired critic has fewer valid transitions than minibatches.")
        for _ in range(cfg.critic_epochs):
            permutation = torch.randperm(len(inputs), device=inputs.device)
            for indices in permutation.tensor_split(cfg.critic_minibatches):
                critic_loss = (critic(inputs[indices]) - targets[indices]).square().mean()
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                critic_grad = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                if not torch.isfinite(critic_grad):
                    raise FloatingPointError("Nonfinite native critic gradient; optimizer not stepped.")
                critic_optimizer.step()
                critic_steps += 1
    return dict(
        actor_loss=float(loss.detach()),
        actor_grad_norm=float(actor_grad),
        depth_task_grad_norm=float(depth_grad),
        critic_loss=float(critic_loss.detach()),
        task_reward=float(imagined["task_rewards"].detach().mean()),
        style_reward=float(style_reward.detach().mean()),
        amp_enabled=style is not None,
        imagined_reward_terms={name: float(value.detach().mean()) for name, value in imagined["reward_terms"].items()},
        terrain_unsupported_fraction=float((~imagined["terrain_supported"]).float().mean()),
        final_survival=float(survival[:, -1].detach().mean()),
        physical_termination_probability=float(physical_hazard.mean()),
        learned_failure_probability=float(learned_hazard.mean()),
        effective_survival_steps=float(before.detach().sum(1).mean()),
        touchdown_cut_fraction=event_fraction,
        policy_updates=1,
        critic_updates=critic_steps,
        perception_prior_ticks=imagined["perception_prior_ticks"],
        final_perception_phase=int(imagined["phases"][0, -1]),
        scope="One native imagination optimization update; no claim of learned dynamics or gait quality.",
    )
