"""TGI policy objective with a full effective batch and bounded rollout memory."""

import torch
from dataclasses import fields, replace

from instinctlab_supermimic.locomotion import event_gated_return

from .native_imagination import frozen_parameters, imagine_native


def mirror_joints(value, names):
    counterparts = [
        name.replace("left_", "right_") if name.startswith("left_") else name.replace("right_", "left_")
        for name in names
    ]
    permutation = [names.index(name) for name in counterparts]
    signs = value.new_tensor([-1 if "_roll_" in name or "_yaw_" in name else 1 for name in names])
    return value[..., permutation] * signs


def mirror_proprio(value, names):
    parts = value.split((24, 24, 24, 232, 232, 232), -1)
    signs = ((-1, 1, -1), (1, -1, 1), (1, -1, -1))
    vectors = [part.reshape(*part.shape[:-1], 8, 3) * value.new_tensor(sign) for part, sign in zip(parts[:3], signs)]
    joints = [mirror_joints(part.reshape(*part.shape[:-1], 8, 29), names) for part in parts[3:]]
    return torch.cat([part.flatten(-2) for part in (*vectors, *joints)], -1)


def lambda_targets(reward, terminated, value, gamma, lam):
    following, targets = value[:, -1], []
    for step in range(reward.shape[1] - 1, -1, -1):
        following = reward[:, step] + gamma * (1 - terminated[:, step]) * (
            (1 - lam) * value[:, step + 1] + lam * following
        )
        targets.append(following)
    return torch.stack(targets[::-1], 1)


def slice_start(start, first, last):
    def select(value):
        if isinstance(value, torch.Tensor):
            return value[first:last]
        if isinstance(value, dict):
            return {name: select(item) for name, item in value.items()}
        if isinstance(value, list):
            return [select(item) for item in value]
        if value is None:
            return None
        raise TypeError(f"Unsupported TGI replay start: {type(value)}")

    return replace(start, **{field.name: select(getattr(start, field.name)) for field in fields(start)})


def update_tgi_depth_policy(
    task,
    actor,
    perception,
    critic,
    start,
    actor_optimizer,
    depth_optimizer,
    critic_optimizer,
    cfg,
    camera,
    microbatch=128,
):
    """Recompute identical noisy rollouts to normalize advantages over the full batch.

    The first pass stores detached critic data and RNG states. The second pass
    accumulates policy/encoder gradients; neither optimizer steps per chunk.
    Targets use the same critic normalizer as the actor objective, as in TGI.
    """
    batch, horizon = len(start.state), cfg.horizon
    if microbatch < 1 or cfg.critic_minibatches > batch * horizon:
        raise ValueError("Invalid TGI policy microbatch or critic batch count.")
    device = start.state.device
    chunks = [(first, min(first + microbatch, batch)) for first in range(0, batch, microbatch)]
    rng_states, inputs, targets, advantages = [], [], [], []
    names = list(task.physics.recorder.joint_names)

    def rng():
        return torch.cuda.get_rng_state(device) if device.type == "cuda" else torch.random.get_rng_state()

    def restore(state):
        if device.type == "cuda":
            torch.cuda.set_rng_state(state, device)
        else:
            torch.random.set_rng_state(state)

    def rollout(first, last):
        return imagine_native(
            task,
            actor,
            perception,
            slice_start(start, first, last),
            horizon=horizon,
            imagined_camera=camera,
            touchdown_cut_threshold=cfg.touchdown_cut_threshold,
            action_noise_std=0.2,
            record_visual_inputs=True,
        )

    metrics = dict(
        task_reward=0.0,
        final_survival=0.0,
        effective_survival_steps=0.0,
        touchdown_cut_fraction=0.0,
        physical_termination_probability=0.0,
        terrain_unsupported_fraction=0.0,
        symmetry_loss=0.0,
        likelihood_ratio_loss=0.0,
    )
    reward_terms = {}
    actor_optimizer.zero_grad(set_to_none=True)
    depth_optimizer.zero_grad(set_to_none=True)
    actor_loss = 0.0
    with frozen_parameters(task.physics.model, critic):
        with torch.no_grad():
            for first, last in chunks:
                rng_states.append(rng())
                imagined = rollout(first, last)
                features = imagined["value_features"]
                value = critic(features)
                target = lambda_targets(imagined["task_rewards"], imagined["terminations"], value, cfg.gamma, cfg.lam)
                inputs.append(features.detach())
                targets.append(target.detach())
                advantages.append(target - value[:, :-1])
                fraction = (last - first) / batch
                survival = (1 - imagined["terminations"]).cumprod(1)
                before = torch.cat((torch.ones_like(survival[:, :1]), survival[:, :-1]), 1)
                values = dict(
                    task_reward=imagined["task_rewards"].mean(),
                    final_survival=survival[:, -1].mean(),
                    effective_survival_steps=before.sum(1).mean(),
                    touchdown_cut_fraction=imagined["touchdown_events"].float().mean(),
                    physical_termination_probability=imagined["terminations"].mean(),
                    terrain_unsupported_fraction=(~imagined["terrain_supported"]).float().mean(),
                )
                for name, value in values.items():
                    metrics[name] += float(value) * fraction
                for name, value in imagined["reward_terms"].items():
                    reward_terms[name] = reward_terms.get(name, 0.0) + float(value.mean()) * fraction
                del imagined, features, value
            advantages = torch.cat(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
            final_rng = rng()
        for (first, last), saved_rng in zip(chunks, rng_states):
            restore(saved_rng)
            imagined = rollout(first, last)
            reward, terminated = imagined["task_rewards"], imagined["terminations"]
            values = critic(imagined["value_features"][:, 1:])
            returns = event_gated_return(reward, terminated, values, imagined["touchdown_events"], cfg.gamma)
            survival = (1 - terminated).cumprod(1)
            before = torch.cat((torch.ones_like(survival[:, :1]), survival[:, :-1]), 1).detach()
            likelihood = -(before * imagined["action_log_probs"] * advantages[first:last]).mean()
            # TGI samples at most 4096 detached imagined observations for symmetry.
            proprio = imagined["policy_observations"].flatten(0, 1)
            images = imagined["depth_inputs"].flatten(0, 1)
            count = max(1, round(min(4096, batch * horizon) * (last - first) / batch))
            selected = torch.randperm(len(proprio), device=device)[:count]
            proprio, images = proprio[selected], images[selected]
            original = actor(proprio, perception.encode(images))
            mirrored = actor(mirror_proprio(proprio, names), perception.encode(images.flip(-1)))
            symmetry = (mirrored - mirror_joints(original, names)).square().mean()
            fraction = (last - first) / batch
            loss = (-returns.mean() + 0.1 * likelihood + 2.0 * symmetry) * fraction
            loss.backward()
            actor_loss += float(loss.detach())
            metrics["symmetry_loss"] += float(symmetry.detach()) * fraction
            metrics["likelihood_ratio_loss"] += float(likelihood.detach()) * fraction
            del imagined, reward, terminated, values, returns, loss, original, mirrored
        restore(final_rng)
        actor_grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        depth_grad = torch.nn.utils.clip_grad_norm_(perception.parameters(), 1.0)
        if not torch.isfinite(actor_grad) or not torch.isfinite(depth_grad):
            raise FloatingPointError("Nonfinite TGI policy gradient; optimizers not stepped.")
        actor_optimizer.step()
        depth_optimizer.step()
    features, targets = torch.cat(inputs), torch.cat(targets).flatten()
    critic.norm.update(features)
    features = features[:, :-1].flatten(0, 1)
    critic_steps, critic_loss = 0, features.new_zeros(())
    for _ in range(cfg.critic_epochs):
        permutation = torch.randperm(len(features), device=device)
        for indices in permutation.tensor_split(cfg.critic_minibatches):
            critic_loss = (critic(features[indices]) - targets[indices]).square().mean()
            critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            if not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite TGI critic gradient; optimizer not stepped.")
            critic_optimizer.step()
            critic_steps += 1
    return dict(
        actor_loss=actor_loss,
        actor_grad_norm=float(actor_grad),
        depth_task_grad_norm=float(depth_grad),
        critic_loss=float(critic_loss.detach()),
        policy_updates=1,
        critic_updates=critic_steps,
        effective_actor_batch=batch,
        policy_microbatch=microbatch,
        amp_enabled=False,
        style_reward=0.0,
        learned_failure_probability=0.0,
        imagined_reward_terms=reward_terms,
        perception_prior_ticks=0,
        final_perception_phase=0,
        **metrics,
    )
