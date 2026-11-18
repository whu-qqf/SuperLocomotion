"""Physical BPTT with either the legacy RSSM or a joint physical/visual model."""

import torch
from contextlib import contextmanager
from dataclasses import dataclass, field

from instinctlab_perception.rssm import RSSMState
from instinctlab_supermimic.paired_reference import paired_reference_features, paired_tracking_reward
from instinctlab_supermimic.physical import flatten_state, unflatten_state
from instinctlab_supermimic.reference import ReferencePlayer, tracking_reward
from instinctlab_supermimic.rotations import mv, quat_to_mat, yaw_mat

from .native_task import AMP_WIDTHS, POLICY_WIDTHS, flatten_history

CRITIC_HISTORY_DIM = 8 * sum(POLICY_WIDTHS) + 10 * sum(AMP_WIDTHS)
CRITIC_LAYOUT = "physical_policy_history_amp_history_belief_pending_phase_v2"


def native_value_features(physical, policy_history, amp_history, belief, pending, phase):
    """One critic layout for imagined states and measured replay states."""
    period = pending.shape[1]
    phase_onehot = torch.nn.functional.one_hot(phase, period).float()
    return torch.cat(
        (
            physical,
            flatten_history(policy_history, POLICY_WIDTHS),
            flatten_history(amp_history, AMP_WIDTHS),
            belief,
            pending.flatten(1),
            phase_onehot,
        ),
        -1,
    )


@contextmanager
def frozen_parameters(*modules):
    """Keep state/action derivatives while preserving each parameter's flags."""
    parameters = [(p, p.requires_grad) for module in modules for p in module.parameters()]
    try:
        for parameter, _ in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, original in parameters:
            parameter.requires_grad_(original)


@dataclass
class NativeImaginationStart:
    state: torch.Tensor
    previous_action: torch.Tensor
    timers: torch.Tensor
    physical_context: torch.Tensor
    task_context: dict[str, torch.Tensor]
    command: dict[str, torch.Tensor]
    sensor_context: list[dict[str, torch.Tensor]]
    policy_history: torch.Tensor
    amp_history: torch.Tensor
    perception_state: RSSMState | None
    control_step: torch.Tensor
    visual_latent: torch.Tensor | None = None
    reference_state: dict[str, torch.Tensor] = field(default_factory=dict)
    depth_history: torch.Tensor | None = None
    source_env: torch.Tensor | None = None


def imagine_native(
    task,
    actor,
    perception,
    start: NativeImaginationStart,
    horizon=32,
    add_noise=True,
    reference_bank=None,
    paired_library=None,
    touchdown_cut_threshold=0.0,
    imagined_camera=None,
    action_noise_std=0.0,
    record_visual_inputs=False,
):
    """Roll out F at 50 Hz, jointly with z on the single-world-model route.

    On the legacy dual-model route, advance P after five submitted actions.

    The caller reconstructs the initial posterior from the complete real P
    prefix using current weights, then selects tick-aligned starts. Target
    resampling is excluded from this horizon. No imagined images or privileged
    physical features enter P. The critic sees belief, interval phase and
    pending actions, including a partial final interval.
    """
    visual = start.visual_latent is not None
    period, batch = (1 if visual else perception.cfg.action_steps), len(start.state)
    if horizon < 1 or (start.control_step.remainder(period) != 0).any():
        raise ValueError("Native imagination requires a positive horizon and tick-aligned start states.")
    if (start.command["time_left"] <= horizon * task.dt).any():
        raise ValueError("Native imagination cannot cross command resampling.")
    if start.policy_history.shape != (batch, 8, 96) or start.amp_history.shape != (batch, 10, 67):
        raise ValueError("Native imagination requires exact policy/AMP frame histories.")
    state = unflatten_state(start.state)
    previous, timers, context = start.previous_action, start.timers, start.physical_context
    policy_history, amp_history = start.policy_history, start.amp_history
    memory = None if visual else start.perception_state.detach()
    belief = start.visual_latent.detach() if visual else perception.feature(memory).detach()
    depth_sequence = None
    if imagined_camera is not None:
        from .native_imagined_camera import ImaginedDepthSequence

        if not visual or task.physics.model.predict_latent:
            raise ValueError("Imagined camera requires a physical-only visual world model.")
        depth_sequence = ImaginedDepthSequence(imagined_camera, perception, start.depth_history, start.source_env)
        if not getattr(actor, "detach_latent", True):
            belief = perception.encode(start.depth_history[:, ::5])
    pending = start.state.new_zeros(batch, period, 29)
    phase = 0
    records = {
        name: []
        for name in (
            "states",
            "actions",
            "timers",
            "task_rewards",
            "terminations",
            "amp_windows",
            "contacts",
            "auxiliary",
            "physical_features",
            "value_features",
            "pending",
            "phases",
            "beliefs",
            "terrain_supported",
        )
    }
    reward_terms = {}
    if record_visual_inputs:
        if depth_sequence is None or action_noise_std <= 0:
            raise ValueError("TGI policy records require depth history and positive action noise.")
        records.update({name: [] for name in ("action_log_probs", "policy_observations", "depth_inputs")})
    touchdown_cuts = []
    previous_foot_contact = (timers[:, 2:] > 0).float()
    player = None
    paired_targets = None
    reference_features = None
    if paired_library is not None:
        if not visual or reference_bank is not None or set(start.reference_state) != {"clip_id", "frame", "terrain_id"}:
            raise ValueError("Paired imagination needs visual state and explicit terrain/clip/frame.")
        frames = start.reference_state["frame"][:, None] + torch.arange(horizon + 2, device=start.state.device)
        paired_targets = paired_library.query(
            start.reference_state["clip_id"], frames, start.command["origin"], start.reference_state["terrain_id"]
        )
        target = {name: value[:, 1] for name, value in paired_targets.items()}
        reference_features = paired_reference_features(state[0], state[1], target, task.dt)
    elif reference_bank is not None:
        if not visual or not start.reference_state:
            raise ValueError("Reference tracking requires a visual start with its real playback cursor.")
        player = ReferencePlayer(reference_bank, batch, start.reference_state)
    elif start.reference_state:
        raise ValueError("Replay contains reference state but no reference library was supplied.")

    def record_value_state():
        physical = task.critic_features(state, value_command, timers, context)
        phase_tensor = torch.full((batch,), phase, dtype=torch.long, device=start.state.device)
        records["states"].append(flatten_state(state))
        records["timers"].append(timers)
        records["physical_features"].append(physical)
        records["value_features"].append(
            torch.cat(
                (
                    physical,
                    flatten_history(policy_history, POLICY_WIDTHS),
                    flatten_history(amp_history, AMP_WIDTHS),
                    belief,
                    *((reference_features,) if reference_features is not None else ()),
                ),
                -1,
            )
            if visual
            else native_value_features(physical, policy_history, amp_history, belief, pending, phase_tensor)
        )
        records["pending"].append(pending.clone())
        records["phases"].append(phase_tensor)
        records["beliefs"].append(belief)

    velocity = start.command["velocity"]  # Native reward uses the pre-step latch.
    value_command = dict(start.command)
    supported = torch.ones(batch, dtype=torch.bool, device=start.state.device)
    if player is not None:
        reference_features, target, reference_valid = player.prepare(velocity.detach(), state[4].detach())
    record_value_state()
    for step in range(horizon):
        policy_inputs = (flatten_history(policy_history, POLICY_WIDTHS), belief)
        mean = (
            actor(*policy_inputs, reference_features).float()
            if reference_features is not None
            else actor(*policy_inputs).float()
        )
        sample = mean + action_noise_std * torch.randn_like(mean) if action_noise_std else mean
        action = sample.clamp(-20, 20)
        if record_visual_inputs:
            records["action_log_probs"].append(-((sample.detach() - mean).square()).sum(-1) / (2 * action_noise_std**2))
            records["policy_observations"].append(policy_inputs[0].detach())
            records["depth_inputs"].append(depth_sequence.images[:, ::5].detach().clone())
        previous_position = state[0]
        predicted, logits, next_belief, auxiliary = task.physics.model.step_with_aux(
            state,
            action,
            context,
            task.physics.model_scan(state),
            belief if visual and task.physics.model.z_dim else None,
        )
        _, next_supported = task.physics.scan_with_validity(predicted)
        supported = supported & next_supported
        # Absorb unsupported predictions at the last supported state. They
        # receive no task/style reward or bootstrap, and never query terrain
        # padding as if it were physical ground. Valid batch rows keep BPTT.
        state = tuple(
            torch.where(supported.reshape(-1, *([1] * (new.ndim - 1))), new, old) for old, new in zip(state, predicted)
        )
        contacts = logits.sigmoid()
        timers = task.advance_feet_timers(timers, contacts[..., -2:])
        termination = task.soft_termination(state, start.command, contacts)
        termination = torch.where(supported, termination, torch.ones_like(termination))
        if paired_targets is not None:
            frames = start.reference_state["frame"] + step + 1
            ended = frames >= paired_library.lengths[start.reference_state["clip_id"]] - 1
            termination = torch.where(ended, torch.ones_like(termination), termination)
        reward, terms = task.soft_reward(
            state,
            action,
            previous,
            velocity,
            start.task_context,
            auxiliary,
            contacts,
            timers,
            termination,
            start.sensor_context,
            previous_position=previous_position,
        )
        if getattr(task.env.cfg, "terrain_locomotion", False):
            from instinctlab_supermimic.locomotion import touchdown_events

            extra_terms, unsupported = task.locomotion_readout(state, start.task_context, contacts[..., -2:], velocity)
            reward = reward + sum(extra_terms.values())
            terms.update(extra_terms)
            touchdown_cuts.append(
                touchdown_events(previous_foot_contact, contacts[..., -2:], unsupported, touchdown_cut_threshold)
            )
            previous_foot_contact = contacts[..., -2:]
        if player is not None:
            data = task.data(state, start.task_context)
            feet = data.body_link_pos_w.torch[:, task.foot_ids]
            local_feet = mv(yaw_mat(state[1]).transpose(-1, -2)[:, None], feet - state[0][:, None])
            reference_reward, reference_terms = tracking_reward(
                state[4], state[5], local_feet, contacts[:, -2:], target, reference_valid, task.dt
            )
            reward = reward + reference_reward
            terms.update(reference_terms)
            terms["reference_active"] = reference_valid.float()
        elif paired_targets is not None:
            target = {name: value[:, step + 1] for name, value in paired_targets.items()}
            data = task.data(state, start.task_context)
            reference_reward, reference_terms = paired_tracking_reward(
                state, data.body_link_pos_w.torch[:, task.foot_ids], contacts[:, -2:], target, task.dt
            )
            reward = torch.where(target["valid"], reward + reference_reward, torch.zeros_like(reward))
            terms.update(reference_terms)
        reward = torch.where(supported, reward, torch.zeros_like(reward))
        for name, value in terms.items():
            value = torch.where(supported, value, torch.zeros_like(value))
            if paired_targets is not None:
                value = torch.where(target["valid"], value, torch.zeros_like(value))
            reward_terms.setdefault(name, []).append(value)
        if paired_targets is not None:
            # The command at state t+1 belongs to the same recorded frame.
            inverse = quat_to_mat(target["root_quat"]).transpose(-1, -2)
            linear = mv(inverse, target["root_velocity"])
            angular = mv(inverse, target["root_angular_velocity"])
            velocity = torch.cat((linear[:, :2], angular[:, 2:3]), -1)
            value_command.update(
                velocity=velocity, random_velocity=velocity, maximum=velocity.abs(), target=target["root_pos"]
            )
            following = {name: value[:, step + 2] for name, value in paired_targets.items()}
            reference_features = paired_reference_features(state[0], state[1], following, task.dt)
        else:
            velocity = task.velocity_command(state, start.command)
        if player is not None:
            player.advance()
            reference_features, target, reference_valid = player.prepare(velocity.detach(), state[4].detach())
        frame = task.proprio(state, action, velocity, start.task_context)
        if add_noise:
            frame = frame + (2 * torch.rand_like(frame) - 1) * task.policy_noise
        policy_history = torch.cat((policy_history[:, 1:], frame[:, None]), 1)
        amp_history = torch.cat((amp_history[:, 1:], task.amp_frame(state, start.task_context)[:, None]), 1)
        if visual:
            if depth_sequence is not None and supported.any():
                # Camera/FK is detached; proprio feedback through F stays live.
                next_belief = depth_sequence(tuple(value[supported] for value in state), supported, step + 1)
                belief = belief.index_copy(0, supported.nonzero(as_tuple=True)[0], next_belief)
            elif depth_sequence is None:
                belief = torch.where(supported[:, None], next_belief, belief)
        else:
            with torch.no_grad():
                pending[:, phase] = action.detach()
                phase = (phase + 1) % period
                if phase == 0:
                    memory = perception.imagine(memory, pending, sample=False)
                    belief = perception.feature(memory).detach()
                    pending = torch.zeros_like(pending)
        context = torch.cat((action, context[..., 29:]), -1)
        previous = action
        for name, value in (
            ("actions", action),
            ("task_rewards", reward),
            ("terminations", termination),
            ("amp_windows", flatten_history(amp_history, AMP_WIDTHS)),
            ("contacts", contacts),
            ("auxiliary", auxiliary),
            ("terrain_supported", supported),
        ):
            records[name].append(value)
        record_value_state()
    result = {name: torch.stack(values, 1) for name, values in records.items()}
    if touchdown_cuts:
        result["touchdown_events"] = torch.stack(touchdown_cuts, 1)
    result["reward_terms"] = {name: torch.stack(values, 1) for name, values in reward_terms.items()}
    result["perception_prior_ticks"] = 0 if visual else horizon // period
    if paired_targets is not None:
        result["reference_valid"] = paired_targets["valid"][:, 1 : horizon + 1]
    return result
