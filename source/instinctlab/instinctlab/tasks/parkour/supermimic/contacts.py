"""Contact events and differentiable physics-rate air/support clocks.

Hard events follow the upstream recurrence at every native physics substep.
Oracle-event validation checks this transport against the simulator clocks.
"""

import math
import torch


def validate_contact_clock_threshold(clock_threshold, reward_threshold):
    """Require initialized sensors to use the same native clock definition."""
    if clock_threshold is None or reward_threshold is None:
        raise ValueError("contact clock thresholds must be resolved by the physics backend")
    if not all(math.isfinite(value) and value >= 0 for value in (clock_threshold, reward_threshold)):
        raise ValueError("contact clock thresholds must be finite and nonnegative")
    if clock_threshold != reward_threshold:
        raise ValueError(
            f"contact_clock threshold {clock_threshold} differs from native reward threshold {reward_threshold}"
        )
    return float(clock_threshold)


def foot_air_time_reward(state, command, velocity_threshold):
    """Soft extension of the native single-support, minimum-mode-time reward."""
    air, stance, current = state[..., :2], state[..., 2:4], state[..., 4:6]
    single = current[..., 0] * (1 - current[..., 1]) + current[..., 1] * (1 - current[..., 0])
    mode = current * stance + (1 - current) * air
    moving = (command[..., :2].norm(dim=-1) > velocity_threshold) | (command[..., 2].abs() > velocity_threshold)
    return single * mode.amin(-1) * moving


def advance_feet_timers(timers, contact, dt):
    air, stance = timers[..., :2], timers[..., 2:4]
    return torch.cat(((air + dt) * (1 - contact), (stance + dt) * contact), -1)


def advance_contact_history(timers, events, dt, substeps):
    """Replay newest-first binary/probabilistic events in chronological order."""
    for step in reversed(range(substeps)):
        timers = advance_feet_timers(timers, events[..., 2 * step : 2 * step + 2], dt / substeps)
    return timers


def foot_contact_rewards(state, command, foot_velocity, foot_quaternion, velocity_threshold):
    """Soft extensions of native single-support, slide and contact-flat rewards."""
    from instinctlab.tasks.shadowing.supermimic.world_model_frames import _rotate_inverse

    # Native slide/flat use any force > 1N in the three-substep sensor history.
    contact = state[..., -9:].reshape(*state.shape[:-1], 3, 3)[..., :2].amax(-2)
    gravity = torch.zeros_like(foot_velocity)
    gravity[..., 2] = -1
    projected = _rotate_inverse(foot_quaternion, gravity)
    return {
        "feet_air_time": foot_air_time_reward(state, command, velocity_threshold),
        "feet_slide": (foot_velocity[..., :2].norm(dim=-1) * contact).sum(-1),
        "feet_flat_ori": (projected[..., :2].norm(dim=-1) * contact).sum(-1),
    }
