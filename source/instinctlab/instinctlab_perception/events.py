"""Camera delivery and per-environment five-action clock.

Capture, delivery and assimilation are distinct events. A delivered frame can
wait until the next perception tick; its age uses capture time throughout.
All clocks use control steps, never training iterations or physics substeps.
"""

import math
import torch
from dataclasses import dataclass

from .rssm import PerceptionModel, RSSMState


@dataclass
class PerceptionTick:
    env: torch.Tensor
    episode_id: torch.Tensor
    control_step: torch.Tensor
    actions: torch.Tensor
    proprio: torch.Tensor
    depth: torch.Tensor
    observed: torch.Tensor
    is_first: torch.Tensor
    capture_step: torch.Tensor
    delivery_step: torch.Tensor
    frame_id: torch.Tensor
    frame_age: torch.Tensor

    def index(self, indices):
        return PerceptionTick(**{name: value[indices] for name, value in vars(self).items()})


class CameraQueue:
    def __init__(self, num_envs, image_shape, device, period_steps=5, delay_steps=0):
        if period_steps < 1 or delay_steps < 0:
            raise ValueError("Camera period must be positive and delay nonnegative.")
        self.period_steps, self.delay_steps = period_steps, delay_steps
        self.slots = math.ceil(delay_steps / period_steps) + 2
        self.depth = torch.zeros(num_envs, self.slots, *image_shape, dtype=torch.uint8, device=device)
        self.occupied = torch.zeros(num_envs, self.slots, dtype=torch.bool, device=device)
        self.capture_step = torch.full((num_envs, self.slots), -1, dtype=torch.long, device=device)
        self.delivery_step = torch.full_like(self.capture_step, -1)
        self.episode_id = torch.full_like(self.capture_step, -1)
        self.frame_id = torch.full_like(self.capture_step, -1)
        self.next_id = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.last_capture = torch.full_like(self.next_id, -period_steps)

    def reset(self, env_ids):
        self.occupied[env_ids] = False
        self.next_id[env_ids] = 0
        self.last_capture[env_ids] = -self.period_steps

    def capture(self, depth, captured, step, episode_id):
        ids = captured.nonzero(as_tuple=True)[0]
        if len(ids) == 0:
            return
        if ((step[ids] - self.last_capture[ids]) < self.period_steps).any():
            raise ValueError("Camera refreshed faster than its declared period or twice at the same control step.")
        slots = self.next_id[ids] % self.slots
        if self.occupied[ids, slots].any():
            raise RuntimeError("Camera queue overflow; frames cannot be silently overwritten.")
        image = depth if depth.dtype == torch.uint8 else (depth.clamp(0, 1) * 255).round().to(torch.uint8)
        self.depth[ids, slots] = image[ids]
        self.capture_step[ids, slots] = step[ids]
        self.delivery_step[ids, slots] = step[ids] + self.delay_steps
        self.episode_id[ids, slots] = episode_id[ids]
        self.frame_id[ids, slots] = self.next_id[ids]
        self.occupied[ids, slots] = True
        self.next_id[ids] += 1
        self.last_capture[ids] = step[ids]

    def deliver(self, step, episode_id):
        current_episode = self.episode_id == episode_id[:, None]
        self.occupied &= current_episode
        ready = self.occupied & (self.delivery_step <= step[:, None])
        # Newest ready frame wins. Every delivered frame is removed exactly once.
        slots = self.capture_step.masked_fill(~ready, -1).argmax(1)
        ids = torch.arange(len(step), device=step.device)
        result = {
            name: getattr(self, name)[ids, slots].clone()
            for name in ("depth", "capture_step", "delivery_step", "episode_id", "frame_id")
        }
        result["available"] = ready.any(1)
        self.occupied &= ~ready
        return result

    def state_dict(self):
        return {name: value.clone() for name, value in vars(self).items() if isinstance(value, torch.Tensor)}

    def load_state_dict(self, state):
        for name, value in state.items():
            getattr(self, name).copy_(value)


class PerceptionRuntime:
    """Same deterministic discrete mode for real inference, replay and imagination.

    The model supports stochastic sampling separately, but this first integrated
    route uses mode with straight-through gradients in training. This makes
    prefix reconstruction exact for fixed weights without stale hidden states.
    """

    def __init__(self, model: PerceptionModel, num_envs: int, control_dt=0.02, delay_steps=0):
        self.model, self.control_dt = model, control_dt
        c = model.cfg
        device = next(model.parameters()).device
        self.camera = CameraQueue(num_envs, (1, c.image_height, c.image_width), device, c.action_steps, delay_steps)
        self.state = model.initial(num_envs, sample=False).detach()
        self.control_step = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.episode_id = torch.zeros_like(self.control_step)
        self.count = torch.zeros_like(self.control_step)
        self.first = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.ready_for_action = torch.zeros_like(self.first)
        self.actions = torch.zeros(num_envs, c.action_steps, c.action_dim, device=device)
        self.latest = dict(
            depth=torch.zeros(num_envs, 1, c.image_height, c.image_width, dtype=torch.uint8, device=device)
        )
        for name in ("capture_step", "delivery_step", "episode_id", "frame_id"):
            self.latest[name] = torch.full_like(self.control_step, -1)
        self.last_assimilated_id = torch.full_like(self.control_step, -1)

    @torch.no_grad()
    def reset(self, env_ids):
        self.episode_id[env_ids] += 1
        self.control_step[env_ids] = 0
        self.count[env_ids] = 0
        self.first[env_ids] = True
        self.ready_for_action[env_ids] = False
        self.actions[env_ids] = 0
        self.camera.reset(env_ids)
        initial = self.model.initial(len(env_ids), sample=False)
        for name in ("deter", "stoch", "logits"):
            getattr(self.state, name)[env_ids] = getattr(initial, name)
        for name, value in self.latest.items():
            value[env_ids] = 0 if name == "depth" else -1
        self.last_assimilated_id[env_ids] = -1

    @torch.no_grad()
    def before_action(self, depth, proprio, captured):
        if self.ready_for_action.any():
            raise RuntimeError("before_action called twice without a submitted action.")
        self.camera.capture(depth, captured, self.control_step, self.episode_id)
        delivery = self.camera.deliver(self.control_step, self.episode_id)
        available = delivery.pop("available")
        for name, value in delivery.items():
            self.latest[name][available] = value[available]
        ids = (self.first | (self.count == self.model.cfg.action_steps)).nonzero(as_tuple=True)[0]
        tick = None
        if len(ids):
            observed = (self.latest["episode_id"][ids] == self.episode_id[ids]) & (
                self.latest["frame_id"][ids] > self.last_assimilated_id[ids]
            )
            age = torch.where(
                self.latest["capture_step"][ids] >= 0,
                (self.control_step[ids] - self.latest["capture_step"][ids]).float() * self.control_dt,
                0.0,
            )
            tick = PerceptionTick(
                env=ids,
                episode_id=self.episode_id[ids].clone(),
                control_step=self.control_step[ids].clone(),
                actions=self.actions[ids].clone(),
                proprio=proprio[ids].clone(),
                depth=self.latest["depth"][ids].clone(),
                observed=observed,
                is_first=self.first[ids].clone(),
                capture_step=self.latest["capture_step"][ids].clone(),
                delivery_step=self.latest["delivery_step"][ids].clone(),
                frame_id=self.latest["frame_id"][ids].clone(),
                frame_age=age,
            )
            posterior, _ = self.model.observe(
                self.state.index(ids),
                tick.actions,
                tick.depth,
                tick.proprio,
                tick.frame_age,
                tick.observed,
                tick.is_first,
                sample=False,
            )
            for name in ("deter", "stoch", "logits"):
                getattr(self.state, name)[ids] = getattr(posterior, name)
            self.last_assimilated_id[ids] = torch.where(
                observed, self.latest["frame_id"][ids], self.last_assimilated_id[ids]
            )
            self.count[ids] = 0
            self.actions[ids] = 0
            self.first[ids] = False
        self.ready_for_action[:] = True
        return self.model.feature(self.state).detach(), tick

    @torch.no_grad()
    def after_action(self, realized_action):
        if not self.ready_for_action.all():
            raise RuntimeError("An action must follow exactly one before_action call.")
        if (self.count >= self.model.cfg.action_steps).any():
            raise RuntimeError("Five-action interval was not consumed at its perception tick.")
        ids = torch.arange(len(self.count), device=self.count.device)
        self.actions[ids, self.count] = realized_action
        self.count += 1
        self.control_step += 1
        self.ready_for_action[:] = False

    def state_dict(self):
        return dict(
            tensors={name: value.clone() for name, value in vars(self).items() if isinstance(value, torch.Tensor)},
            rssm={name: value.clone() for name, value in vars(self.state).items()},
            latest={name: value.clone() for name, value in self.latest.items()},
            camera=self.camera.state_dict(),
            config=dict(
                num_envs=len(self.count),
                control_dt=self.control_dt,
                delay_steps=self.camera.delay_steps,
                period_steps=self.camera.period_steps,
            ),
        )

    def load_state_dict(self, state):
        expected = self.state_dict()["config"]
        if state["config"] != expected:
            raise ValueError("Perception clock checkpoint is incompatible with this runtime.")
        for name, value in state["tensors"].items():
            getattr(self, name).copy_(value)
        self.state = RSSMState(**{name: value.to(self.actions.device).clone() for name, value in state["rssm"].items()})
        for name, value in state["latest"].items():
            self.latest[name].copy_(value)
        self.camera.load_state_dict(state["camera"])
