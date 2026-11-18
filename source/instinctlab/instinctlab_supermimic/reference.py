"""Dataset target playback shared by collection, imagination and deployment."""

import torch

REFERENCE_INPUT_DIM = 39  # q29, heading-local feet6, contacts2, remaining1, validity1


class ReferenceWindows:
    """Whole-horizon command matching with explicit unsupported-command masks.

    Each start contains H+1 frames: actor input at t, reward target at t+1.
    Matching checks every command in the window, not just its average, so a
    turning clip cannot silently supervise a straight-walking command. Joint
    continuity chooses among compatible starts; it never overrides the gate.
    """

    def __init__(self, clips, horizon=32, stride=5, tolerance=(0.3, 0.2, 0.5)):
        if horizon < 1 or stride < 1 or len(tolerance) != 3 or min(tolerance) <= 0:
            raise ValueError("Invalid reference horizon, stride or command tolerance.")
        self.horizon = horizon
        self.frames = {name: torch.cat([clip[name] for clip in clips]) for name in clips[0]}
        starts, offset = [], 0
        for clip in clips:
            starts.extend(range(offset, offset + len(clip["command"]) - horizon, stride))
            offset += len(clip["command"])
        if not starts:
            raise ValueError("No reference clip covers a complete imagination horizon.")
        device = self.frames["command"].device
        self.starts = torch.tensor(starts, device=device)
        self.tolerance = self.frames["command"].new_tensor(tolerance)
        self.offsets = torch.arange(horizon + 1, device=device)
        command = self.frames["command"][self.starts[:, None] + self.offsets]
        self.command_min = command.amin(1)
        self.command_max = command.amax(1)
        self.mean_command = command.mean(1)

    @torch.no_grad()
    def select(self, command, joint_pos):
        """Return frame index and validity. Invalid rows must not supervise.

        Match in small batches to avoid a B x windows x horizon allocation.
        The index is always safe to gather, even when validity is false.
        """
        indices, valid = [], []
        for cmd, q in zip(command.split(64), joint_pos.split(64)):
            distance = (
                torch.maximum((cmd[:, None] - self.command_min).abs(), (cmd[:, None] - self.command_max).abs())
                / self.tolerance
            )
            allowed = distance.amax(-1) <= 1
            continuity = (q[:, None] - self.frames["joint_pos"][self.starts]).square().mean(-1)
            speed_error = ((cmd[:, None] - self.mean_command) / self.tolerance).square().mean(-1)
            cost = (continuity + 0.1 * speed_error).masked_fill(~allowed, float("inf"))
            indices.append(self.starts[cost.argmin(-1)])
            valid.append(allowed.any(-1))
        return torch.cat(indices), torch.cat(valid)

    def targets(self, starts):
        """Gather a complete window without looping or resampling clip ends."""
        indices = starts[:, None] + self.offsets
        return {name: value[indices] for name, value in self.frames.items()}

    def state_dict(self):
        return dict(
            frames=self.frames,
            starts=self.starts,
            horizon=self.horizon,
            tolerance=self.tolerance,
            command_min=self.command_min,
            command_max=self.command_max,
            mean_command=self.mean_command,
        )

    @classmethod
    def from_state_dict(cls, state, device):
        bank = object.__new__(cls)
        for name, value in state.items():
            setattr(
                bank,
                name,
                (
                    {k: v.to(device) for k, v in value.items()}
                    if isinstance(value, dict)
                    else value.to(device) if isinstance(value, torch.Tensor) else value
                ),
            )
        bank.offsets = torch.arange(bank.horizon + 1, device=device)
        return bank


class ReferencePlayer:
    """Explicit per-environment cursor; no hidden or randomly changing phase."""

    def __init__(self, bank, batch, state=None):
        self.bank = bank
        device = bank.starts.device
        self.state = (
            dict(
                cursor=torch.zeros(batch, dtype=torch.long, device=device),
                remaining=torch.zeros(batch, dtype=torch.long, device=device),
                valid=torch.zeros(batch, dtype=torch.bool, device=device),
            )
            if state is None
            else {name: value.clone() for name, value in state.items()}
        )

    @torch.no_grad()
    def prepare(self, command, joint_pos):
        """Return next-frame targets. Re-select only at an explicit window end."""
        expired = self.state["remaining"] == 0
        if expired.any():
            start, valid = self.bank.select(command[expired], joint_pos[expired])
            self.state["cursor"][expired] = start
            self.state["remaining"][expired] = self.bank.horizon
            self.state["valid"][expired] = valid
        target = {name: value[self.state["cursor"] + 1] for name, value in self.bank.frames.items()}
        # Hiking may change its velocity latch within a target segment. Do
        # not continue imitation when that newly requested motion conflicts.
        compatible = ((target["command"] - command).abs() <= self.bank.tolerance).all(-1)
        valid = self.state["valid"] & compatible
        features = torch.cat(
            (
                target["joint_pos"],
                target["foot_pos"].flatten(1),
                target["contact"],
                self.state["remaining"][:, None] / self.bank.horizon,
                valid[:, None],
            ),
            -1,
        )
        return torch.where(valid[:, None], features, torch.zeros_like(features)), target, valid

    def advance(self, reset_ids=None):
        self.state["cursor"] += 1
        self.state["remaining"] -= 1
        if reset_ids is not None:
            self.state["remaining"][reset_ids] = 0


def tracking_reward(joint_pos, joint_vel, foot_pos, contact, target, valid, dt):
    """Dense differentiable tracking on predicted state; detached dataset targets.

    No reference world position or world heading is tracked. Hiking retains
    ownership of travel direction and speed. Weights are per second, matching
    native task reward scaling.
    """
    errors = dict(
        reference_joint_pos=(joint_pos - target["joint_pos"].detach()).square().mean(-1) / 0.25**2,
        reference_joint_vel=(joint_vel - target["joint_vel"].detach()).square().mean(-1) / 3.0**2,
        reference_foot_pos=(foot_pos - target["foot_pos"].detach()).square().mean((-1, -2)) / 0.08**2,
        reference_contact=(contact - target["contact"].detach()).square().mean(-1) / 0.5**2,
    )
    weights = dict(reference_joint_pos=2.5, reference_joint_vel=0.25, reference_foot_pos=2.0, reference_contact=2.0)
    terms = {
        name: torch.where(valid, (-error).exp() * weights[name] * dt, torch.zeros_like(error))
        for name, error in errors.items()
    }
    return sum(terms.values()), terms
