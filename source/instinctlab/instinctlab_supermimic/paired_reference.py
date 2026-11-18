"""Terrain-bound reference conditions and rewards, shared by real and imagined steps."""

import torch

from .rotations import mv, quat_to_mat

PAIRED_REFERENCE_INPUT_DIM = 59  # q29, feet6, contact2+mask1, touchdown6, time2, mask2, root delta3, rotation6, phase2


class PairedReferenceTargets:
    """Processed terrain-bound targets; usable at inference without simulator imports."""

    def query(self, clip_ids, frames, origins, terrain_ids):
        """Lookup B x T frames with explicit endpoint and measured-contact masks."""
        if (clip_ids < 0).any() or (clip_ids >= len(self.lengths)).any():
            raise ValueError("Paired query clip is outside the exported library.")
        if not torch.equal(self.terrain_ids[clip_ids], terrain_ids):
            raise ValueError("Reference clip does not belong to the environment's terrain.")
        if frames.ndim != 2 or len(frames) != len(clip_ids) or origins.shape != (len(clip_ids), 3):
            raise ValueError("Paired query requires B clip IDs, BxT frame indices and Bx3 origins.")
        lengths = self.lengths[clip_ids, None]
        valid = (frames >= 0) & (frames < lengths)
        safe = torch.minimum(frames.clamp_min(0), lengths - 1) + self.starts[clip_ids, None]
        result = {name: value[safe].detach() for name, value in self.frames.items()}
        for name in ("root_pos", "foot_pos", "touchdown_pos"):
            shift = origins[:, None] if name == "root_pos" else origins[:, None, None]
            result[name] = result[name] + shift
        result["valid"] = valid
        result["contact_valid"] = result["contact_valid"] & valid
        result["touchdown_valid"] = result["touchdown_valid"] & valid[..., None]
        return result

    def state_dict(self):
        return dict(
            format="paired_reference_targets_v1",
            frames=self.frames,
            lengths=self.lengths,
            starts=self.starts,
            terrain_ids=self.terrain_ids,
            dt=self.dt,
            contract=self.contract,
        )

    @classmethod
    def from_state_dict(cls, state, device="cpu"):
        if state["format"] != "paired_reference_targets_v1":
            raise ValueError("Unsupported paired reference target format.")
        targets = cls()
        targets.device = torch.device(device)
        targets.frames = {key: value.to(device) for key, value in state["frames"].items()}
        for name in ("lengths", "starts", "terrain_ids"):
            setattr(targets, name, state[name].to(device))
        targets.dt, targets.contract = state["dt"], state["contract"]
        return targets


def paired_reference_features(root_pos, root_rotation, target, dt):
    """Next-frame targets in the current torso frame, with measured-event masks.

    Reference labels are detached. Coordinates retain derivatives with respect
    to the predicted torso pose during imagination. No terrain/clip rematching
    happens here; the caller owns the explicit recorded timeline.
    """
    target = {key: value.detach() for key, value in target.items()}
    inverse = root_rotation.transpose(-1, -2)
    feet = mv(inverse[..., None, :, :], target["foot_pos"] - root_pos[..., None, :])
    touchdown = mv(inverse[..., None, :, :], target["touchdown_pos"] - root_pos[..., None, :])
    touchdown = torch.where(target["touchdown_valid"][..., None], touchdown, torch.zeros_like(touchdown))
    times = torch.where(target["touchdown_valid"], target["touchdown_steps"] * dt, 0)
    root_delta = mv(inverse, target["root_pos"] - root_pos)
    rotation = inverse @ quat_to_mat(target["root_quat"])
    contact = torch.where(target["contact_valid"][..., None], target["contact"], 0)
    features = torch.cat(
        (
            target["joint_pos"],
            feet.flatten(-2),
            contact,
            target["contact_valid"][..., None],
            touchdown.flatten(-2),
            times,
            target["touchdown_valid"],
            root_delta,
            rotation[..., :2].flatten(-2),
            target["phase"][..., None],
            target["remaining_time"][..., None],
        ),
        -1,
    )
    return torch.where(target["valid"][..., None], features, torch.zeros_like(features))


def paired_tracking_reward(state, foot_pos, contact, target, dt):
    """World trajectory, full joint pose and measured contact tracking.

    ``state`` uses the native (p, R, v, w, q, qd) link-origin convention.
    Contact validity masks only contact rewards, not the pose reward. At a
    padded clip endpoint all terms vanish; callers must also terminate and
    mask critic bootstrap there. These initial weights are not tuned results.
    """
    target = {key: value.detach() for key, value in target.items()}
    p, rotation, velocity, angular_velocity, q, qd = state
    errors = dict(
        paired_root_pos=((p - target["root_pos"]) / 0.15).square().mean(-1),
        paired_root_rotation=((rotation - quat_to_mat(target["root_quat"])) / 0.3).square().mean((-1, -2)),
        paired_root_velocity=((velocity - target["root_velocity"]) / 0.5).square().mean(-1),
        paired_root_angular_velocity=((angular_velocity - target["root_angular_velocity"]) / 1.0).square().mean(-1),
        paired_joint_pos=((q - target["joint_pos"]) / 0.25).square().mean(-1),
        paired_joint_vel=((qd - target["joint_vel"]) / 3.0).square().mean(-1),
        paired_foot_pos=((foot_pos - target["foot_pos"]) / 0.08).square().mean((-1, -2)),
        paired_contact=((contact - target["contact"].float()) / 0.5).square().mean(-1),
    )
    weights = dict(
        paired_root_pos=2.0,
        paired_root_rotation=1.0,
        paired_root_velocity=0.5,
        paired_root_angular_velocity=0.25,
        paired_joint_pos=2.5,
        paired_joint_vel=0.25,
        paired_foot_pos=2.0,
        paired_contact=2.0,
    )
    terms = {}
    for name, error in errors.items():
        valid = target["valid"]
        if name == "paired_contact":
            valid = valid & target["contact_valid"]
        terms[name] = torch.where(valid, (-error).exp() * weights[name] * dt, torch.zeros_like(error))
    return sum(terms.values()), terms
