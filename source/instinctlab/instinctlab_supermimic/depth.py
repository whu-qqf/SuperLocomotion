"""Depth encoding, native camera histories and inference-only controllers.

Legacy imagined-camera encoding learns physical prediction and reconstruction.
The terrain-model route also trains encoding from policy returns. Proprioceptive
feedback retains state/action derivatives through physics in both modes.
"""

import torch
from torch import nn

from .networks import RunningNorm, mlp, run_mlp
from .paired_reference import PAIRED_REFERENCE_INPUT_DIM, PairedReferenceTargets, paired_reference_features
from .reference import REFERENCE_INPUT_DIM, ReferencePlayer, ReferenceWindows

DEPTH_SHAPE = (8, 18, 32)
DEPTH_HISTORY = 36
LATENT_DIM = 32


class DepthAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(32 * 5 * 8, LATENT_DIM),
        )
        self.decoder = nn.Sequential(
            nn.Linear(LATENT_DIM, 32 * 5 * 8),
            nn.ELU(),
            nn.Unflatten(1, (32, 5, 8)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=(0, 1)),
            nn.ELU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def encode(self, depth):
        return run_mlp(self.encoder, depth.reshape(-1, *DEPTH_SHAPE).float()).reshape(*depth.shape[:-3], LATENT_DIM)

    def reconstruction_loss(self, depth):
        images = depth.reshape(-1, *DEPTH_SHAPE).float()
        return (run_mlp(self.decoder, self.encode(images)) - images).abs().mean()


class DepthLatentPolicy(nn.Module):
    """Eight proprio frames, depth z, and optional explicit motion targets."""

    def __init__(self, reference_dim=0, detach_latent=True, normalize_inputs=False):
        super().__init__()
        self.detach_latent = detach_latent
        self.reference_dim = reference_dim
        self.actor = mlp(768 + LATENT_DIM + reference_dim, [512, 256, 128], 29)
        self.normalize_inputs = normalize_inputs
        if normalize_inputs:
            self.proprio_norm = RunningNorm(768)
            self.latent_norm = RunningNorm(LATENT_DIM)
            with torch.no_grad():
                self.actor[-1].weight.mul_(0.01)
                self.actor[-1].bias.zero_()

    def forward(self, proprio, latent, reference=None):
        if self.normalize_inputs:
            proprio, latent = self.proprio_norm(proprio), self.latent_norm(latent)
        inputs = (proprio, latent.detach() if self.detach_latent else latent)
        if self.reference_dim:
            if reference is None or reference.shape[-1] != self.reference_dim:
                raise ValueError("Reference-conditioned actor requires its explicit motion targets.")
            inputs += (reference.detach(),)
        elif reference is not None:
            raise ValueError("This actor was trained without reference inputs.")
        return run_mlp(self.actor, torch.cat(inputs, -1))


class DepthHistory:
    """50 Hz camera history, eight frames at stride five; reset fills never mix episodes."""

    def __init__(self, image):
        self.images = image[:, None].expand(-1, DEPTH_HISTORY, -1, -1).clone()

    def stack(self):
        return self.images[:, ::5].clone()

    def advance(self, image):
        self.images = torch.cat((self.images[:, 1:], image[:, None]), 1)
        return self.stack()

    def reset(self, ids, image):
        self.images[ids] = image[ids, None]


class DepthLatentController(nn.Module):
    """Stateful inference with the training camera/proprio contract."""

    def __init__(self, reference_bank=None, normalize_inputs=False):
        super().__init__()
        self.encoder = DepthAutoencoder().encoder
        self.actor = DepthLatentPolicy(
            REFERENCE_INPUT_DIM if reference_bank is not None else 0, normalize_inputs=normalize_inputs
        )
        self.reference_bank = reference_bank
        self.reference_player = None
        self.history = None

    @torch.no_grad()
    def forward(self, proprio, image, reset_ids=None, command=None, joint_pos=None):
        if self.history is None:
            self.history = DepthHistory(image)
        else:
            self.history.advance(image)
        if reset_ids is not None:
            self.history.reset(reset_ids, image)
        reference = None
        if self.reference_bank is not None:
            if command is None or joint_pos is None:
                raise ValueError("Tracking controller needs velocity command and canonical measured joint positions.")
            if self.reference_player is None:
                self.reference_player = ReferencePlayer(self.reference_bank, len(proprio))
            if reset_ids is not None:
                self.reference_player.state["remaining"][reset_ids] = 0
            reference, _, _ = self.reference_player.prepare(command, joint_pos)
        action = self.actor(proprio, run_mlp(self.encoder, self.history.stack()), reference)
        if self.reference_player is not None:
            self.reference_player.advance()
        return action


class PairedDepthLatentController(nn.Module):
    """Stage-one replay with explicit measured root pose and terrain/clip/frame.

    The caller owns resets and the timeline. No world model, FK or source
    dataset is needed; the processed targets are embedded in the export.
    """

    def __init__(self, targets):
        super().__init__()
        self.encoder = DepthAutoencoder().encoder
        self.actor = DepthLatentPolicy(PAIRED_REFERENCE_INPUT_DIM)
        self.targets = targets
        self.history = None

    @torch.no_grad()
    def forward(self, proprio, image, root_pos, root_rotation, clip_ids, frames, origins, terrain_ids, reset_ids=None):
        if (frames < 0).any() or (frames >= self.targets.lengths[clip_ids] - 1).any():
            raise ValueError("Paired controller reached a clip endpoint; reset the reference and environment.")
        target = self.targets.query(clip_ids, frames[:, None] + 1, origins, terrain_ids)
        target = {name: value[:, 0] for name, value in target.items()}
        reference = paired_reference_features(root_pos, root_rotation, target, self.targets.dt)
        if self.history is None:
            self.history = DepthHistory(image)
        else:
            self.history.advance(image)
        if reset_ids is not None:
            self.history.reset(reset_ids, image)
        return self.actor(proprio, run_mlp(self.encoder, self.history.stack()), reference)


def load_depth_controller(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    if (
        payload["format"]
        not in (
            "hiking_depth_latent_controller_v1",
            "hiking_depth_latent_controller_v2",
            "hiking_depth_reference_controller_v1",
            "hiking_depth_paired_controller_v1",
        )
        or tuple(payload["depth_shape"]) != DEPTH_SHAPE
        or payload["depth_stride"] != 5
    ):
        raise ValueError("Unsupported depth latent controller contract.")
    bank = (
        ReferenceWindows.from_state_dict(payload["reference_bank"], device)
        if payload["format"] == "hiking_depth_reference_controller_v1"
        else None
    )
    controller = (
        PairedDepthLatentController(PairedReferenceTargets.from_state_dict(payload["paired_reference"], device))
        if payload["format"] == "hiking_depth_paired_controller_v1"
        else DepthLatentController(bank, normalize_inputs=payload["format"] == "hiking_depth_latent_controller_v2")
    ).to(device)
    controller.encoder.load_state_dict(payload["depth"])
    controller.actor.load_state_dict(payload["actor"])
    return controller.eval()
