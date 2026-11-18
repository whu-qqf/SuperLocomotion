"""Action-conditioned visual RSSM for Hiking.

The discrete state, straight-through sampling, unimix, balanced KL and normalized
GRU follow WMP/DreamerV3-torch. See THIRD_PARTY_LICENSE.txt for attribution.
The observation interface, 29-joint action clock and heads are specific to G1.
No simulator state, terrain scan or future observation enters the policy feature.
"""

import torch
from dataclasses import dataclass
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class PerceptionCfg:
    deter: int = 512
    stoch: int = 32
    categories: int = 32
    embed: int = 256
    action_dim: int = 29
    action_steps: int = 5
    proprio_dim: int = 64
    image_height: int = 18
    image_width: int = 32
    unimix: float = 0.01
    free_nats: float = 1.0
    dynamics_kl_weight: float = 0.5
    representation_kl_weight: float = 0.1
    overshoot_steps: int = 6
    overshoot_weight: float = 0.1

    def __post_init__(self):
        dimensions = (self.deter, self.stoch, self.embed, self.action_dim, self.action_steps, self.proprio_dim)
        if min(dimensions) < 1 or self.categories < 2:
            raise ValueError("Perception dimensions must be positive; categories >= 2.")
        if not 0 <= self.unimix < 1 or self.overshoot_steps < 1:
            raise ValueError("Invalid unimix or overshooting horizon.")

    @property
    def feature_dim(self):
        return self.deter + self.stoch * self.categories


@dataclass
class RSSMState:
    deter: torch.Tensor
    stoch: torch.Tensor
    logits: torch.Tensor

    def detach(self):
        return RSSMState(self.deter.detach(), self.stoch.detach(), self.logits.detach())

    def index(self, indices):
        return RSSMState(self.deter[indices], self.stoch[indices], self.logits[indices])

    @staticmethod
    def choose(mask, yes, no):
        return RSSMState(
            torch.where(mask[..., None], yes.deter, no.deter),
            torch.where(mask[..., None, None], yes.stoch, no.stoch),
            torch.where(mask[..., None, None], yes.logits, no.logits),
        )


def _mlp(inp, hidden, out):
    return nn.Sequential(nn.Linear(inp, hidden), nn.LayerNorm(hidden, eps=1e-3), nn.SiLU(), nn.Linear(hidden, out))


def _symlog(x):
    return x.sign() * torch.log1p(x.abs())


class PerceptionModel(nn.Module):
    def __init__(self, cfg: PerceptionCfg = PerceptionCfg()):
        super().__init__()
        self.cfg = cfg
        flat = cfg.stoch * cfg.categories
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * ((cfg.image_height + 3) // 4) * ((cfg.image_width + 3) // 4), cfg.embed),
            nn.SiLU(),
        )
        self.observation_encoder = _mlp(cfg.embed + cfg.proprio_dim + 2, cfg.embed, cfg.embed)
        self.transition_input = nn.Sequential(
            nn.Linear(flat + cfg.action_steps * cfg.action_dim, cfg.deter, bias=False),
            nn.LayerNorm(cfg.deter, eps=1e-3),
            nn.SiLU(),
        )
        # WMP's normalized GRU: update bias -1, candidate gate before tanh.
        self.gru = nn.Sequential(
            nn.Linear(2 * cfg.deter, 3 * cfg.deter, bias=False), nn.LayerNorm(3 * cfg.deter, eps=1e-3)
        )
        self.prior_head = _mlp(cfg.deter, cfg.deter, flat)
        self.posterior_head = _mlp(cfg.deter + cfg.embed, cfg.deter, flat)
        self.initial_deter = nn.Parameter(torch.zeros(cfg.deter))
        self.depth_decoder = _mlp(cfg.feature_dim, cfg.embed, cfg.image_height * cfg.image_width)
        self.proprio_decoder = _mlp(cfg.feature_dim, cfg.embed, cfg.proprio_dim)

    def probabilities(self, logits):
        p = logits.softmax(-1)
        return (1 - self.cfg.unimix) * p + self.cfg.unimix / self.cfg.categories

    def _stochastic(self, logits, sample, generator=None):
        probs = self.probabilities(logits)
        if sample:
            indices = torch.multinomial(probs.reshape(-1, self.cfg.categories), 1, generator=generator)
            indices = indices.reshape(probs.shape[:-1])
        else:
            indices = probs.argmax(-1)
        hard = F.one_hot(indices, self.cfg.categories).to(probs.dtype)
        return hard + probs - probs.detach()

    def initial(self, batch_size, sample=True, generator=None):
        # Runtime resets write individual rows; expanded views alias envs.
        h = self.initial_deter.tanh().expand(batch_size, -1).clone()
        logits = self.prior_head(h).reshape(batch_size, self.cfg.stoch, self.cfg.categories)
        return RSSMState(h, self._stochastic(logits, sample, generator), logits)

    def feature(self, state):
        """Expose the current posterior immediately, including current image evidence."""
        return torch.cat((state.deter, self.probabilities(state.logits).flatten(-2)), -1)

    def decoder_feature(self, state):
        return torch.cat((state.deter, state.stoch.flatten(-2)), -1)

    def imagine(self, previous, actions, sample=True, generator=None):
        expected = (self.cfg.action_steps, self.cfg.action_dim)
        if actions.shape[-2:] != expected:
            raise ValueError(f"RSSM needs ordered actions of shape (..., {expected}), got {actions.shape}.")
        x = self.transition_input(torch.cat((previous.stoch.flatten(-2), actions.flatten(-2)), -1))
        reset, candidate, update = self.gru(torch.cat((x, previous.deter), -1)).chunk(3, -1)
        candidate = (reset.sigmoid() * candidate).tanh()
        update = (update - 1).sigmoid()
        h = update * candidate + (1 - update) * previous.deter
        logits = self.prior_head(h).reshape(*h.shape[:-1], self.cfg.stoch, self.cfg.categories)
        return RSSMState(h, self._stochastic(logits, sample, generator), logits)

    def observe(self, previous, actions, depth, proprio, frame_age, observed, is_first, sample=True, generator=None):
        initial = self.initial(len(actions), sample, generator)
        previous = RSSMState.choose(is_first.bool(), initial, previous)
        actions = torch.where(is_first[:, None, None].bool(), torch.zeros_like(actions), actions)
        prior = self.imagine(previous, actions, sample, generator)
        depth = depth.float() / 255 if depth.dtype == torch.uint8 else depth.float()
        # A missing image is not an observed all-zero depth image.
        depth = torch.where(observed[:, None, None, None].bool(), depth, torch.zeros_like(depth))
        image = self.image_encoder(depth)
        obs = torch.cat((image, _symlog(proprio), frame_age[:, None], observed[:, None].float()), -1)
        embedding = self.observation_encoder(obs)
        logits = self.posterior_head(torch.cat((prior.deter, embedding), -1))
        logits = logits.reshape(-1, self.cfg.stoch, self.cfg.categories)
        posterior = RSSMState(prior.deter, self._stochastic(logits, sample, generator), logits)
        return RSSMState.choose(observed.bool(), posterior, prior), prior

    def kl(self, posterior_logits, prior_logits):
        post = self.probabilities(posterior_logits)
        prior = self.probabilities(prior_logits)
        return (post * (post.log() - prior.log())).sum((-1, -2))

    def balanced_kl(self, posterior, prior):
        dyn = self.kl(posterior.logits.detach(), prior.logits).clamp(min=self.cfg.free_nats)
        rep = self.kl(posterior.logits, prior.logits.detach()).clamp(min=self.cfg.free_nats)
        return self.cfg.dynamics_kl_weight * dyn + self.cfg.representation_kl_weight * rep

    def sequence_loss(self, batch, initial=None, sample=True, generator=None):
        """Batch-major tick sequences. Padding is masked; reset is explicit.

        actions[t] contains the controls BEFORE observation[t]. Prefix replay
        supplies initial with current parameters; target segments use BPTT.
        """
        actions = batch["actions"]
        b, t = actions.shape[:2]
        if not batch["mask"].bool().any():
            raise ValueError("Perception loss requires at least one valid tick.")
        state = self.initial(b, sample, generator) if initial is None else initial
        states, depth_losses, proprio_losses, kls = [], [], [], []
        for j in range(t):
            valid = batch["mask"][:, j].bool()
            observed = batch["observed"][:, j].bool() & valid
            posterior, prior = self.observe(
                state,
                actions[:, j],
                batch["depth"][:, j],
                batch["proprio"][:, j],
                batch["frame_age"][:, j],
                observed,
                batch["is_first"][:, j],
                sample,
                generator,
            )
            state = RSSMState.choose(valid, posterior, state)
            states.append(state)
            feat = self.decoder_feature(state)
            predicted_depth = (
                self.depth_decoder(feat).sigmoid().reshape(b, 1, self.cfg.image_height, self.cfg.image_width)
            )
            target = batch["depth"][:, j]
            target = target.float() / 255 if target.dtype == torch.uint8 else target.float()
            target = torch.where(observed[:, None, None, None], target, torch.zeros_like(target))
            depth_losses.append((predicted_depth - target).square().mean((1, 2, 3)))
            proprio_losses.append((self.proprio_decoder(feat) - _symlog(batch["proprio"][:, j])).square().mean(-1))
            kls.append(self.balanced_kl(state, prior))
        valid = batch["mask"].bool()
        seen = batch["observed"].bool() & valid

        def average(values, mask):
            return torch.where(mask, torch.stack(values, 1), 0).sum() / mask.sum().clamp(min=1)

        depth_loss = average(depth_losses, seen)
        proprio_loss = average(proprio_losses, valid)
        kl_loss = average(kls, seen)
        # Train the very same action-only prior used by imagined trajectories.
        overshoots, overshoot_masks = [], []
        for start in range(t - 1):
            predicted = states[start].detach()
            intact = valid[:, start].clone()
            for end in range(start + 1, min(t, start + self.cfg.overshoot_steps + 1)):
                intact = intact & valid[:, end] & ~batch["is_first"][:, end].bool()
                predicted = self.imagine(predicted, actions[:, end], sample, generator)
                if end - start >= 2:
                    overshoots.append(self.kl(states[end].logits.detach(), predicted.logits))
                    overshoot_masks.append(intact & seen[:, end])
        overshoot_loss = average(overshoots, torch.stack(overshoot_masks, 1)) if overshoots else depth_loss * 0
        total = depth_loss + proprio_loss + kl_loss + self.cfg.overshoot_weight * overshoot_loss
        return (
            total,
            {
                "depth": depth_loss.detach(),
                "proprio": proprio_loss.detach(),
                "kl": kl_loss.detach(),
                "overshoot": overshoot_loss.detach(),
                "valid_ticks": valid.sum().detach(),
                "observed_ticks": seen.sum().detach(),
            },
            state,
        )
