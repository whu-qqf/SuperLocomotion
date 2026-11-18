"""Physical SuperMimic F and value/failure heads shared across both backends.

Adapted from the SuperMimic reference implementation (MIT License), Chong Zhang / ETH Zurich.
Moved from the verified InstinctMJ implementation without changing numerical equations.
The dual-model route uses z_dim=0; perception is owned by instinctlab_perception.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .physical import INV_DIM, NUM_J, invariant_features
from .rotations import mm, mv, so3_exp, so3_log, yaw_mat

WM_OUT = 3 + 3 + 3 + 3 + NUM_J + NUM_J
AMP_DTYPE: torch.dtype | None = None


def mlp(inp: int, hidden: list[int], out: int, act=nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers, d = [], inp
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(act())
        d = h
    layers.append(nn.Linear(d, out))
    return nn.Sequential(*layers)


class RunningNorm(nn.Module):
    """Running mean/std normalizer; updated from data without gradients, frozen after ``max_count``."""

    def __init__(self, dim: int, max_count: float = 2e8, eps: float = 1e-2):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros(()))
        self.max_count, self.eps = max_count, eps
        self._host_count = 0.0  # host-side mirror of ``count`` (avoids a device sync per update)

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        self._host_count = float(self.count)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.reshape(-1, x.shape[-1]).float()
        if self._host_count >= self.max_count:
            return
        n = x.shape[0]
        self._host_count += n
        m, v = x.mean(0), x.var(0, unbiased=False)
        tot = self.count + n
        delta = m - self.mean
        self.mean += delta * n / tot
        self.var = (self.var * self.count + v * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x):
        return (x - self.mean) / torch.sqrt(self.var + self.eps**2)

    def std(self):
        return torch.sqrt(self.var + self.eps**2)


def masked_latent_moments(z: torch.Tensor, mask: torch.Tensor):
    """Population moments of valid transitions, excluding post-reset padding.

    Select before arithmetic: multiplying invalid values by zero would still
    propagate NaNs. Population variance also stays defined for one valid row.
    """
    valid = z.reshape(-1, z.shape[-1])[mask.reshape(-1).bool()]
    if valid.shape[0] == 0:
        raise ValueError("Latent statistics require a valid transition.")
    return valid.mean(0), valid.var(0, unbiased=False)


def run_mlp(net: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if AMP_DTYPE is None:
        return net(x)
    with torch.autocast("cuda", dtype=AMP_DTYPE):
        return net(x).float()


class WorldModel(nn.Module):
    """Frame-local physical increments, contact logits and optional auxiliary readouts.

    Visual inputs use running normalization. With predict_latent=False they
    condition physics without a future-latent head; the next observation comes
    from the imagined camera. The legacy route predicts latent increments.
    With z_dim=0 neither route requires a visual input.
    """

    def __init__(
        self,
        scan_dim: int,
        num_contact: int,
        xi_dim: int,
        z_dim: int = 0,
        hidden=(2048, 2048, 2048),
        aux_dim=0,
        predict_latent=True,
    ):
        super().__init__()
        self.scan_dim = scan_dim
        self.in_dim = INV_DIM + NUM_J + xi_dim + scan_dim
        self.num_contact, self.z_dim = num_contact, z_dim
        self.predict_latent = predict_latent
        self.latent_out_dim = z_dim if predict_latent else 0
        self.aux_dim = aux_dim
        self.net = mlp(
            self.in_dim + z_dim,
            list(hidden),
            WM_OUT + num_contact + self.latent_out_dim + aux_dim,
            act=nn.SiLU,
            layer_norm=True,
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[WM_OUT : WM_OUT + num_contact] = -3.0  # contacts are rare
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(WM_OUT, eps=1e-4)
        self.aux_norm = RunningNorm(aux_dim, eps=1e-4) if aux_dim else None
        self.register_buffer("z_mean", torch.zeros(z_dim))
        self.register_buffer("z_std", torch.ones(z_dim))
        self.register_buffer("dz_std", torch.ones(self.latent_out_dim))

    @torch.no_grad()
    def update_z_stats(self, z: torch.Tensor, dz: torch.Tensor, w: torch.Tensor, rate: float = 0.01):
        """Normalize valid visual inputs; increment statistics belong to the legacy head only."""
        mean, var = masked_latent_moments(z, w)
        self.z_mean.lerp_(mean, rate)
        self.z_std.lerp_(var.sqrt().clamp(min=1e-3), rate)
        if self.latent_out_dim:
            valid_dz = dz.reshape(-1, self.z_dim)[w.reshape(-1).bool()]
            self.dz_std.lerp_(valid_dz.square().mean(0).sqrt().clamp(min=1e-3), rate)

    def inputs(self, S, a, xi_n, scan):
        return torch.cat((invariant_features(S), a, xi_n, scan), -1)

    @staticmethod
    def targets(S, S1):
        """Frame-local increments from S to S1 (the regression targets)."""
        p, R, v, w, q, qd = S
        p1, R1, v1, w1, q1, qd1 = S1
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv(H.transpose(-1, -2), p1 - p),
                so3_log(mm(Rt, R1)),
                mv(Rt, v1 - v),
                mv(Rt, w1 - w),
                q1 - q,
                qd1 - qd,
            ),
            -1,
        )

    def step(self, S, a, xi_n, scan, z=None):
        """Returns the next state tuple, the contact logits at the next state and the next visual latent (None if
        ``z_dim = 0``)."""
        return self.step_with_aux(S, a, xi_n, scan, z)[:3]

    def step_with_aux(self, S, a, xi_n, scan, z=None):
        """Also return supervised physical readouts, e.g. native torque/qacc.

        Auxiliary readouts are outputs of F, not recurrent state or reward
        predictions. aux_dim=0 preserves the original checkpoint layout.
        """
        p, R, v, w, q, qd = S
        x = self.in_norm(self.inputs(S, a, xi_n, scan))
        y = run_mlp(self.net, x if self.z_dim == 0 else torch.cat((x, (z - self.z_mean) / self.z_std), -1))
        d = y[..., :WM_OUT] * self.out_norm.std() + self.out_norm.mean
        H = yaw_mat(R)
        dp, dth, dv, dw, dq, dqd = d.split((3, 3, 3, 3, NUM_J, NUM_J), -1)
        p1 = p + mv(H, dp)
        R1 = mm(R, so3_exp(dth))
        v1 = v + mv(R, dv)
        w1 = w + mv(R, dw)
        start = WM_OUT + self.num_contact
        z1 = None if self.latent_out_dim == 0 else z + y[..., start : start + self.latent_out_dim] * self.dz_std
        aux = y[..., start + self.latent_out_dim :]
        if self.aux_norm is not None:
            aux = aux * self.aux_norm.std() + self.aux_norm.mean
        return (p1, R1, v1, w1, q + dq, qd + dqd), y[..., WM_OUT : WM_OUT + self.num_contact], z1, aux

    @staticmethod
    def state_error(S_hat, S):
        """Frame-local state error components (same layout as ``targets``)."""
        p, R, v, w, q, qd = S
        ph, Rh, vh, wh, qh, qdh = S_hat
        H = yaw_mat(R)
        Rt = R.transpose(-1, -2)
        return torch.cat(
            (
                mv(H.transpose(-1, -2), ph - p),
                so3_log(mm(Rt, Rh)),
                mv(Rt, vh - v),
                mv(Rt, wh - w),
                qh - q,
                qdh - qd,
            ),
            -1,
        )


class FailureHead(nn.Module):
    """Dreamer-style "continue" predictor: P(the transition into this state is a REAL termination), from the privileged critic
    features of the (true or imagined) state (invariant state, robot-vs-reference errors, xi, goal, terrain scan)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.norm = RunningNorm(in_dim)
        self.net = mlp(in_dim, [256, 128], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Logits (B,)."""
        return run_mlp(self.net, self.norm(x)).squeeze(-1)


class Critic(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = mlp(in_dim, [256, 128], 1)
        self.norm = RunningNorm(in_dim)

    def forward(self, x):
        return run_mlp(self.net, self.norm(x)).squeeze(-1)
