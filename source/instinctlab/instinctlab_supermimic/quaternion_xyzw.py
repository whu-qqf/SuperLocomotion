"""Differentiable XYZW quaternion operations used by native Hiking sensors."""

import torch


def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    av, aw = a[..., :3], a[..., 3:]
    bv, bw = b[..., :3], b[..., 3:]
    vector = aw * bv + bw * av + torch.linalg.cross(av, bv)
    return torch.cat((vector, aw * bw - (av * bv).sum(-1, keepdim=True)), dim=-1)


def rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    axis, scalar = q[..., :3], q[..., 3:]
    twice_cross = 2.0 * torch.linalg.cross(axis, vector)
    return vector + scalar * twice_cross + torch.linalg.cross(axis, twice_cross)


def rotate_inverse(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    conjugate = torch.cat((-q[..., :3], q[..., 3:]), dim=-1)
    return rotate(conjugate, vector)
