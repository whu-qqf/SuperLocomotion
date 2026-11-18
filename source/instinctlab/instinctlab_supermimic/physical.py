"""G1 physical state: root-link origin, world twists, policy-order joints.

Adapted from the SuperMimic reference implementation (MIT License), Chong Zhang / ETH Zurich.
Quaternion storage is WXYZ; callers must explicitly identify the root link.
"""

import torch

from .rotations import mat_to_6d, mat_to_quat, mm, mv, quat_to_mat, yaw_mat

NUM_J = 29
STATE_DIM = 71
INV_DIM = 6 + 3 + 3 + NUM_J + NUM_J


def flatten_state(S) -> torch.Tensor:
    p, R, v, w, q, qd = S
    return torch.cat((p, mat_to_quat(R), v, w, q, qd), -1)


def unflatten_state(x: torch.Tensor):
    return x[..., 0:3], quat_to_mat(x[..., 3:7]), x[..., 7:10], x[..., 10:13], x[..., 13:42], x[..., 42:71]


def invariant_features(S) -> torch.Tensor:
    """Heading- and position-invariant state features: tilt (6D), body-frame velocities, joints.

    (The reference version also contains the root height; on terrain the height scan carries that information.)
    """
    p, R, v, w, q, qd = S
    H = yaw_mat(R)
    Rt = R.transpose(-1, -2)
    return torch.cat(
        (mat_to_6d(mm(H.transpose(-1, -2), R)), mv(Rt, v), mv(Rt, w), q, qd),
        -1,
    )
