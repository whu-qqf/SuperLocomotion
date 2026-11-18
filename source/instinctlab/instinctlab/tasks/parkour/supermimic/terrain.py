"""Differentiable queries of a fixed, world-aligned terrain height raster.

The raster represents the highest downward ray hit. It cannot represent overhangs or
vertical collision surfaces; collision rewards still need the original terrain mesh.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TerrainHeightField(nn.Module):
    """Bilinear heights, with explicit validity instead of extrapolated terrain.

    ``heights[y, x]`` has world origin ``origin_xy`` and spacing ``resolution``.
    Invalid raster cells must contain a finite placeholder and have ``valid=False``.
    Queries return a validity mask so a caller can reject unsupported rollouts.
    """

    def __init__(self, heights, valid, origin_xy, resolution: float, local_xy):
        super().__init__()
        if heights.ndim != 2 or min(heights.shape) < 2 or valid.shape != heights.shape:
            raise ValueError("heights and valid must be matching two-dimensional rasters")
        if resolution <= 0 or not torch.isfinite(heights).all():
            raise ValueError("terrain heights must be finite and resolution positive")
        self.resolution = float(resolution)
        self.register_buffer("heights", heights.float())
        self.register_buffer("valid", valid.bool())
        self.register_buffer("origin_xy", torch.as_tensor(origin_xy, dtype=torch.float32))
        self.register_buffer("local_xy", torch.as_tensor(local_xy, dtype=torch.float32))
        self.register_buffer(
            "span_xy",
            self.heights.new_tensor((heights.shape[1] - 1, heights.shape[0] - 1)) * self.resolution,
            persistent=False,
        )

    def query(self, xy: torch.Tensor, *, exact=False) -> tuple[torch.Tensor, torch.Tensor]:
        """Query (..., 2) points; exact uses nearest values with bilinear coordinate gradients."""
        grid = 2 * (xy - self.origin_xy) / self.span_xy - 1
        shape = xy.shape[:-1]
        sample_grid = grid.reshape(1, 1, -1, 2)
        values = F.grid_sample(
            self.heights[None, None], sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        ).reshape(shape)
        if exact:
            nearest = F.grid_sample(
                self.heights[None, None], sample_grid, mode="nearest", padding_mode="zeros", align_corners=True
            ).reshape(shape)
            values = values + (nearest - values).detach()
        support = F.grid_sample(
            self.valid[None, None].to(self.heights.dtype),
            sample_grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).reshape(shape)
        inside = (grid.abs() <= 1).all(dim=-1)
        return values, inside & (support >= 1 - 1e-6)

    @torch.no_grad()
    def build_support_field(self, window=0.5, hole_depth=0.25, sigma=0.05):
        """TGI-style local-top support mask, Gaussian-smoothed toward safe surfaces.

        Invalid cells and map padding carry zero support. This field belongs
        only to privileged reward geometry; it is not a visual model input.
        """
        if window <= 0 or hole_depth <= 0 or sigma <= 0:
            raise ValueError("Support field scales must be positive.")
        kernel = max(3, int(round(window / self.resolution)) | 1)
        height = self.heights.masked_fill(~self.valid, -torch.inf)[None, None]
        top = F.max_pool2d(height, kernel, stride=1, padding=kernel // 2)
        solid = (self.valid[None, None] & (height > top - hole_depth)).float()
        radius = max(1, int(round(3 * sigma / self.resolution)))
        offsets = torch.arange(-radius, radius + 1, device=height.device) * self.resolution
        gaussian = torch.exp(-0.5 * (offsets / sigma).square())
        gaussian = gaussian / gaussian.sum()
        smooth = F.conv2d(solid, gaussian.view(1, 1, -1, 1), padding=(radius, 0))
        smooth = F.conv2d(smooth, gaussian.view(1, 1, 1, -1), padding=(0, radius))
        self.register_buffer("support_field", smooth[0, 0], persistent=False)

    def query_support(self, xy):
        """Smooth [0,1] support, with coordinate derivatives and zero outside the map."""
        grid = 2 * (xy - self.origin_xy) / self.span_xy - 1
        return (
            F.grid_sample(
                self.support_field[None, None],
                grid.reshape(1, 1, -1, 2),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )
            .reshape(xy.shape[:-1])
            .clamp(0, 1)
        )

    def scan(self, root_pos: torch.Tensor, root_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Root-centred yaw-aligned scan, returned as root z minus ground z.

        Quaternion order is XYZW. The ray's elevation offset never rotates with
        root tilt, unlike a sensor offset attached to the torso.
        """
        x, y, z, w = root_quat.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = yaw.cos().unsqueeze(-1), yaw.sin().unsqueeze(-1)
        lx, ly = self.local_xy.unbind(-1)
        rotated = torch.stack((c * lx - s * ly, s * lx + c * ly), dim=-1)
        world_xy = root_pos[..., None, :2] + rotated
        heights, valid = self.query(world_xy)
        return root_pos[..., 2:3] - heights, valid.all(dim=-1)
