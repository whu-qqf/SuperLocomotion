"""Terrain mesh retention and height-field rasterization for Hiking dynamics."""

import numpy as np
import torch

from instinctlab.terrains.terrain_importer import TerrainImporter


class DynamicsTerrainImporter(TerrainImporter):
    """Retain the exact imported mesh only for the dynamics collection experiment."""

    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        self.dynamics_mesh = mesh


def rasterize_terrain(mesh, device, resolution, local_xy):
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

    from .terrain import TerrainHeightField

    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
    origin = torch.tensor(mesh.bounds[0, :2], dtype=torch.float32, device=device)
    counts = np.ceil((mesh.bounds[1, :2] - mesh.bounds[0, :2]) / resolution).astype(int) + 1
    x = origin[0] + torch.arange(int(counts[0]), device=device) * resolution
    y = origin[1] + torch.arange(int(counts[1]), device=device) * resolution
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    starts = torch.stack((xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), 20.0)), -1)
    heights = []
    for chunk in starts.split(65536):
        directions = torch.zeros_like(chunk)
        directions[:, 2] = -1
        heights.append(raycast_mesh(chunk, directions, wp_mesh)[0][:, 2])
    heights = torch.cat(heights).reshape(len(y), len(x))
    valid = torch.isfinite(heights)
    terrain = TerrainHeightField(torch.where(valid, heights, 0), valid, origin, resolution, local_xy).to(device)
    return terrain, wp_mesh
