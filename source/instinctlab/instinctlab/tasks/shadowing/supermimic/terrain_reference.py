"""Native reference tracking on physically paired Hiking terrain and motion."""

from __future__ import annotations

import copy
import torch
import yaml
from pathlib import Path

from isaaclab.managers import ObservationTermCfg

from instinctlab.motion_reference.motion_files.terrain_motion_cfg import TerrainMotionCfg
from instinctlab.tasks.parkour.supermimic.collection import rasterize_terrain
from instinctlab.tasks.shadowing.perceptive.perceptive_env_cfg import PerceptiveShadowingSceneCfg
from instinctlab.terrains.terrain_importer import TerrainImporter

from .supermimic_provider import build_imagination


class ReferenceTerrainImporter(TerrainImporter):
    """Keep the imported geometry and preserve Hiking's heightfield collision path.

    The motion-matched generator enters through ``import_ground_plane`` and bypasses
    IsaacLab's standard generator branch, so the collider tag must be authored here.
    """

    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        self.reference_mesh = mesh.copy()
        self._tag_heightfield_collider(self.terrain_prim_paths[-1], self.cfg.terrain_generator.horizontal_scale)


def reference_height_field(env):
    """One raster of the actual imported scene; query coordinates remain differentiable."""
    terrain = env.scene.terrain
    if not hasattr(terrain, "reference_height_field"):
        x = torch.linspace(-0.8, 0.8, 17, device=env.device)
        y = torch.linspace(-0.5, 0.5, 11, device=env.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        local_xy = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
        terrain.reference_height_field, _ = rasterize_terrain(terrain.reference_mesh, env.device, 0.025, local_xy)
    return terrain.reference_height_field


def height_scan(env):
    robot = env.scene["robot"]
    scan, _ = reference_height_field(env).scan(robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch)
    return scan.clamp(-5.0, 5.0)


def terrain_scan_out_of_bounds(env):
    robot = env.scene["robot"]
    _, valid = reference_height_field(env).scan(robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch)
    return ~valid


def build_terrain_imagination(env, obs_format):
    if tuple(obs_format["ref_cursor"]["ref_cursor"]) != (5,):
        raise ValueError("terrain reference replay requires clip/time and the recorded xyz motion origin")
    return build_imagination(env, obs_format, terrain=reference_height_field(env.unwrapped))


def configure_paired_terrain(cfg, dataset: str | Path):
    """Install the recorded geometry without requiring a motion-reference manager.

    Tile dimensions come from the dataset, including captured scan margins. The
    references and terrain are never scaled or fitted to one another.
    """
    dataset = Path(dataset).expanduser().resolve()
    metadata = yaml.safe_load((dataset / "metadata.yaml").read_text())
    audit = yaml.safe_load((dataset / "audit.json").read_text())
    if any(item["mesh_source"] != "recorded_collision_heightfield" for item in audit["terrains"]):
        raise ValueError("paired tracking requires the recorded collision surface, not a re-rasterized visual mesh")
    ids = [entry["terrain_id"] for entry in metadata["terrains"]]
    if ids != list(range(len(ids))):
        raise ValueError("motion-matched terrain IDs must follow their metadata order")
    cfg.scene.terrain = copy.deepcopy(PerceptiveShadowingSceneCfg(num_envs=cfg.scene.num_envs).terrain)
    cfg.scene.terrain.class_type = ReferenceTerrainImporter
    generator = cfg.scene.terrain.terrain_generator
    generator.num_rows, generator.num_cols = len(ids), 1
    generator.size = tuple(metadata["tile_size"])
    generator.curriculum = True
    generator.difficulty_range = (0.0, 1.0)
    generator.use_cache = False
    generator.seed = 1537
    generator.horizontal_scale = 0.05  # original released-teacher Hiking collider resolution
    sub = generator.sub_terrains["motion_matched"]
    sub.path = str(dataset)
    sub.metadata_yaml = str(dataset / "metadata.yaml")
    sub.convert_to_heightfield = True
    return cfg


def configure_terrain_reference(cfg, dataset: str | Path):
    """Use existing reference-reset/reward logic with recorded terrain."""
    from isaaclab.managers import TerminationTermCfg

    dataset = Path(dataset).expanduser().resolve()
    configure_paired_terrain(cfg, dataset)
    source = next(iter(cfg.scene.motion_reference.motion_buffers.values()))
    motion = TerrainMotionCfg()
    for name, value in source.__dict__.items():
        if name not in ("class_type", "path", "filtered_motion_selection_filepath"):
            setattr(motion, name, copy.deepcopy(value))
    motion.path = str(dataset)
    motion.metadata_yaml = str(dataset / "metadata.yaml")
    motion.filtered_motion_selection_filepath = None
    cfg.scene.motion_reference.motion_buffers = {"TerrainMotion": motion}
    cfg.scene.motion_reference.mp_split_method = "None"
    cfg.observations.ref_cursor.ref_cursor.params["include_origin"] = True
    # Plane resets perturb z/tilt across stair edges. Start from the physical reference;
    # broader terrain-safe reset perturbations can be introduced after tracking works.
    reset = cfg.events.reset_robot.params
    reset["position_offset"] = [0.0, 0.0, 0.0]
    reset["randomize_pose_range"] = {}
    reset["randomize_velocity_range"] = {}
    reset["randomize_joint_pos_range"] = (0.0, 0.0)
    cfg.observations.policy.height_scan = ObservationTermCfg(func=height_scan)
    cfg.observations.critic.height_scan = ObservationTermCfg(func=height_scan)
    cfg.terminations.terrain_scan_out_of_bounds = TerminationTermCfg(func=terrain_scan_out_of_bounds)
    return cfg
