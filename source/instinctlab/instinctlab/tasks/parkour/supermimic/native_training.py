"""Flat-terrain diagnostic configuration for the current Hiking learner."""

def configure_flat_walk(cfg):
    """Isolate basic forward locomotion with the existing network and task heads."""
    from isaaclab.terrains import FlatPatchSamplingCfg, TerrainGeneratorCfg
    from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshPlaneTerrainCfg

    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=0,
        size=(30.0, 8.0),
        num_rows=1,
        num_cols=1,
        border_width=3.0,
        use_cache=False,
        curriculum=False,
        sub_terrains={
            "flat_walk": MeshPlaneTerrainCfg(
                flat_patch_sampling={
                    "target": FlatPatchSamplingCfg(
                        num_patches=50,
                        patch_radius=0.2,
                        x_range=(12.0, 13.0),
                        y_range=(-0.1, 0.1),
                        max_height_diff=0.001,
                    )
                },
            )
        },
    )
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.curriculum.terrain_levels = None
    command = cfg.commands.base_velocity
    command.random_velocity_terrain = []
    command.velocity_ranges = {
        "flat_walk": {"lin_vel_x": (0.5, 0.5), "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-0.5, 0.5)}
    }
    command.rel_standing_envs = 0.0
    return cfg
