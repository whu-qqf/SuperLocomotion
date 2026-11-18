"""Terrain type and difficulty rows remain separate in evaluation."""

import torch
from types import SimpleNamespace as NS

from instinctlab.tasks.parkour.supermimic.native_latent_evaluation import (
    assign_fixed_terrain_evaluation,
    terrain_column_type_names,
)


def test_terrain_columns_follow_tgi_proportion_grouping():
    generator = NS(
        num_cols=10,
        sub_terrains={
            "flat": NS(proportion=0.2),
            "stairs": NS(proportion=0.5),
            "stones": NS(proportion=0.3),
        },
    )
    env = NS(cfg=NS(scene=NS(terrain=NS(terrain_generator=generator))))
    assert terrain_column_type_names(env) == (
        "flat",
        "flat",
        "stairs",
        "stairs",
        "stairs",
        "stairs",
        "stairs",
        "stones",
        "stones",
        "stones",
    )


def test_fixed_terrain_evaluation_sets_one_type_and_one_level():
    generator = NS(
        num_cols=2,
        sub_terrains={"flat": NS(proportion=0.5), "stairs": NS(proportion=0.5)},
    )
    terrain_cfg = NS(terrain_generator=generator)
    terrain = NS(
        terrain_generator=generator,
        terrain_origins=torch.arange(3 * 2 * 3, dtype=torch.float32).reshape(3, 2, 3),
        terrain_levels=torch.zeros(4, dtype=torch.long),
        terrain_types=torch.zeros(4, dtype=torch.long),
        env_origins=torch.zeros(4, 3),
    )
    env = NS(cfg=NS(scene=NS(terrain=terrain_cfg)), scene=NS(terrain=terrain, env_origins=torch.zeros(4, 3)))
    info = assign_fixed_terrain_evaluation(env, "stairs", 2)
    assert info == {"terrain_type": "stairs", "terrain_column": 1, "terrain_level": 2}
    assert terrain.terrain_levels.tolist() == [2, 2, 2, 2]
    assert terrain.terrain_types.tolist() == [1, 1, 1, 1]
    torch.testing.assert_close(terrain.env_origins, terrain.terrain_origins[2, 1].expand(4, -1))
    torch.testing.assert_close(env.scene.env_origins, terrain.env_origins)
