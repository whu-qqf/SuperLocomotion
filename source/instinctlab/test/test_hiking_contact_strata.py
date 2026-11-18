"""Physics-rate phase boundaries and descriptive group counts."""

import torch
from types import SimpleNamespace

from instinctlab.tasks.parkour.supermimic.contact_strata import (
    contact_phases,
    event_metrics,
    stratify_replay_contacts,
    summarize_records,
)


def test_phases_include_boundary_from_previous_control_step():
    previous = torch.tensor([[0, 1]])
    labels = torch.tensor([[[1, 1], [1, 0], [0, 0], [0, 1]]])
    expected = torch.tensor([[[2, 1], [1, 3], [3, 0], [0, 2]]])
    torch.testing.assert_close(contact_phases(previous, labels), expected)


def test_strata_partition_errors_and_use_training_material_boundaries():
    labels = torch.tensor([[[0, 1], [1, 0], [1, 0], [0, 1]]] * 4).float()
    probability = labels * 0.8 + 0.1
    probability[3, 0, 0] = 0.9  # Exactly one held-out false positive.
    records = {
        "probability": probability,
        "labels": labels,
        "phase": contact_phases(torch.tensor([[0, 1]] * 4), labels),
        "env_id": torch.arange(4),
        "terrain_column": torch.tensor([0, 1, 0, 1]),
        "terrain_row": torch.zeros(4, dtype=torch.long),
        "terminal": torch.tensor([False, False, False, True]),
        "previous_timers": torch.zeros(4, 4),
        "timer_error": torch.zeros(4, 4),
        "foot_mu": torch.tensor([[0.2, 0.4], [0.6, 0.8], [1.0, 1.2], [100.0, 100.0]]),
        "foot_restitution": torch.tensor([[0.0, 0.1], [0.2, 0.3], [0.4, 0.5], [100.0, 100.0]]),
    }
    records["previous_timers"][3, 3] = 0.8
    records["timer_error"][3, 3] = 0.2
    result = summarize_records(records, torch.tensor([3]))
    assert result["training"]["overall"]["errors"] == 0
    assert result["heldout"]["overall"]["errors"] == 1
    assert result["heldout"]["overall"]["false_positives"] == 1
    support = result["heldout"]["one_step_clocks"]["support"]
    assert support["previous_ge_0.5s"]["share_of_squared_error"] == 1.0
    assert support["previous_lt_0.5s"]["share_of_squared_error"] == 0.0
    assert result["heldout"]["persistence_baseline"]["events"] == labels[3].numel()
    cuts = result["material_tertile_boundaries_from_training_env_feet"]["foot_mu"]
    assert max(cuts) < 1.2
    for pool in ("training", "heldout"):
        groups = result[pool]["groups"]
        for prefix in ("phase/", "terrain_column/", "foot_mu/", "foot_restitution/", "terminal/"):
            selected = [v for k, v in groups.items() if k.startswith(prefix)]
            assert sum(v["events"] for v in selected) == result[pool]["overall"]["events"]
            assert sum(v.get("errors", 0) for v in selected) == result[pool]["overall"]["errors"]
    assert event_metrics(probability, labels, torch.zeros_like(labels, dtype=torch.bool)) == {"events": 0}


def test_exhaustive_pairs_follow_ring_order_and_use_terminal_target():
    class Model(torch.nn.Module):
        def forward(self, state, action, context):
            return action

    grid = torch.tensor([[[0.0, 0.0, 0.0], [8.0, 0.0, 0.0]]])
    task = SimpleNamespace(
        dt=0.02,
        clock_substeps=4,
        parts=lambda state: {"contact_state": state},
        context_parts=lambda context: {"shape_friction": context[:, :2], "shape_restitution": context[:, 2:]},
        shape_materials=SimpleNamespace(labels=["left_ankle_roll_link:0", "right_ankle_roll_link:0"]),
        env=SimpleNamespace(
            scene=SimpleNamespace(terrain=SimpleNamespace(terrain_origins=grid, terrain_types=torch.arange(2)))
        ),
    )
    current = torch.zeros(3, 2, 21)
    current[2, :, :2] = 0.02
    current[0, 0, :2] = 0.04
    final = current.clone()
    final[0, 1, :2] = 0.04  # Terminal state, before the current observation reset.
    done = torch.zeros(3, 2, 1)
    done[0, 1] = 1
    replay = {
        "filled": 3,
        "capacity_per_env": 3,
        "cursor": 1,
        "num_envs": 2,
        "buffer/contact_state": current,
        "buffer/contact_state@final": final,
        "buffer/action": final.clone(),
        "buffer/env_origin": grid.expand(3, -1, -1),
        "buffer/command_time_left": torch.ones(3, 2, 1),
        "buffer/shape_friction": torch.full((3, 2, 2), 0.5),
        "buffer/shape_restitution": torch.zeros(3, 2, 2),
        "buffer/done": done,
    }
    alg = SimpleNamespace(
        imagination=task,
        fit_terminal_transitions=True,
        state_names=["contact_state"],
        context_names=["shape_friction", "shape_restitution"],
        world_model=Model(),
        device="cpu",
        validation_env_ids=torch.tensor([1]),
    )
    summary, records = stratify_replay_contacts(alg, replay, chunk_size=3)
    assert summary["available_pairs"] == 4
    assert records["replay_index"].tolist() == [1, 1, 2, 2]
    assert records["terminal"].tolist() == [False, False, False, True]
    assert records["timer_error"].abs().max() == 0
    assert summary["oracle_clock_max_error_seconds"] < 1e-7
