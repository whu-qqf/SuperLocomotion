"""Exhaustive logged one-step contact errors, with descriptive condition groups."""

import torch

from .contacts import advance_contact_history


def contact_phases(previous, labels):
    """Classify chronological physics events: air, support, touchdown, liftoff."""
    before = torch.cat((previous[:, None], labels[:, :-1]), 1).bool()
    after = labels.bool()
    return torch.where(before, torch.where(after, 1, 3), torch.where(after, 2, 0))


def event_metrics(probability, labels, selected):
    probability, labels = probability[selected], labels[selected].bool()
    count = labels.numel()
    if not count:
        return {"events": 0}
    decision = probability >= 0.5
    wrong = decision != labels
    positives = int(labels.sum())
    negatives = count - positives
    false_negative = int((~decision & labels).sum())
    false_positive = int((decision & ~labels).sum())
    return {
        "events": count,
        "errors": int(wrong.sum()),
        "error_rate": float(wrong.float().mean()),
        "brier": float((probability - labels.float()).square().mean()),
        "positive_events": positives,
        "false_negatives": false_negative,
        "false_positives": false_positive,
        "false_negative_rate": false_negative / positives if positives else None,
        "false_positive_rate": false_positive / negatives if negatives else None,
    }


def summarize_records(records, validation_env_ids):
    """Apply marginal groups; all counts refer to physics-rate foot clock events."""
    probability, labels = records["probability"], records["labels"]
    initial = (records["phase"][:, 0] == 1) | (records["phase"][:, 0] == 3)
    persistence = initial[:, None, :].expand_as(labels).float()
    heldout = torch.isin(records["env_id"], validation_env_ids.cpu())
    training = ~heldout
    # One parameter vector per training environment, not one per recorded step.
    envs = records["env_id"].unique(sorted=True)
    first = torch.stack([(records["env_id"] == env).nonzero()[0, 0] for env in envs])
    train_first = first[training[first]]
    thresholds = {
        name: torch.quantile(records[name][train_first].flatten(), torch.tensor([1 / 3, 2 / 3]))
        for name in ("foot_mu", "foot_restitution")
    }
    result = {"material_tertile_boundaries_from_training_env_feet": {k: v.tolist() for k, v in thresholds.items()}}
    for pool, rows in [("training", training), ("heldout", heldout)]:
        selected = rows[:, None, None].expand_as(labels)
        total = event_metrics(probability, labels, selected)
        groups = {}

        def add(name, mask):
            metric = event_metrics(probability, labels, selected & mask.expand_as(labels))
            if metric["events"]:
                metric["share_of_pool_errors"] = metric["errors"] / total["errors"] if total["errors"] else 0.0
            groups[name] = metric

        for index, phase in enumerate(("stable_air", "stable_support", "touchdown", "liftoff")):
            add("phase/" + phase, records["phase"] == index)
        for column in records["terrain_column"].unique(sorted=True):
            col = records["terrain_column"] == column
            add(f"terrain_column/{int(column)}", col[:, None, None])
            for index, phase in enumerate(("stable_air", "stable_support", "touchdown", "liftoff")):
                add(f"terrain_phase/{int(column)}/{phase}", col[:, None, None] & (records["phase"] == index))
            for row in records["terrain_row"].unique(sorted=True):
                add(f"terrain_cell/{int(row)},{int(column)}", (col & (records["terrain_row"] == row))[:, None, None])
        for name, cuts in thresholds.items():
            bins = torch.bucketize(records[name].contiguous(), cuts)
            for index in range(3):
                add(f"{name}/tertile_{index}", (bins == index)[:, None, :])
        for foot in range(2):
            add(f"foot/{foot}", (torch.arange(2) == foot)[None, None, :])
        add("terminal/yes", records["terminal"][:, None, None])
        add("terminal/no", ~records["terminal"][:, None, None])
        for name, durations in [
            ("air", records["previous_timers"][:, :2]),
            ("support", records["previous_timers"][:, 2:]),
        ]:
            add(f"previous_{name}_duration/ge_0.5s", (durations >= 0.5)[:, None, :])
        environments = {}
        for env in envs[torch.isin(envs, records["env_id"][rows].unique())]:
            mask = (records["env_id"] == env)[:, None, None].expand_as(labels)
            environments[str(int(env))] = event_metrics(probability, labels, mask)
        clocks = {}
        for name, sl in [("air", slice(0, 2)), ("support", slice(2, 4))]:
            error = records["timer_error"][:, sl]
            durations = records["previous_timers"][:, sl]
            squared = error.square()
            total_squared = squared[rows].sum()
            masks = {
                "all": rows[:, None].expand_as(error),
                "previous_ge_0.5s": rows[:, None] & (durations >= 0.5),
                "previous_lt_0.5s": rows[:, None] & (durations < 0.5),
                "any_contact_transition": rows[:, None] & (records["phase"] >= 2).any(1),
                "no_contact_transition": rows[:, None] & ~(records["phase"] >= 2).any(1),
            }
            clocks[name] = {}
            for group, mask in masks.items():
                values = squared[mask]
                clocks[name][group] = {
                    "foot_steps": int(mask.sum()),
                    "rmse_seconds": float(values.mean().sqrt()) if values.numel() else None,
                    "share_of_squared_error": float(values.sum() / total_squared) if total_squared > 0 else 0.0,
                }
            if "persistence_timer_error" in records:
                clocks[name]["persistence_rmse_seconds"] = float(
                    records["persistence_timer_error"][rows, sl].square().mean().sqrt()
                )
        result[pool] = {
            "transitions": int(rows.sum()),
            "overall": total,
            "persistence_baseline": event_metrics(persistence, labels, selected),
            "one_step_clocks": clocks,
            "groups": groups,
            "environments": environments,
        }
    return result


@torch.no_grad()
def stratify_replay_contacts(alg, replay, chunk_size=512):
    """Match the learner's state[i], action[i+1], final-state[i+1] indexing.

    Every available adjacent pair is evaluated once. Terminal targets are the
    saved pre-reset observations; command-expired starts are excluded as in fit.
    Assigned terrain cells are checked against saved origins, not inferred from
    the robot's current world position.
    """
    task = alg.imagination
    if not alg.fit_terminal_transitions:
        raise ValueError("exhaustive contact diagnostic requires saved terminal targets")
    filled, capacity = replay["filled"], replay["capacity_per_env"]
    if filled < 2:
        raise ValueError("contact stratification needs adjacent replay entries")
    start = replay["cursor"] if filled == capacity else 0
    order = (torch.arange(filled) + start) % capacity
    env_count = replay["num_envs"]
    time = order[:-1, None].expand(-1, env_count).flatten()
    following = order[1:, None].expand(-1, env_count).flatten()
    env = torch.arange(env_count)[None].expand(filled - 1, -1).flatten()
    origins = replay["buffer/env_origin"][time, env]
    terrain = task.env.scene.terrain
    grid = terrain.terrain_origins.detach().cpu()
    distances = (origins[:, None] - grid.reshape(-1, 3)[None]).abs().amax(-1)
    distance, cell = distances.min(-1)
    if float(distance.max()) > 1e-5:
        raise ValueError("saved environment origins do not match diagnostic terrain grid")
    columns = cell % grid.shape[1]
    expected_columns = terrain.terrain_types.detach().cpu()[env]
    torch.testing.assert_close(columns, expected_columns, rtol=0, atol=0)
    left = replay["buffer/command_time_left"][time, env].flatten()
    keep = left > 0
    time, following, env, cell = (v[keep] for v in (time, following, env, cell))
    labels_order = task.shape_materials.labels
    foot_shapes = [
        [i for i, label in enumerate(labels_order) if label.startswith(foot + ":")]
        for foot in ("left_ankle_roll_link", "right_ankle_roll_link")
    ]
    if any(not indices for indices in foot_shapes):
        raise ValueError("no realized shape parameters for a foot")
    records = {
        name: []
        for name in (
            "probability",
            "labels",
            "phase",
            "previous_timers",
            "timer_error",
            "persistence_timer_error",
            "foot_mu",
            "foot_restitution",
        )
    }
    model = alg.world_model.eval()
    original = {name: value.clone() for name, value in model.state_dict().items()}
    oracle_max_error = 0.0
    for offset in range(0, len(time), chunk_size):
        i, j, e = (
            time[offset : offset + chunk_size],
            following[offset : offset + chunk_size],
            env[offset : offset + chunk_size],
        )

        def pack(names, indices, suffix=""):
            return torch.cat([replay["buffer/" + n + suffix][indices, e].flatten(1) for n in names], -1).to(alg.device)

        state = pack(alg.state_names, i)
        target = pack(alg.state_names, j, "@final")
        context = pack(alg.context_names, i)
        action = replay["buffer/action"][j, e].to(alg.device)
        prediction = model(state, action, context)
        previous = task.parts(state)["contact_state"]
        actual = task.parts(target)["contact_state"]
        predicted = task.parts(prediction)["contact_state"]
        probability = predicted[:, 4:12].reshape(-1, 4, 2).flip(1)
        labels = actual[:, 4:12].reshape(-1, 4, 2).flip(1)
        oracle = advance_contact_history(previous[:, :4], actual[:, 4:12], task.dt, task.clock_substeps)
        oracle_max_error = max(oracle_max_error, float((oracle - actual[:, :4]).abs().max()))
        persistent_events = previous[:, 4:6].repeat(1, 4)
        persistent_timers = advance_contact_history(previous[:, :4], persistent_events, task.dt, task.clock_substeps)
        values = {
            "probability": probability,
            "labels": labels,
            "phase": contact_phases(previous[:, 4:6], labels),
            "previous_timers": previous[:, :4],
            "timer_error": predicted[:, :4] - actual[:, :4],
            "persistence_timer_error": persistent_timers - actual[:, :4],
        }
        parts = task.context_parts(context)
        for name, field in [("foot_mu", "shape_friction"), ("foot_restitution", "shape_restitution")]:
            values[name] = torch.stack([parts[field][:, ids].mean(-1) for ids in foot_shapes], -1)
        for name, value in values.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite contact stratum {name}")
            records[name].append(value.cpu())
        if offset % (chunk_size * 20) == 0:
            print("CONTACT_STRATA", offset, "/", len(time), flush=True)
    records = {name: torch.cat(values) for name, values in records.items()}
    records.update(
        env_id=env,
        replay_index=time,
        terrain_row=cell // grid.shape[1],
        terrain_column=cell % grid.shape[1],
        terminal=replay["buffer/done"][following, env].flatten().bool(),
    )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    result = summarize_records(records, alg.validation_env_ids)
    result.update(
        available_pairs=(filled - 1) * env_count,
        excluded_expired_command_starts=int((~keep).sum()),
        terrain_origin_max_error=float(distance.max()),
        world_model_and_normalizers_unchanged=True,
        oracle_clock_max_error_seconds=oracle_max_error,
        terrain_grouping="assigned origin cell, not current physical surface",
        material_grouping="per-foot mean of saved shape values; descriptive only; network still sees every shape",
        event_order="chronological 4 physics substeps x left,right; normal-force contact events only",
    )
    return result, records
