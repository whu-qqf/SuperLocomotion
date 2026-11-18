"""Separate event classification, physical-state drift and soft clock attenuation."""

import copy
import torch
import torch.nn.functional as F

from .contact_strata import contact_phases, event_metrics
from .contacts import advance_contact_history, advance_feet_timers


@torch.no_grad()
def single_event_clock_cost(initial_timers, labels, mask, dt, substeps):
    """Clock SSE from each isolated teacher-event flip, at valid control ends.

    All other events follow the logged labels, including future labels. These
    costs are offline supervision diagnostics, never deployable model inputs.
    Multiple-error costs are not additive because timer resets interact.
    """
    if (
        dt <= 0
        or substeps < 1
        or initial_timers.shape != (labels.shape[0], 4)
        or labels.shape[-1] != 2 * substeps
        or mask.shape != labels.shape[:2]
        or mask.dtype != torch.bool
        or (mask[:, 1:] & ~mask[:, :-1]).any()
    ):
        raise ValueError("require four initial timers and matching prefix-masked event histories")
    batch, horizon = mask.shape
    size = horizon * substeps * 2
    events = torch.where(mask[..., None], labels, torch.zeros_like(labels)).to(initial_timers)
    if not ((events == 0) | (events == 1)).all():
        raise ValueError("valid teacher contact events must be binary")
    scenarios = initial_timers[:, None].expand(-1, size, -1).clone()
    teacher = initial_timers.clone()
    costs = initial_timers.new_zeros(batch, size)
    for step in range(horizon):
        for sample in range(substeps):
            start = 2 * (substeps - 1 - sample)
            contact = events[:, step, start : start + 2]
            altered = contact[:, None].expand(-1, size, -1).clone()
            index = 2 * (step * substeps + sample)
            altered[:, index, 0] = 1 - contact[:, 0]
            altered[:, index + 1, 1] = 1 - contact[:, 1]
            teacher = advance_feet_timers(teacher, contact, dt / substeps)
            scenarios = advance_feet_timers(scenarios, altered, dt / substeps)
        error = (scenarios - teacher[:, None]).square().sum(-1)
        costs += torch.where(mask[:, step, None], error, torch.zeros_like(error))
    costs = costs.reshape(batch, horizon, substeps, 2).flip(-2).flatten(-2)
    return torch.where(mask[..., None], costs, torch.zeros_like(costs))


def expected_clock_squared_error(initial_timers, probabilities, target_timers, dt, substeps):
    """Exact marginal clock MSE for fixed, independent Bernoulli foot events.

    Offline diagnostic only: keep the probability sequence fixed, rather than
    rerunning the world model for every possible binary trajectory. Therefore
    this is not an expectation over a stochastic autoregressive world model.
    Unlike squaring expected clocks, retain clock variance. No hard threshold
    or straight-through derivative is used. Events are newest-first per step.
    """
    if (
        initial_timers.shape[-1] != 4
        or probabilities.shape[-1] != 2 * substeps
        or probabilities.shape[:2] != target_timers.shape[:2]
        or target_timers.shape[-1] != 4
        or dt <= 0
        or substeps < 1
    ):
        raise ValueError("require four timers, positive cadence and matched native event/target windows")
    mean, second = initial_timers, initial_timers.square()
    errors = []
    delta = dt / substeps
    for step in range(probabilities.shape[1]):
        for sample in reversed(range(substeps)):
            p = probabilities[:, step, 2 * sample : 2 * sample + 2]
            keep = torch.cat((1 - p, p), -1)
            second = (second + 2 * delta * mean + delta**2) * keep
            mean = (mean + delta) * keep
        target = target_timers[:, step]
        errors.append(second - 2 * target * mean + target.square())
    return torch.stack(errors, 1)


def probe_contact_directions(model, task, batch, displacements=(1e-4, 1e-3, 1e-2)):
    """Finite perturbations of disposable copies, never cumulative training.

    Displacements are absolute network-parameter L2 distances, identical for
    BCE and clock directions. Probe both signs from the same original weights.
    The task/terrain is shared read-only; no optimizer or source .grad writes.
    """
    if not displacements or any(step <= 0 for step in displacements):
        raise ValueError("positive parameter displacements are required")
    candidate = copy.deepcopy(model, {id(task): task}).eval()
    parameters = tuple(candidate.net.parameters())
    base = tuple(p.detach().clone() for p in parameters)
    initial, actions, context, targets, mask = batch
    losses = candidate.rollout_loss(initial, actions, context, targets, mask=mask, update_normalizer=False)
    objectives = {
        "weighted_clock": model.contact_clock_loss_weight * losses["loss_contact_clock"],
        "contact_bce": losses["loss_contact_binary"],
    }
    directions = {}
    for name, loss in objectives.items():
        gradients = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        directions[name] = tuple(
            torch.zeros_like(p) if g is None else g.detach() for p, g in zip(parameters, gradients, strict=True)
        )

    @torch.no_grad()
    def measure():
        measured = candidate.rollout_loss(initial, actions, context, targets, mask=mask, update_normalizer=False)
        p = task.parts(candidate.rollout(initial, actions, context))["contact_state"][mask]
        t = task.parts(targets)["contact_state"][mask]
        probability, labels = p[..., 4:12], t[..., 4:12]
        positive = labels.bool()
        error = F.binary_cross_entropy(probability.clamp(1e-7, 1 - 1e-7), labels, reduction="none")
        decision = probability >= 0.5
        metrics = {
            "total_loss": float(measured["loss"]),
            "weighted_clock": float(model.contact_clock_loss_weight * measured["loss_contact_clock"]),
            "contact_bce": float(measured["loss_contact_binary"]),
            "support_foot_bce": float(error[positive].mean()) if positive.any() else None,
            "air_foot_bce": float(error[~positive].mean()) if (~positive).any() else None,
            "stance_rmse_s": float((p[..., 2:4] - t[..., 2:4]).double().square().mean().sqrt()),
            "air_rmse_s": float((p[..., :2] - t[..., :2]).double().square().mean().sqrt()),
            "false_negative_events": int((~decision & positive).sum()),
            "false_positive_events": int((decision & ~positive).sum()),
        }
        return metrics, decision

    baseline, original_decision = measure()
    rows = []
    for name, gradients in directions.items():
        norm = sum(g.double().square().sum() for g in gradients).sqrt()
        if norm == 0:
            raise ValueError(f"cannot probe a zero gradient direction: {name}")
        for step in displacements:
            for sign, multiplier in (("descent", -1), ("ascent", 1)):
                with torch.no_grad():
                    for p, value, gradient in zip(parameters, base, gradients, strict=True):
                        p.copy_(value + multiplier * step * gradient / norm.to(gradient.dtype))
                measured, decision = measure()
                distance = sum(
                    (p.detach() - value).double().square().sum() for p, value in zip(parameters, base)
                ).sqrt()
                rows.append(
                    {
                        "direction": name,
                        "sign": sign,
                        "requested_l2_displacement": step,
                        "actual_l2_displacement": float(distance),
                        "gradient_norm": float(norm),
                        "linear_prediction_of_own_loss_change": multiplier * step * float(norm),
                        "actual_own_loss_change": measured[name] - baseline[name],
                        "flipped_foot_events": int((decision != original_decision).sum()),
                        "metrics": measured,
                    }
                )
    return {
        "baseline": baseline,
        "perturbations": rows,
        "source_mode": model.training,
        "scope": "Finite read-only-copy probes; absolute L2 displacement, no optimizer, no checkpoint selection.",
    }


def audit_contact_gradients(model, task, batch):
    """Measure existing objective gradients without stepping or writing .grad.

    Regression includes physical state, rewards and illegal torso contact.
    Clock weight follows the model configuration; conditional foot BCEs are diagnostics
    only. Negative dot product means those local descent directions conflict.
    """
    initial, actions, context, targets, mask = batch
    parameters = tuple(model.net.parameters())
    original = {name: value.clone() for name, value in model.state_dict().items()}
    mode = model.training
    model.eval()
    try:
        losses = model.rollout_loss(initial, actions, context, targets, mask=mask, update_normalizer=False)
        objectives = {
            "total": losses["loss"],
            "contact_bce": losses["loss_contact_binary"],
            "weighted_clock": model.contact_clock_loss_weight * losses["loss_contact_clock"],
        }
        objectives["non_contact_regression"] = (
            objectives["total"] - objectives["contact_bce"] - objectives["weighted_clock"]
        )
        predicted = model.rollout(initial, actions, context)
        probability = task.parts(predicted)["contact_state"][..., 4:][mask]
        labels = task.parts(targets)["contact_state"][..., 4:][mask]
        foot_p, foot_l = probability[..., :8], labels[..., :8]
        error = F.binary_cross_entropy(foot_p.clamp(1e-7, 1 - 1e-7), foot_l, reduction="none")
        for name, selection in (("support_foot_bce", foot_l.bool()), ("air_foot_bce", ~foot_l.bool())):
            if selection.any():
                objectives[name] = error[selection].mean()
        gradients = {}
        for name, objective in objectives.items():
            values = torch.autograd.grad(objective, parameters, retain_graph=True, allow_unused=True)
            gradients[name] = torch.cat(
                [(torch.zeros_like(p) if g is None else g).flatten() for p, g in zip(parameters, values, strict=True)]
            ).detach()
        norms = {name: float(g.double().norm()) for name, g in gradients.items()}
        pairs = {}
        names = list(gradients)
        for i, first in enumerate(names):
            for second in names[i + 1 :]:
                dot = float((gradients[first].double() * gradients[second].double()).sum())
                denominator = norms[first] * norms[second]
                pairs[first + "/" + second] = {
                    "cosine": dot / denominator if denominator else None,
                    "dot": dot,
                    "second_loss_derivative_along_negative_first_gradient": -dot,
                }
        summed = gradients["non_contact_regression"] + gradients["contact_bce"] + gradients["weighted_clock"]
        result = {
            "valid_steps": int(mask.sum()),
            "losses": {name: float(loss.detach()) for name, loss in objectives.items()},
            "gradient_norms": norms,
            "pairs": pairs,
            "component_gradient_sum_max_error": float((summed - gradients["total"]).abs().max()),
            "saturation": {},
        }
        for name, p, labels in (("all_events", probability, labels), ("foot_events", foot_p, foot_l)):
            saturated = (p <= 1e-7) | (p >= 1 - 1e-7)
            wrong = (p >= 0.5) != labels.bool()
            result["saturation"][name] = {
                "labels": p.numel(),
                "wrong": int(wrong.sum()),
                "clamped_or_saturated": int(saturated.sum()),
                "wrong_and_clamped_or_saturated": int((wrong & saturated).sum()),
            }
    finally:
        model.train(mode)
    for name, value in model.state_dict().items():
        if not torch.equal(original[name], value):
            raise RuntimeError(f"gradient audit changed model state: {name}")
    return result


def persistent_contacts(initial, horizon, dt, substeps):
    """Hold only initial latest events; transport clocks without future labels."""
    if substeps != 4 or initial.shape[-1] != 21 or horizon < 1:
        raise ValueError("requires positive horizon and native four-substep contact encoding")
    state = initial.clone()
    state[..., 4:12] = initial[..., 4:6].repeat(1, substeps)
    state[..., 12:] = initial[..., 12:15].repeat(1, 3)
    predictions = []
    for _ in range(horizon):
        state = state.clone()
        state[..., :4] = advance_contact_history(state[..., :4], state[..., 4:12], dt, substeps)
        predictions.append(state)
    return torch.stack(predictions, 1)


def contact_clock_records(initial, predicted, truth, mask, dt, substeps):
    """Separate event errors from clock transport on reset-safe window prefixes.

    All arguments contain contact_state only. Oracle/true-previous recurrences
    are diagnostics; they never replace inputs to the evaluated model.
    """
    if substeps != 4 or initial.shape[-1] != 21:
        raise ValueError("diagnostic requires the native four-substep, 21-field contact encoding")
    previous = torch.cat((initial[:, None], truth[:, :-1]), 1)
    labels = truth[..., 4:12].reshape(*mask.shape, substeps, 2).flip(-2)
    probability = predicted[..., 4:12].reshape(*mask.shape, substeps, 2).flip(-2)
    phases = contact_phases(previous[..., 4:6].flatten(0, 1), labels.flatten(0, 1)).reshape_as(labels)
    oracle, hard = initial[..., :4], initial[..., :4]
    oracle_history, hard_history, oracle_local = [], [], []
    for step in range(mask.shape[1]):
        oracle = advance_contact_history(oracle, truth[:, step, 4:12], dt, substeps)
        hard = advance_contact_history(hard, (predicted[:, step, 4:12] >= 0.5).float(), dt, substeps)
        oracle_history.append(oracle)
        hard_history.append(hard)
        oracle_local.append(advance_contact_history(previous[:, step, :4], truth[:, step, 4:12], dt, substeps))
    records = {
        "probability": probability,
        "labels": labels,
        "phase": phases,
        "body_probability": predicted[..., 12:],
        "body_labels": truth[..., 12:],
        "clock_error": predicted[..., :4] - truth[..., :4],
        "oracle_error": torch.stack(oracle_history, 1) - truth[..., :4],
        "oracle_local_error": torch.stack(oracle_local, 1) - truth[..., :4],
        "hard_transport_error": torch.stack(hard_history, 1) - predicted[..., :4],
        "previous_timers": previous[..., :4],
        "target_timers": truth[..., :4],
        "any_wrong_event_in_prefix": ((probability >= 0.5) != labels.bool()).any(-2).cumsum(1) > 0,
    }
    return {name: value[mask].detach().cpu() for name, value in records.items()}


def summarize_clock_records(records):
    """Count physics foot events and locate clock SSE on long support histories."""
    probability, labels = records["probability"], records["labels"]
    selected = torch.ones_like(labels, dtype=torch.bool)
    result = {"valid_steps": len(labels), "foot_events": event_metrics(probability, labels, selected)}
    result["foot_phases"] = {
        name: event_metrics(probability, labels, records["phase"] == index)
        for index, name in enumerate(("stable_air", "stable_support", "touchdown", "liftoff"))
    }
    body_p, body_l = records["body_probability"], records["body_labels"]
    result["body_force_events"] = event_metrics(body_p, body_l, torch.ones_like(body_l, dtype=torch.bool))
    for name in ("oracle_error", "oracle_local_error", "hard_transport_error"):
        error = records[name]
        result[name] = {"max_abs_s": float(error.abs().max()), "rmse_s": float(error.square().mean().sqrt())}
    result["clocks"] = {}
    for name, sl in (("air", slice(0, 2)), ("stance", slice(2, 4))):
        error = records["clock_error"][:, sl].double()
        previous = records["previous_timers"][:, sl]
        square = error.square()
        total = float(square.sum())
        groups = {
            "all": torch.ones_like(error, dtype=torch.bool),
            "previous_ge_0.5s": previous >= 0.5,
            "previous_lt_0.5s": previous < 0.5,
            "no_wrong_foot_event_in_prefix": ~records["any_wrong_event_in_prefix"],
            "some_wrong_foot_event_in_prefix": records["any_wrong_event_in_prefix"],
        }
        if name == "stance":
            support = labels.bool().all(-2)
            missed = (~(probability >= 0.5)).any(-2)
            groups["long_true_support_with_predicted_liftoff"] = (previous >= 0.5) & support & missed
        result["clocks"][name] = {}
        for group, mask in groups.items():
            values = error[mask]
            result["clocks"][name][group] = {
                "foot_steps": values.numel(),
                "rmse_s": float(values.square().mean().sqrt()) if values.numel() else None,
                "bias_s": float(values.mean()) if values.numel() else None,
                "share_of_sse": float(values.square().sum()) / total if total else 0.0,
            }
    return result


def summarize_contact_groups(records):
    """Describe phase/terrain/movement error shares without calling them success."""
    probability, labels = records["probability"], records["labels"]
    groups = {"all": torch.ones_like(labels, dtype=torch.bool)}
    for name, select in (("moving", records["moving"]), ("other", ~records["moving"])):
        groups[name] = select[:, None, None].expand_as(labels)
    for column in range(4):
        select = (records["terrain_column"] == column)[:, None, None]
        groups[f"terrain/{column}"] = select.expand_as(labels)
        groups[f"moving_terrain/{column}"] = select & groups["moving"]
    for index, phase in enumerate(("stable_air", "stable_support", "touchdown", "liftoff")):
        select = records["phase"] == index
        groups[f"phase/{phase}"] = select
        groups[f"moving_phase/{phase}"] = select & groups["moving"]
    total = event_metrics(probability, labels, groups["all"])
    result = {}
    for name, select in groups.items():
        metric = event_metrics(probability, labels, select)
        if metric["events"]:
            metric["share_of_all_false_positives"] = (
                metric["false_positives"] / total["false_positives"] if total["false_positives"] else 0.0
            )
            metric["share_of_all_false_negatives"] = (
                metric["false_negatives"] / total["false_negatives"] if total["false_negatives"] else 0.0
            )
        result[name] = metric
    return result


def fit_one_step_contacts(model, task, training, validation, steps=400):
    """Finite event-only fitting diagnostic; never mutate the production model.

    Only window starts are used, so every prediction has a logged previous state
    and its matching command context. Normalizers stay frozen. The small subset
    measures memorization; the larger fixed pool measures fit and generalization.
    These diagnostic models are unsuitable for policy optimization.
    """

    def starts(batch):
        state, actions, context, targets, mask = batch
        valid = mask[:, 0]
        return state[valid], actions[valid, 0], context[valid], targets[valid, 0]

    def events(candidate, batch):
        state, action, context, target = batch
        prediction = candidate(state, action, context)
        return task.parts(prediction)["contact_state"][..., 4:], task.parts(target)["contact_state"][..., 4:]

    @torch.no_grad()
    def measure(candidate, batch):
        probability, labels = events(candidate, batch)
        return {
            "bce": float(F.binary_cross_entropy(probability.clamp(1e-7, 1 - 1e-7), labels)),
            "event_rmse": float((probability - labels).square().mean().sqrt()),
            "event_error_rate": float(((probability >= 0.5) != labels.bool()).float().mean()),
            "clock_event_error_rate": float(((probability[:, :8] >= 0.5) != labels[:, :8].bool()).float().mean()),
            "positive_rate": float(labels.mean()),
        }

    train, valid = starts(training), starts(validation)
    if len(train[0]) < 64 or len(valid[0]) == 0:
        raise ValueError("contact fitting requires at least 64 valid training starts and nonempty validation")
    result = {"steps": steps, "learning_rate": 3e-4, "normalizers_frozen": True, "fits": {}}
    for name, count in [("tiny_64", 64), ("training_pool", len(train[0]))]:
        candidate = copy.deepcopy(model, {id(task): task}).eval()
        subset = tuple(value[:count] for value in train)
        optimizer = torch.optim.Adam(candidate.net.parameters(), lr=3e-4)
        generator = torch.Generator(device=train[0].device).manual_seed(44)
        history = []
        for step in range(steps + 1):
            if step % 100 == 0 or step == steps:
                record = {"step": step, "train": measure(candidate, subset), "validation": measure(candidate, valid)}
                history.append(record)
                print("CONTACT_ONE_STEP", name, record, flush=True)
            if step == steps:
                break
            ids = torch.randperm(count, device=train[0].device, generator=generator)[:256]
            probability, labels = events(candidate, tuple(value[ids] for value in subset))
            loss = F.binary_cross_entropy(probability.clamp(1e-7, 1 - 1e-7), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite one-step contact loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(candidate.net.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        result["fits"][name] = {
            "training_starts": count,
            "validation_starts": len(valid[0]),
            "history": history,
            "validation_rollout": diagnose_contacts(candidate, task, validation),
        }
    return result


@torch.no_grad()
def diagnose_contacts(model, task, batch):
    state, actions, context, targets, mask = batch
    truth = task.parts(targets)["contact_state"]
    initial = task.parts(state)["contact_state"]
    ar = model.rollout(state, actions, context)
    previous = torch.cat((state[:, None], targets[:, :-1]), 1)
    tf = torch.stack([model(previous[:, i], actions[:, i], context) for i in range(actions.shape[1])], 1)
    result = {}
    for name, predicted in [("autoregressive", ar), ("true_previous_state", tf)]:
        contacts = task.parts(predicted)["contact_state"]
        events, labels = contacts[..., 4:], truth[..., 4:]
        error = (events - labels)[mask]
        result[name] = {
            "event_rmse": float(error.square().mean().sqrt()),
            "event_error_rate": float(((events[mask] >= 0.5) != labels[mask].bool()).float().mean()),
            "clock_event_error_rate": float(
                ((events[..., :8][mask] >= 0.5) != labels[..., :8][mask].bool()).float().mean()
            ),
        }
        for transport, sequence in [
            ("soft", events),
            ("hard_diagnostic_only", (events >= 0.5).float()),
            ("oracle", labels),
        ]:
            timers = initial[..., :4]
            history = []
            for i in range(actions.shape[1]):
                timers = advance_contact_history(timers, sequence[:, i], task.dt, task.clock_substeps)
                history.append(timers)
            error = (torch.stack(history, 1) - truth[..., :4])[mask]
            result[name][transport] = {
                "air_rmse": float(error[..., :2].square().mean().sqrt()),
                "stance_rmse": float(error[..., 2:].square().mean().sqrt()),
                "stance_bias": float(error[..., 2:].mean()),
                "max_error": float(error.abs().max()),
            }
    result["valid_steps"] = int(mask.sum())
    result["stance_duration_quantiles"] = torch.quantile(
        truth[..., 2:4][mask].flatten(), torch.tensor([0.5, 0.9, 0.99], device=state.device)
    ).tolist()
    return result


def audit_material_context(task):
    """Read realized foot materials to quantify information lost by the context mean."""
    import warp as wp

    robot = task.robot
    model = task.env.sim.backend_manager.get_model()
    mu = wp.to_torch(robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
    restitution = wp.to_torch(robot._root_view.get_attribute("shape_material_restitution", model)[:, 0])
    selected = []
    for body_id in robot.map_body_ids_to_backend(task.foot_ids):
        selected.extend(robot._root_view.body_shapes[body_id])
    foot_mu, foot_restitution = mu[:, selected], restitution[:, selected]
    return {
        "foot_collision_shapes": len(selected),
        "mean_friction_context_std_across_envs": float(mu.mean(-1).std()),
        "foot_friction_std_all_samples": float(foot_mu.std()),
        "foot_friction_rmse_from_context_mean": float((foot_mu - mu.mean(-1, keepdim=True)).square().mean().sqrt()),
        "foot_restitution_min": float(foot_restitution.min()),
        "foot_restitution_max": float(foot_restitution.max()),
        "note": (
            "full per-shape friction and restitution are included in the v5 context"
            if "shape_friction" in task.context_format
            else "friction is reduced to one mean; randomized restitution is absent from current context"
        ),
    }
