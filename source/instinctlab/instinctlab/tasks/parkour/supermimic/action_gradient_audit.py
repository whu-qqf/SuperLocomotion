"""Compare model action derivatives to restored simulator perturbation probes.

These are measurements, not a training gate with invented acceptance thresholds.
Unresolved differences below the unperturbed replay floor are inconclusive.
"""

import math
import torch


def first_action_derivative(predict, actions, direction):
    """Differentiate N,H,K physical readouts along the first action only.

    ``predict`` must operate independently per environment and return physical
    readouts, without adding a direct action penalty to the measurement.
    Parameter gradients and optimizer state are not changed.
    """
    if actions.ndim != 3 or direction.shape != actions[:, 0].shape:
        raise ValueError("Expected N,H,A actions and N,A perturbation direction.")
    with torch.enable_grad():
        inputs = actions.detach().clone().requires_grad_(True)
        outputs = predict(inputs)
        if outputs.ndim != 3 or outputs.shape[:2] != actions.shape[:2]:
            raise ValueError("Physical readouts must have shape N,H,K.")
        derivatives = []
        for step in range(outputs.shape[1]):
            components = []
            for component in range(outputs.shape[2]):
                gradient = torch.autograd.grad(outputs[:, step, component].sum(), inputs, retain_graph=True)[0]
                components.append((gradient[:, 0] * direction).sum(-1))
            derivatives.append(torch.stack(components, -1))
        result = torch.stack(derivatives, 1).detach()
    if not torch.isfinite(result).all():
        raise FloatingPointError("Nonfinite model action derivative.")
    return result


def compare_probe_derivatives(
    probes, readout, model_derivative, epsilons=(0.05, 0.15), noise_ratio=3.0, component_mask=None
):
    """Read the branch format produced by collect_newton_action_probes.

    ``readout(states)`` returns N,H,K in the same units as model_derivative.
    All zero repeats contribute to the replay floor. A terminal state itself
    remains a valid physical target; subsequent states are excluded.
    """
    if (
        not epsilons
        or any(not math.isfinite(e) or e <= 0 for e in epsilons)
        or not math.isfinite(noise_ratio)
        or noise_ratio <= 0
    ):
        raise ValueError("Positive finite perturbation sizes and noise ratio are required.")
    restore_errors = probes["restore_max_state_errors"]
    if not torch.isfinite(restore_errors).all() or restore_errors.abs().max() != 0:
        raise ValueError("Probe starting states were not restored exactly.")
    outcomes = probes["outcomes"]
    observed = {name: readout(branch["states"]).detach().cpu() for name, branch in outcomes.items()}
    baseline = observed["zero_a"]
    model = model_derivative.detach().cpu()
    if model.shape != baseline.shape or baseline.ndim != 3:
        raise ValueError("Model and simulator readouts must have matching N,H,K shapes.")
    if not all(torch.isfinite(value).all() for value in [model, *observed.values()]):
        raise FloatingPointError("Nonfinite derivative or simulator readout.")
    zero_names = [name for name in outcomes if name.startswith("zero_")]
    if len(zero_names) < 2:
        raise ValueError("At least two unperturbed simulator branches are required.")
    repeats = torch.stack([observed[name] for name in zero_names])
    noise = repeats.amax(0) - repeats.amin(0)
    # No branch is allowed to continue past termination, including zero repeats.
    done = torch.stack([branch["done"].cpu().bool() for branch in outcomes.values()]).any(0)
    valid = torch.cat((torch.ones_like(done[:, :1]), (~done[:, :-1]).cumprod(1).bool()), 1)[..., None]
    if component_mask is not None:
        if component_mask.shape != baseline.shape or component_mask.dtype != torch.bool:
            raise ValueError("Derivative strata require a boolean mask matching N,H,K readouts.")
        valid = valid & component_mask.cpu()
    reports = []
    for epsilon in epsilons:
        fd = (observed[f"plus_{epsilon}"] - observed[f"minus_{epsilon}"]) / (2 * epsilon)
        noise_derivative = noise / epsilon
        resolved = valid & (fd.abs() > noise_ratio * noise_derivative)
        actual, predicted = fd[resolved], model[resolved]
        norm = actual.norm()
        denominator = norm * predicted.norm()
        reports.append(
            dict(
                epsilon=epsilon,
                valid_components=int(valid.expand_as(fd).sum()),
                resolved_components=int(resolved.sum()),
                status="resolved" if resolved.any() else "inconclusive",
                repeat_noise_max=float(noise.max()),
                derivative_noise_max=float(noise_derivative.max()),
                sign_agreement=float((actual.sign() == predicted.sign()).float().mean()) if len(actual) else None,
                cosine=float(torch.dot(actual, predicted) / denominator) if denominator > 0 else None,
                relative_error=float((predicted - actual).norm() / norm) if norm > 0 else None,
            )
        )
    return dict(
        perturbations=reports,
        model_updates=0,
        policy_updates=0,
        scope="Local directional physical derivatives only; not a certificate of controller or rollout quality.",
    )
