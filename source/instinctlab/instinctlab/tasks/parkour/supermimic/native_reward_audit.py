"""Frozen-model reward derivatives, independent of actor/critic fitting.

Finite differences test the model's reward equations, not real physics. Exact
terrain heights have an intentional straight-through coordinate derivative.
"""

import torch

from instinctlab_supermimic.physical import unflatten_state

from .native_rewards import tgi_core_reward


def directional_reward_audit(objective, actions, epsilons=(1e-3, 1e-2)):
    """Per-term gradients and per-environment directional finite differences.

    Probe the total gradient, a fixed random direction and the first action.
    All directions have unit norm independently for each environment.
    """
    controls = actions.detach().clone().requires_grad_(True)
    terms = objective(controls)
    names = list(terms)
    gradients = []
    for value in terms.values():
        gradient = (
            torch.autograd.grad(value.sum(), controls, retain_graph=True, allow_unused=True)[0]
            if value.requires_grad
            else None
        )
        gradients.append(torch.zeros_like(controls) if gradient is None else gradient)
    gradients = torch.stack(gradients, 1).detach()
    if not torch.isfinite(gradients).all():
        raise FloatingPointError("Nonfinite reward action gradient")
    total = gradients[:, names.index("total")]
    generator = torch.Generator(device=actions.device).manual_seed(724)
    random = torch.randn(actions.shape, device=actions.device, generator=generator)
    first = torch.zeros_like(random)
    first[:, 0] = random[:, 0]
    directions = torch.stack((total, random, first))
    directions = directions / directions.flatten(2).norm(dim=-1).clamp_min(1e-12)[..., None, None]
    automatic = torch.einsum("btha,pbha->pbt", gradients, directions)
    differences = []
    with torch.no_grad():
        for direction in directions:
            rows = []
            for epsilon in epsilons:
                plus, minus = objective(actions + epsilon * direction), objective(actions - epsilon * direction)
                rows.append(torch.stack([(plus[n] - minus[n]) / (2 * epsilon) for n in names], -1))
            differences.append(torch.stack(rows))
    differences = torch.stack(differences)
    if not torch.isfinite(differences).all():
        raise FloatingPointError("Nonfinite reward finite difference")
    summary = {}
    for index, name in enumerate(names):
        gradient = gradients[:, index]
        comparisons = []
        for j, epsilon in enumerate(epsilons):
            ad, fd = automatic[..., index], differences[:, j, :, index]
            significant = (ad.abs() + fd.abs()) > 1e-5
            comparisons.append(
                dict(
                    epsilon=epsilon,
                    automatic_rms=float(ad.square().mean().sqrt()),
                    finite_difference_rms=float(fd.square().mean().sqrt()),
                    absolute_error_rms=float((ad - fd).square().mean().sqrt()),
                    significant_probes=int(significant.sum()),
                    sign_agreement=(float(((ad * fd) > 0)[significant].float().mean()) if significant.any() else None),
                )
            )
        summary[name] = dict(
            mean_value=float(terms[name].detach().mean()),
            gradient_rms=float(gradient.square().mean().sqrt()),
            first_action_gradient_rms=float(gradient[:, 0].square().mean().sqrt()),
            nonzero_environment_fraction=float((gradient.flatten(1).norm(dim=-1) > 1e-10).float().mean()),
            finite_difference=comparisons,
        )
    return dict(
        terms=names,
        values=torch.stack([value.detach() for value in terms.values()], -1),
        summary=summary,
        automatic=automatic,
        finite_difference=differences,
        gradients=gradients,
        directions=directions,
    )


def audit_native_rewards(task, initial, actions, physical_context, task_context, command, gamma=0.99):
    """Open-loop controls with production F, reward, terrain and termination.

    Actor feedback, critic bootstrap, likelihood ratio and touchdown cuts are
    excluded to isolate reward derivatives. Raw reward terms are discounted
    without survival; survival_total separately includes soft terminations.
    """

    def objective(controls):
        state = unflatten_state(initial)
        context, previous = physical_context, physical_context[:, :29]
        totals = {}
        survival = controls.new_ones(len(controls))
        survival_total = controls.new_zeros(len(controls))
        supported = task.physics.scan_with_validity(state)[1]
        for step in range(controls.shape[1]):
            action = controls[:, step].clamp(-20, 20)
            predicted, logits, _, _ = task.physics.model.step_with_aux(
                state, action, context, task.physics.model_scan(state)
            )
            supported = supported & task.physics.scan_with_validity(predicted)[1]
            state = tuple(
                torch.where(supported.reshape(-1, *([1] * (new.ndim - 1))), new, old)
                for old, new in zip(state, predicted)
            )
            contacts = logits.sigmoid()
            _, terms = tgi_core_reward(task, state, action, previous, command["velocity"], task_context)
            extra, _ = task.locomotion_readout(state, task_context, contacts[:, -2:], command["velocity"])
            terms.update(extra)
            terms = {name: torch.where(supported, value, torch.zeros_like(value)) for name, value in terms.items()}
            reward = sum(terms.values())
            termination = task.soft_termination(state, command, contacts)
            termination = torch.where(supported, termination, torch.ones_like(termination))
            for name, value in terms.items():
                totals[name] = totals.get(name, 0) + gamma**step * value
            survival_total = survival_total + gamma**step * survival * reward
            survival = survival * (1 - termination)
            previous = action
            context = torch.cat((action, context[:, 29:]), -1)
        totals["total"] = sum(totals.values())
        totals["survival_total"] = survival_total
        totals["final_survival"] = survival
        totals["final_terrain_supported"] = supported.float()
        return totals

    parameters = list(task.physics.model.parameters())
    trainable = [p.requires_grad for p in parameters]
    before = {name: value.clone() for name, value in task.physics.model.state_dict().items()}
    try:
        task.physics.model.requires_grad_(False)
        result = directional_reward_audit(objective, actions)
    finally:
        for parameter, enabled in zip(parameters, trainable):
            parameter.requires_grad_(enabled)
    for name, value in task.physics.model.state_dict().items():
        if not torch.equal(value, before[name]):
            raise RuntimeError(f"Reward audit changed world model: {name}")
    result["scope"] = (
        "Frozen model, fixed open-loop actions; excludes actor feedback, critic and event cuts. "
        "Finite differences are of model rewards, not real simulator rewards. "
        "Nearest terrain heights use bilinear surrogate gradients; disagreement there can be intentional."
    )
    return result
