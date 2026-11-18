"""Bounded dynamics-only fits on archived windows with location holdout."""

import copy
import torch

from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_plane_cfg import TRACKED_BODIES


@torch.no_grad()
def calibrate_native_inputs(model, windows, chunk_size=128):
    """Rebase training-only input moments on a disposable, equivalent model.

    Compensate the first affine layer for the changed input coordinates. Output
    scales and objective weights stay fixed. No optimizer state is converted:
    this copy is for offline diagnostics, not a complete learner resume.
    """
    training = windows["training_indices"]
    validation = windows["validation_indices"]
    if chunk_size < 1 or not len(training) or training.unique().numel() != len(training):
        raise ValueError("require positive chunk size and unique nonempty training pool")
    if torch.isin(windows["env_id"][training], windows["env_id"][validation]).any():
        raise ValueError("training and validation environments overlap")
    mask = windows["mask"]
    if mask.dtype != torch.bool or mask.shape != windows["actions"].shape[:2]:
        raise ValueError("native mask must be boolean with action batch/horizon shape")
    if not mask[training, 0].all() or (mask[training, 1:] & ~mask[training, :-1]).any():
        raise ValueError("training windows must contain nonempty valid prefixes")
    normalizer = model.input_normalizer
    count = 0
    mean = torch.zeros_like(normalizer._mean.squeeze(0), dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    for ids in training.split(chunk_size):
        state, actions, context, targets = (windows[k][ids] for k in ("state0", "actions", "context", "targets"))
        previous = torch.cat((state[:, None], targets[:, :-1]), 1)
        context = context[:, None].expand(-1, actions.shape[1], -1)
        x = model._concat(previous, actions, context)[mask[ids]].double()
        if not torch.isfinite(x).all():
            raise ValueError("non-finite training input")
        n = len(x)
        delta = x.mean(0) - mean
        m2 += x.var(0, unbiased=False) * n + delta.square() * (count * n / (count + n))
        mean += delta * (n / (count + n))
        count += n
    candidate = copy.deepcopy(model, {id(model.task): model.task}).eval()
    first = candidate.net.model[0]
    if not isinstance(first, torch.nn.Linear) or first.bias is None:
        raise ValueError("input rebasing requires the native MLP's first biased Linear layer")
    new = candidate.input_normalizer
    new._mean.copy_(mean[None])
    new._var.copy_((m2 / count)[None])
    new._std.copy_(new._var.sqrt())
    new.count.fill_(count)
    new._inc_count.zero_()
    new._inc_mean.zero_()
    new._inc_var.fill_(1)
    old_scale = (normalizer._std + normalizer.eps).squeeze(0).double()
    new_scale = (new._std + new.eps).squeeze(0).double()
    shift = (new._mean - normalizer._mean).squeeze(0).double() / old_scale
    weight = first.weight.double()
    first.bias.copy_(first.bias.double() + weight @ shift)
    first.weight.copy_(weight * (new_scale / old_scale)[None])
    return candidate, {"training_steps": count, "model_updates": 0, "optimizer_state_converted": False}


def fit_native_windows(
    model,
    windows,
    updates,
    batch_size=128,
    learning_rate=2e-4,
    seed=20261112,
    omit_clock_loss=False,
    contact_head_only=False,
):
    """Fit a dynamics copy on the supplied native training pool, for at most 64 updates.

    Preserve archived reset/command masks and inherited normalization statistics.
    The static task/terrain is shared read-only; network and statistics are copied.
    Validation labels are never indexed here. Evaluation belongs to the caller.
    """
    if not 1 <= updates <= 64 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("require 1..64 updates, positive batch_size and learning_rate")
    if contact_head_only and omit_clock_loss:
        raise ValueError("contact-only BCE does not use a clock objective to omit")
    training, validation = windows["training_indices"], windows["validation_indices"]
    size = len(windows["state0"])
    for indices in (training, validation):
        if indices.ndim != 1 or not len(indices) or indices.dtype != torch.long:
            raise ValueError("pools must be nonempty one-dimensional integer indices")
        if indices.min() < 0 or indices.max() >= size or indices.unique().numel() != len(indices):
            raise ValueError("pool contains duplicate or out-of-range indices")
    if torch.isin(windows["env_id"][training], windows["env_id"][validation]).any():
        raise ValueError("training and validation environment pools overlap")
    if windows["mask"].dtype != torch.bool or windows["mask"].shape != windows["actions"].shape[:2]:
        raise ValueError("native mask must be boolean with action batch/horizon shape")
    mask = windows["mask"][training]
    if not mask[:, 0].all() or (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError("each training window must contain a nonempty valid prefix")
    fitted = copy.deepcopy(model, {id(model.task): model.task})
    statistics = {name: value.clone() for name, value in fitted.state_dict().items() if "normalizer" in name}
    original = {name: value.clone() for name, value in fitted.state_dict().items()} if contact_head_only else {}
    requires_grad = {name: p.requires_grad for name, p in fitted.net.named_parameters()}
    if contact_head_only:
        head = fitted.net.model[-1]
        if (
            not fitted.parameterization.contact_events
            or fitted.parameterization.event_size != 17
            or not isinstance(head, torch.nn.Linear)
            or head.bias is None
            or head.out_features != fitted.parameterization.target_size
        ):
            raise ValueError("contact-only fit requires the native final 17 event output rows")
        for p in fitted.net.parameters():
            p.requires_grad_(False)
        head.weight.requires_grad_(True)
        head.bias.requires_grad_(True)
    parameters = [p for p in fitted.net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    batches, losses = [], []
    fitted.train()
    for _ in range(updates):
        positions = torch.randint(len(training), (batch_size,), generator=generator).to(training.device)
        batch = training[positions]
        optimizer.zero_grad(set_to_none=True)
        objectives = fitted.rollout_loss(
            windows["state0"][batch],
            windows["actions"][batch],
            windows["context"][batch],
            windows["targets"][batch],
            mask=windows["mask"][batch],
            update_normalizer=False,
        )
        loss = objectives["loss_contact_binary"] if contact_head_only else objectives["loss"]
        if omit_clock_loss:
            # Offline ablation only. Preserve the production event transport,
            # physical/reward regression and BCE, removing just this objective.
            loss = loss - fitted.contact_clock_loss_weight * objectives["loss_contact_clock"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite native short-fit loss")
        loss.backward()
        if contact_head_only:
            # Fresh Adam, no decay: zero gradients leave these rows exact,
            # including their initially zero optimizer moments.
            head.weight.grad[:-17].zero_()
            head.bias.grad[:-17].zero_()
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        batches.append(batch.tolist())
        losses.append(float(loss.detach()))
    for name, value in statistics.items():
        if not torch.equal(value, fitted.state_dict()[name]):
            raise RuntimeError(f"short fit changed normalizer statistics: {name}")
    if contact_head_only:
        head_name = f"net.model.{len(fitted.net.model) - 1}."
        for name, value in fitted.state_dict().items():
            before = original[name]
            if name in (head_name + "weight", head_name + "bias"):
                before, value = before[:-17], value[:-17]
            if not torch.equal(before, value):
                raise RuntimeError(f"contact-only fit changed frozen weights/statistics: {name}")
        for name, p in fitted.net.named_parameters():
            p.requires_grad_(requires_grad[name])
    fitted.eval()
    return fitted, {
        "optimizer_updates": updates,
        "policy_updates": 0,
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "fresh Adam; gradient norm clipped at 1.0",
        "sampling": "uniform with replacement from fixed archived training indices; CPU generator",
        "training_indices": training.tolist(),
        "validation_indices": validation.tolist(),
        "sampled_window_indices": batches,
        "minibatch_losses_before_update": losses,
        "normalizer_statistics_unchanged": True,
        "omit_clock_loss": omit_clock_loss,
        "clock_loss_weight": 0.0 if omit_clock_loss or contact_head_only else fitted.contact_clock_loss_weight,
        "contact_head_only": contact_head_only,
        "objective": "contact BCE only" if contact_head_only else "world model rollout loss",
        "scope": "Bounded dynamics fitting only; no policy or physical rollout assessment.",
    }


def split_model_windows(task, context):
    """Keep all windows sharing a terrain origin on the same side."""
    origins, groups = torch.unique(task.context_parts(context)["env_origin"], dim=0, return_inverse=True)
    if len(origins) < 2:
        raise ValueError("location holdout requires at least two distinct terrain origins")
    heldout = groups == len(origins) - 1
    return (~heldout).nonzero().flatten(), heldout.nonzero().flatten(), origins


def short_model_fit(model, task, kinematics, initial, actions, context, targets, updates, learning_rate=2e-4):
    """Fit a copy with fixed inherited statistics and a fresh Adam optimizer.

    This tests local fitting/transfer of a small archive, not policy performance
    or independent evaluation of the checkpoint's original training distribution.
    """
    if updates < 1:
        raise ValueError("updates must be positive")
    training, heldout, origins = split_model_windows(task, context)
    fitted = copy.deepcopy(model)
    optimizer = torch.optim.Adam(fitted.net.parameters(), lr=learning_rate)
    feet = [TRACKED_BODIES.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]

    @torch.no_grad()
    def evaluate(indices):
        state, act, ctx, target = initial[indices], actions[indices], context[indices], targets[indices]
        fitted.eval()
        predicted = fitted.rollout(state, act, ctx)
        p, actual = task.parts(predicted), task.parts(target)
        links = kinematics.links(p, task.context_parts(ctx)["base_com"][:, None])
        errors = {
            "root_position_rmse_m": p["root_pos_w"] - actual["root_pos_w"],
            "joint_position_rmse_rad": p["joint_pos"] - actual["joint_pos"],
            "link_position_rmse_m": p["body_link_pos_w"] - actual["body_link_pos_w"],
            "feet_height_rmse_m": p["body_link_pos_w"][..., feet, 2] - actual["body_link_pos_w"][..., feet, 2],
            "kinematic_feet_height_rmse_m": p["body_link_pos_w"][..., feet, 2] - links["body_link_pos_w"][..., feet, 2],
        }
        result = {name: float(error.square().mean().sqrt()) for name, error in errors.items()}
        result.update({"final_" + name: float(error[:, -1].square().mean().sqrt()) for name, error in errors.items()})
        result["supervised_loss"] = float(fitted.rollout_loss(state, act, ctx, target, update_normalizer=False)["loss"])
        return result

    history = []
    for step in range(updates + 1):
        if step == 0 or step % 16 == 0 or step == updates:
            history.append({"updates": step, "training": evaluate(training), "heldout": evaluate(heldout)})
        if step == updates:
            break
        fitted.train()
        optimizer.zero_grad(set_to_none=True)
        loss = fitted.rollout_loss(
            initial[training], actions[training], context[training], targets[training], update_normalizer=False
        )["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite short-fit dynamics loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(fitted.net.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    fitted.eval()
    return fitted, {
        "optimizer_updates": updates,
        "policy_updates": 0,
        "learning_rate": learning_rate,
        "optimizer": "fresh Adam; gradient norm clipped at 1.0",
        "normalizers": "frozen checkpoint statistics; no training or validation updates",
        "split": {
            "method": "whole terrain origins; lexicographically last origin held out",
            "origins": origins.tolist(),
            "training_indices": training.tolist(),
            "heldout_indices": heldout.tolist(),
        },
        "history": history,
        "scope": (
            "Small-archive dynamics fitting/transfer only. Spatial holdout is disjoint from this fit;"
            " independence from the source checkpoint's unavailable replay is unproven."
            " No physical rollout or policy-quality claim."
        ),
    }
