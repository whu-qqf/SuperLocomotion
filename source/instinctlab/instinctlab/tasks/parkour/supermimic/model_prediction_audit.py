"""Matched zero-learning probes of dynamics errors and contact feedback."""

import torch

from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_plane_cfg import TRACKED_BODIES


def audit_kinematic_loss_gradients(model, task, kinematics, initial, actions, context, targets):
    """Measure whether the existing supervised loss opposes geometric consistency.

    Compute derivatives only: no optimizer, parameter update, normalizer update,
    or changes to parameter .grad buffers. The FK loss is in square metres.
    """
    parameters = tuple(model.net.parameters())
    rows = []
    for horizon in sorted({1, actions.shape[1]}):
        supervised = model.rollout_loss(
            initial, actions[:, :horizon], context, targets[:, :horizon], update_normalizer=False
        )["loss"]
        predicted = model.rollout(initial, actions[:, :horizon], context)
        parts = task.parts(predicted)
        root_com = task.context_parts(context)["base_com"][:, None]
        links = kinematics.links(parts, root_com)
        consistency = (parts["body_link_pos_w"] - links["body_link_pos_w"]).square().mean()
        joint_gradient = torch.autograd.grad(consistency, parts["joint_pos"], retain_graph=True)[0]
        measured_grad = torch.autograd.grad(supervised, parameters)
        consistency_grad = torch.autograd.grad(consistency, parameters)
        dot = sum((a * b).sum() for a, b in zip(measured_grad, consistency_grad, strict=True))
        measured_norm = sum(g.square().sum() for g in measured_grad).sqrt()
        consistency_norm = sum(g.square().sum() for g in consistency_grad).sqrt()
        denominator = measured_norm * consistency_norm
        rows.append(
            {
                "horizon": horizon,
                "supervised_loss": float(supervised.detach()),
                "kinematic_position_mse_m2": float(consistency.detach()),
                "supervised_net_gradient_norm": float(measured_norm),
                "kinematic_net_gradient_norm": float(consistency_norm),
                "kinematic_joint_gradient_norm": float(joint_gradient.norm()),
                "gradient_cosine": float(dot / denominator) if denominator > 0 else None,
                "kinematic_directional_derivative_along_negative_supervised_gradient": -float(dot),
            }
        )
    return {
        "windows": len(initial),
        "steps": rows,
        "scope": (
            "Local gradient alignment on saved windows, not an optimizer step or proof of training convergence."
            " Negative directional derivative means infinitesimal supervised descent reduces FK error."
        ),
    }


@torch.no_grad()
def compare_kinematic_predictions(model, task, kinematics, initial, actions, context, targets):
    """Check measured/FK agreement before comparing independent and projected links.

    Projection uses only predicted root/joints and fixed robot structure. Logged
    future states are used exclusively for metrics, never as rollout inputs.
    """
    feet = [TRACKED_BODIES.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link")]
    root_com = task.context_parts(context)["base_com"]

    def consistency(state):
        parts = task.parts(state)
        links = kinematics.links(parts, root_com)
        difference = parts["body_link_pos_w"] - links["body_link_pos_w"]
        return {
            "link_position_rmse_m": float(difference.square().mean().sqrt()),
            "max_link_position_distance_m": float(difference.norm(dim=-1).max()),
            "feet_height_rmse_m": float(difference[..., feet, 2].square().mean().sqrt()),
            "feet_height_bias_vs_fk_m": difference[..., feet, 2].mean(0).tolist(),
            "link_position_rmse_by_body_m": {
                body: float(error.square().mean().sqrt())
                for body, error in zip(TRACKED_BODIES, difference.unbind(-2), strict=True)
            },
            "link_origin_velocity_rmse_m_s": float(
                (parts["body_link_lin_vel_w"] - links["body_link_lin_vel_w"]).square().mean().sqrt()
            ),
        }

    logged = [consistency(initial)] + [consistency(targets[:, step]) for step in range(actions.shape[1])]
    states = {name: initial for name in ("independent_links", "projected_links")}
    rows = []
    for step in range(actions.shape[1]):
        row = {"horizon": step + 1, "logged_consistency": logged[step + 1]}
        actual = task.parts(targets[:, step])
        for name, previous in states.items():
            predicted = model(previous, actions[:, step], context)
            before_projection = consistency(predicted)
            if name == "projected_links":
                physical = kinematics.project(predicted[:, : task.physical_size], task.frames, root_com)
                predicted = torch.cat((physical, predicted[:, task.physical_size :]), -1)
            states[name] = predicted
            parts = task.parts(predicted)
            row[name] = {
                "before_projection_consistency": before_projection,
                "root_position_rmse_m": float((parts["root_pos_w"] - actual["root_pos_w"]).square().mean().sqrt()),
                "joint_position_rmse_rad": float((parts["joint_pos"] - actual["joint_pos"]).square().mean().sqrt()),
                "link_position_rmse_m": float(
                    (parts["body_link_pos_w"] - actual["body_link_pos_w"]).square().mean().sqrt()
                ),
                "feet_height_rmse_m": float(
                    (parts["body_link_pos_w"][..., feet, 2] - actual["body_link_pos_w"][..., feet, 2])
                    .square()
                    .mean()
                    .sqrt()
                ),
            }
        rows.append(row)
    return {
        "windows": len(initial),
        "initial_logged_consistency": logged[0],
        "max_logged_position_distance_m": max(row["max_link_position_distance_m"] for row in logged),
        "steps": rows,
        "scope": "Offline FK consistency and projected feedback; no training or physical-policy validation.",
    }


@torch.no_grad()
def compare_position_integration(model, task, initial, actions, context, targets):
    """Counterfactual position integration using predicted link-origin velocity.

    Translate all predicted links with the root correction, preserving relative
    geometry. No future ground-truth velocity is supplied to any branch.
    """
    torso = TRACKED_BODIES.index("torso_link")
    states = {
        name: initial
        for name in (
            "position_head",
            "previous_velocity",
            "next_velocity",
            "trapezoid",
            "previous_velocity_xy",
            "previous_velocity_z",
        )
    }
    rows = []
    for step in range(actions.shape[1]):
        row = {"horizon": step + 1}
        actual = task.parts(targets[:, step])
        for name, previous in states.items():
            predicted = model(previous, actions[:, step], context)
            p, old = task.parts(predicted), task.parts(previous)
            if name != "position_head":
                v0 = old["body_link_lin_vel_w"][:, torso]
                v1 = p["body_link_lin_vel_w"][:, torso]
                velocity = v1 if name == "next_velocity" else (v0 + v1) / 2 if name == "trapezoid" else v0
                position = old["root_pos_w"] + task.dt * velocity
                if name == "previous_velocity_xy":
                    position = torch.cat((position[:, :2], p["root_pos_w"][:, 2:]), -1)
                elif name == "previous_velocity_z":
                    position = torch.cat((p["root_pos_w"][:, :2], position[:, 2:]), -1)
                correction = position - p["root_pos_w"]
                predicted = predicted.clone()
                predicted[:, task.state_slices["root_pos_w"]] = position
                predicted[:, task.state_slices["body_link_pos_w"]] = (
                    p["body_link_pos_w"] + correction[:, None]
                ).flatten(1)
                predicted[:, task.state_slices["velocity_command"]] = task.generated_command(predicted, context)
                p = task.parts(predicted)
            states[name] = predicted
            error = p["root_pos_w"] - actual["root_pos_w"]
            row[name] = {
                "root_position_rmse_m": float(error.square().mean().sqrt()),
                "root_position_bias_xyz_m": error.mean(0).tolist(),
                "link_position_rmse_m": float(
                    (p["body_link_pos_w"] - actual["body_link_pos_w"]).square().mean().sqrt()
                ),
                "joint_position_rmse_rad": float((p["joint_pos"] - actual["joint_pos"]).square().mean().sqrt()),
            }
        rows.append(row)
    return {
        "windows": len(initial),
        "steps": rows,
        "scope": (
            "Offline counterfactuals on identical windows, not a validated dynamics change or physical controller."
        ),
    }


@torch.no_grad()
def prediction_error_decomposition(model, task, initial, actions, context, targets):
    """Compare the same windows with logged states and oracle contact inputs.

    Oracle contact is an offline counterfactual, not a deployable controller or
    a change to the model-validation gate. No normalizer statistics are updated.
    """
    free = oracle = initial
    previous = initial
    contact_slice = task.state_slices["contact_state"]
    steps = []
    for step in range(actions.shape[1]):
        target = targets[:, step]
        teacher_forced = model(previous, actions[:, step], context)
        free = model(free, actions[:, step], context)
        oracle = model(oracle, actions[:, step], context)
        actual = task.parts(target)
        row = {"horizon": step + 1}
        for name, predicted in (
            ("teacher_forced", teacher_forced),
            ("open_loop", free),
            ("oracle_contact_input", oracle),
        ):
            parts = task.parts(predicted)
            for field in ("root_pos_w", "root_lin_vel_w", "joint_pos"):
                row[f"{name}_{field}_rmse"] = float((parts[field] - actual[field]).square().mean().sqrt())
            error = parts["contact_state"] - actual["contact_state"]
            row[f"{name}_stance_clock_rmse_s"] = float(error[:, 2:4].square().mean().sqrt())
            row[f"{name}_event_error_rate"] = float(
                ((parts["contact_state"][:, 4:] >= 0.5) != actual["contact_state"][:, 4:].bool()).float().mean()
            )
        steps.append(row)
        # Keep the predicted physical state; substitute only contact history for
        # the NEXT call. Record errors above before replacing any predictions.
        oracle = torch.cat(
            (oracle[:, : contact_slice.start], target[:, contact_slice], oracle[:, contact_slice.stop :]), -1
        )
        previous = target
    return {
        "windows": len(initial),
        "steps": steps,
        "scope": (
            "Same real windows/actions; unbounded open loop, one-step teacher forcing, and oracle contact feedback."
            " Counterfactuals, not physical rollout or policy quality."
        ),
    }
