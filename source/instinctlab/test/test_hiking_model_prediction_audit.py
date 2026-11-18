"""Ensure oracle contact probes isolate feedback without leaking output labels."""

import torch
from types import SimpleNamespace

from instinctlab.tasks.parkour.supermimic.model_prediction_audit import prediction_error_decomposition


def test_oracle_changes_only_future_inputs_and_errors_precede_label_substitution():
    def parts(state):
        return {
            "root_pos_w": state[:, :1],
            "root_lin_vel_w": state[:, 1:2],
            "joint_pos": state[:, 2:3],
            "contact_state": state[:, 3:],
        }

    task = SimpleNamespace(parts=parts, state_slices={"contact_state": slice(3, 9)})

    def model(state, action, context):
        prediction = state.clone()
        prediction[:, 0] += state[:, 7]  # Motion depends on the previous left contact.
        prediction[:, 7] = 0  # Deliberately incorrect contact prediction.
        return prediction

    initial = torch.zeros(1, 9)
    targets = torch.zeros(1, 3, 9)
    targets[0, :, 0] = torch.arange(3.0)
    targets[:, :, 7] = 1
    original = targets.clone()
    result = prediction_error_decomposition(model, task, initial, torch.zeros(1, 3, 1), None, targets)
    torch.testing.assert_close(targets, original)
    for step in result["steps"]:
        assert step["teacher_forced_root_pos_w_rmse"] == 0
        assert step["oracle_contact_input_root_pos_w_rmse"] == 0
        assert step["oracle_contact_input_event_error_rate"] == 0.5
    assert result["steps"][-1]["open_loop_root_pos_w_rmse"] == 2


def test_position_integration_preserves_geometry_and_uses_only_predicted_origin_velocity():
    from instinctlab.tasks.parkour.supermimic.model_prediction_audit import compare_position_integration
    from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_plane_cfg import TRACKED_BODIES

    count = len(TRACKED_BODIES)
    torso = TRACKED_BODIES.index("torso_link")
    fields = {
        "root_pos_w": slice(0, 3),
        "body_link_pos_w": slice(3, 3 + count * 3),
        "body_link_lin_vel_w": slice(3 + count * 3, 3 + count * 6),
        "joint_pos": slice(3 + count * 6, 4 + count * 6),
        "velocity_command": slice(4 + count * 6, 5 + count * 6),
    }

    def parts(state):
        result = {name: state[:, index] for name, index in fields.items()}
        for name in ("body_link_pos_w", "body_link_lin_vel_w"):
            result[name] = result[name].reshape(-1, count, 3)
        return result

    task = SimpleNamespace(
        parts=parts,
        state_slices=fields,
        dt=0.5,
        generated_command=lambda state, context: state[:, :1],
    )
    initial = torch.zeros(1, 5 + count * 6)
    offsets = torch.arange(count * 3, dtype=torch.float32).reshape(1, count, 3)
    parts(initial)["body_link_pos_w"].copy_(offsets)
    parts(initial)["body_link_lin_vel_w"][:, torso] = 2
    calls = []

    def model(state, action, context):
        calls.append(state.clone())
        result = state.clone()
        parts(result)["root_pos_w"].add_(10)
        parts(result)["body_link_pos_w"].add_(10)
        parts(result)["body_link_lin_vel_w"][:, torso] = 4
        return result

    actions = torch.zeros(1, 2, 1)
    targets = torch.zeros(1, 2, initial.shape[-1])
    original = initial.clone()
    compare_position_integration(model, task, initial, actions, None, targets)
    first_calls = calls[:]
    torch.testing.assert_close(initial, original)
    # The second model call for each branch receives its own first-step state.
    expected = (
        (10.0, 10.0, 10.0),
        (1.0, 1.0, 1.0),
        (2.0, 2.0, 2.0),
        (1.5, 1.5, 1.5),
        (1.0, 1.0, 10.0),
        (10.0, 10.0, 1.0),
    )
    for state, position in zip(calls[6:], expected, strict=True):
        p = parts(state)
        torch.testing.assert_close(p["root_pos_w"], torch.tensor([position]))
        torch.testing.assert_close(p["body_link_pos_w"] - p["root_pos_w"][:, None], offsets)
    for state in calls[7:]:
        torch.testing.assert_close(parts(state)["velocity_command"], state[:, :1])
    calls.clear()
    compare_position_integration(model, task, initial, actions, None, targets + 1000)
    for before, after in zip(first_calls, calls, strict=True):
        torch.testing.assert_close(before, after)


def test_kinematic_projection_isolates_inconsistent_links_without_future_labels():
    from instinctlab.tasks.parkour.supermimic.model_prediction_audit import compare_kinematic_predictions
    from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_plane_cfg import TRACKED_BODIES

    count = len(TRACKED_BODIES)
    size = 4 + 6 * count

    def parts(state):
        return {
            "root_pos_w": state[:, :3],
            "joint_pos": state[:, 3:4],
            "body_link_pos_w": state[:, 4 : 4 + 3 * count].reshape(-1, count, 3),
            "body_link_lin_vel_w": state[:, 4 + 3 * count :].reshape(-1, count, 3),
        }

    def links(p, root_com):
        return {
            "body_link_pos_w": (p["root_pos_w"] + p["joint_pos"])[:, None].expand(-1, count, -1),
            "body_link_lin_vel_w": torch.zeros_like(p["body_link_lin_vel_w"]),
        }

    def project(state, frames, root_com):
        result = state.clone()
        parts(result)["body_link_pos_w"].copy_(links(parts(state), root_com)["body_link_pos_w"])
        return result

    task = SimpleNamespace(
        parts=parts,
        context_parts=lambda context: {"base_com": context},
        physical_size=size,
        frames=None,
    )
    kinematics = SimpleNamespace(links=links, project=project)
    calls = []

    def model(state, action, context):
        calls.append(state.clone())
        result = state.clone()
        parts(result)["joint_pos"].add_(parts(state)["body_link_pos_w"][:, 0, :1])
        parts(result)["body_link_pos_w"].add_(1)
        return result

    initial = torch.zeros(1, size)
    targets = torch.zeros(1, 2, size)
    actions = torch.zeros(1, 2, 1)
    result = compare_kinematic_predictions(model, task, kinematics, initial, actions, None, targets)
    assert result["max_logged_position_distance_m"] == 0
    assert result["steps"][0]["projected_links"]["before_projection_consistency"]["feet_height_rmse_m"] == 1
    assert result["steps"][0]["projected_links"]["feet_height_rmse_m"] == 0
    assert result["steps"][1]["projected_links"]["joint_position_rmse_rad"] == 0
    assert result["steps"][1]["independent_links"]["joint_position_rmse_rad"] == 1
    first_calls = calls[:]
    calls.clear()
    compare_kinematic_predictions(model, task, kinematics, initial, actions, None, targets + 100)
    for before, after in zip(first_calls, calls, strict=True):
        torch.testing.assert_close(before, after)
    assert not initial.any()
    assert not targets.any()


def test_kinematic_gradient_audit_detects_alignment_and_conflict_without_updates():
    from instinctlab.tasks.parkour.supermimic.model_prediction_audit import audit_kinematic_loss_gradients

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = torch.nn.Linear(1, 2, bias=False)
            with torch.no_grad():
                self.net.weight.copy_(torch.tensor([[1.0], [0.0]]))
            self.net.weight.grad = torch.full_like(self.net.weight, 9)

        def rollout(self, initial, actions, context):
            return self.net(torch.ones(*actions.shape[:2], 1))

        def rollout_loss(self, initial, actions, context, targets, update_normalizer=True):
            assert update_normalizer is False
            return {"loss": (self.rollout(initial, actions, context) - targets).square().mean()}

    model = Model()
    task = SimpleNamespace(
        parts=lambda state: {"joint_pos": state[..., :1], "body_link_pos_w": state[..., 1:]},
        context_parts=lambda context: {"base_com": context},
    )
    kinematics = SimpleNamespace(links=lambda parts, root_com: {"body_link_pos_w": parts["joint_pos"]})
    state, actions, context = torch.zeros(1, 2), torch.zeros(1, 3, 1), torch.zeros(1, 3)
    before = model.net.weight.detach().clone()
    aligned = audit_kinematic_loss_gradients(model, task, kinematics, state, actions, context, torch.zeros(1, 3, 2))
    opposed = audit_kinematic_loss_gradients(
        model, task, kinematics, state, actions, context, torch.tensor([[[2.0, -1.0]]]).expand(1, 3, 2)
    )
    assert [row["horizon"] for row in aligned["steps"]] == [1, 3]
    for a, b in zip(aligned["steps"], opposed["steps"], strict=True):
        assert a["gradient_cosine"] > 0
        assert a["kinematic_directional_derivative_along_negative_supervised_gradient"] < 0
        assert b["gradient_cosine"] < 0
        assert b["kinematic_directional_derivative_along_negative_supervised_gradient"] > 0
        assert a["kinematic_joint_gradient_norm"] > 0
    torch.testing.assert_close(model.net.weight, before, rtol=0, atol=0)
    torch.testing.assert_close(model.net.weight.grad, torch.full_like(before, 9), rtol=0, atol=0)
