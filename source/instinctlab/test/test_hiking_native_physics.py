"""Native physical fitting uses pre-reset labels and preserves recurrent gradients."""

import torch
from pathlib import Path
from types import SimpleNamespace

import pytest
from instinctlab_supermimic.networks import WorldModel

from instinctlab.tasks.parkour.supermimic.native_physics import NativePhysics
from instinctlab.tasks.parkour.supermimic.stair_kinematics import NativeLinkKinematics


def fixture():
    torch.manual_seed(18)
    learner = object.__new__(NativePhysics)
    learner.foot_loss_weight = 1.0
    learner.foot_kinematics = NativeLinkKinematics(
        Path(__file__).resolve().parents[1]
        / "instinctlab/tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf",
        "cpu",
        ("left_ankle_roll_link", "right_ankle_roll_link"),
    )
    offsets = torch.tensor([[-0.1, 0.0], [0.0, 0.0], [0.1, 0.0]])

    def query(xy):
        assert torch.isfinite(xy).all(), "Invalid post-terminal padding reached the terrain query."
        return xy[..., 0] * 0.2 + xy[..., 1] * 0.1, torch.ones(xy.shape[:-1], dtype=torch.bool)

    learner.terrain = SimpleNamespace(local_xy=offsets, query=query)
    learner.model = WorldModel(3, 2, 7, hidden=(32, 32))
    learner.error_floor = torch.full((70,), 0.01)
    learner.foot_contact_ids = [0, 1]
    state = torch.zeros(2, 2, 71)
    state[..., 2], state[..., 3] = 1, 1
    target = state.clone()
    target[..., 13:42] = 0.1
    target[..., 42:71] = 0.2
    actions, context, contacts = torch.ones(2, 2, 29), torch.zeros(2, 2, 7), torch.ones(2, 2, 2)
    mask = torch.tensor([[True, False], [True, True]])
    return learner, [state, target, actions, context, contacts, mask]


def test_visual_dynamics_receives_no_privileged_scan_but_critic_keeps_it():
    from instinctlab_supermimic.physical import unflatten_state

    learner, batch = fixture()
    learner.model = WorldModel(0, 2, 7, z_dim=32, hidden=(32, 32))
    state = unflatten_state(batch[0][:, 0])
    assert learner.scan(state).shape == (2, 3)  # Privileged critic features remain available.
    assert learner.model_scan(state).shape == (2, 0)

    def forbid_lookup(_):
        raise AssertionError("The visual dynamics path queried privileged terrain.")

    learner.terrain.query = forbid_lookup
    with torch.no_grad():
        learner.model.net[-1].weight.normal_(std=0.003)
    actions = torch.ones(2, 29, requires_grad=True)
    prediction, _, z = learner.model.step(
        state, actions, torch.zeros(2, 7), learner.model_scan(state), torch.ones(2, 32)
    )
    loss = prediction[0].sum() + z.sum()
    gradient = torch.autograd.grad(loss, actions)[0]
    assert gradient.abs().sum() > 0 and torch.isfinite(gradient).all()


def test_shoe_geometry_tracks_root_and_knee_with_finite_difference_gradients():
    from instinctlab_supermimic.physical import unflatten_state

    from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1

    learner, batch = fixture()
    state = batch[0][:, 0].clone()
    knee = 13 + list(G1_29DOF_POLICY_JOINT_ORDER_V1).index("left_knee_joint")
    state[:, knee] = 0.5
    state.requires_grad_()
    points = learner.foot_points(unflatten_state(state))
    derivative = torch.autograd.grad(points[:, 0, :2, 2].sum(), state)[0]
    epsilon = 1e-3
    plus, minus = state.detach().clone(), state.detach().clone()
    plus[:, knee] += epsilon
    minus[:, knee] -= epsilon
    finite_difference = (
        learner.foot_points(unflatten_state(plus))[:, 0, :2, 2].sum(-1)
        - learner.foot_points(unflatten_state(minus))[:, 0, :2, 2].sum(-1)
    ) / (2 * epsilon)
    torch.testing.assert_close(derivative[:, knee], finite_difference, atol=1e-4, rtol=1e-3)
    assert derivative[:, knee].abs().min() > 0.01
    shift = state.detach().clone()
    shift[:, :3] += torch.tensor([2.0, -1.0, 0.3])
    torch.testing.assert_close(
        learner.foot_points(unflatten_state(shift)) - points.detach(),
        torch.tensor([2.0, -1.0, 0.3]).expand_as(points),
        atol=3e-7,
        rtol=1e-5,
    )


def test_foot_supervision_excludes_reset_padding_and_reaches_model():
    learner, batch = fixture()
    loss, metrics = learner.window_loss(*batch)
    assert metrics["feet"] > 0
    batch[1][0, 1] = float("nan")
    masked_loss, masked_metrics = learner.window_loss(*batch)
    torch.testing.assert_close(loss, masked_loss)
    torch.testing.assert_close(metrics["feet"], masked_metrics["feet"])
    masked_loss.backward()
    assert learner.model.net[-1].weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in learner.model.parameters() if p.grad is not None)


def test_foot_loss_ablation_preserves_original_objective_and_adds_leg_gradients():
    from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1

    learner, batch = fixture()
    augmented, metrics = learner.window_loss(*batch)
    learner.foot_loss_weight = 0.0
    original, _ = learner.window_loss(*batch)
    torch.testing.assert_close(original.detach(), metrics["physical"] + 0.1 * metrics["contact"])
    torch.testing.assert_close((augmented - original).detach(), metrics["feet"])
    (augmented - original).backward()
    gradients = learner.model.net[-1].weight.grad
    knee = list(G1_29DOF_POLICY_JOINT_ORDER_V1).index("left_knee_joint")
    shoulder = list(G1_29DOF_POLICY_JOINT_ORDER_V1).index("left_shoulder_pitch_joint")
    assert gradients[12 + knee].norm() > 0
    torch.testing.assert_close(gradients[12 + shoulder], torch.zeros_like(gradients[12 + shoulder]))


def test_legacy_resume_keeps_original_foot_objective(tmp_path):
    from dataclasses import asdict, replace

    from instinctlab.tasks.parkour.supermimic.native_runner import NativeDualRunner, NativeRunnerCfg

    runner = NativeDualRunner.__new__(NativeDualRunner)
    runner.cfg, runner.contract = NativeRunnerCfg(), {"fixture": True}
    runner.replay = SimpleNamespace(
        controls=0, load_state_dict=lambda value: None, begin_collection_segment=lambda: None
    )
    runner.perception_replay = SimpleNamespace(load_state_dict=lambda value: None, episodes=[])
    runner.modules, runner.optimizers, runner.update_counts = {}, {}, {}
    runner.refresh_runtime = lambda: None
    saved_cfg = asdict(runner.cfg)
    saved_cfg.pop("physical_foot_weight")
    saved_cfg.pop("physical_replay_ratio")
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        dict(
            format="native_hiking_dual_v1",
            cfg=saved_cfg,
            contract=runner.contract,
            modules={},
            optimizers={},
            replay={},
            perception_replay={},
            iteration=6200,
            update_counts={},
        ),
        checkpoint,
    )
    runner.load(checkpoint)
    assert runner.iteration == 6200
    runner.cfg = replace(runner.cfg, physical_replay_ratio=16.0)
    with pytest.raises(ValueError, match="configuration"):
        runner.load(checkpoint)
    runner.cfg = replace(runner.cfg, physical_replay_ratio=None)
    runner.cfg = replace(runner.cfg, physical_foot_weight=1.0)
    with pytest.raises(ValueError, match="configuration"):
        runner.load(checkpoint)


def test_checkpoint_contract_distinguishes_terrain_support_and_joint_order(tmp_path):
    from instinctlab.tasks.parkour.supermimic.terrain import TerrainHeightField

    learner, _ = fixture()
    urdf = tmp_path / "robot.urdf"
    urdf.write_text("<robot name='fixture' />")
    learner.robot = SimpleNamespace(cfg=SimpleNamespace(spawn=SimpleNamespace(asset_path=str(urdf))))
    learner.recorder = SimpleNamespace(joint_names=[f"joint_{i}" for i in range(29)])
    learner.contact_target_names = ["left", "right"]
    learner.context_names, learner.context_widths = ("previous_action",), [29]
    learner.auxiliary_names = ("torque", "acceleration")
    learner.terrain = TerrainHeightField(
        torch.zeros(3, 3), torch.ones(3, 3, dtype=torch.bool), [0, 0], 0.1, torch.tensor([[0.0, 0.0]])
    )
    original = learner.checkpoint_contract()
    assert original == learner.checkpoint_contract()
    learner.terrain.valid[0, 0] = False
    assert learner.checkpoint_contract()["terrain_sha256"] != original["terrain_sha256"]
    learner.terrain.valid[0, 0] = True
    learner.terrain.heights[0, 0] = 0.1
    assert learner.checkpoint_contract()["terrain_sha256"] != original["terrain_sha256"]
    learner.recorder.joint_names.reverse()
    assert learner.checkpoint_contract()["joint_names"] != original["joint_names"]


def test_native_scan_reports_per_row_support_but_real_scan_stays_strict():
    from instinctlab_supermimic.physical import unflatten_state

    from instinctlab.tasks.parkour.supermimic.terrain import TerrainHeightField

    learner, batch = fixture()
    learner.terrain = TerrainHeightField(
        torch.zeros(5, 5), torch.ones(5, 5, dtype=torch.bool), [-1, -1], 0.5, torch.tensor([[0.0, 0.0], [0.1, 0.0]])
    )
    packed = batch[0][:, 0].clone()
    packed[1, 0] = 2
    state = unflatten_state(packed)
    heights, valid = learner.scan_with_validity(state)
    assert valid.tolist() == [True, False]
    assert torch.isfinite(heights).all()
    with pytest.raises(ValueError, match="supported native terrain"):
        learner.scan(state)


def test_padded_failure_windows_fit_without_nan_or_normalizer_updates():
    learner, batch = fixture()
    baseline, _ = learner.window_loss(*batch)
    for value in batch[:-1]:
        value[0, 1] = float("nan")
    loss, metrics = learner.window_loss(*batch)
    torch.testing.assert_close(loss, baseline, rtol=0, atol=0)
    assert metrics["valid"] == 3
    assert learner.model.in_norm.count == learner.model.out_norm.count == 0
    optimizer = torch.optim.Adam(learner.model.parameters(), lr=1e-3)
    loss.backward()
    assert learner.model.net[-1].weight.grad.norm() > 0
    assert all(torch.isfinite(p.grad).all() for p in learner.model.parameters() if p.grad is not None)
    optimizer.step()
    following, _ = learner.window_loss(*batch)
    assert following < loss


def test_terrain_derivative_and_first_action_reach_later_physical_state():
    learner, batch = fixture()
    with torch.no_grad():
        learner.model.net[-1].weight.normal_(std=0.01)
    batch[2].requires_grad_(True)
    loss, _ = learner.window_loss(*batch)
    gradient = torch.autograd.grad(loss, batch[2])[0]
    assert gradient[1, 0].norm() > 0
    assert gradient[1, 1].norm() > 0
    assert gradient[0, 1].norm() == 0
    position = torch.tensor([[0.0, 0.0, 1.0]], requires_grad=True)
    scan = learner.scan((position, torch.eye(3)[None]))
    derivative = torch.autograd.grad(scan.sum(), position)[0]
    torch.testing.assert_close(derivative, torch.tensor([[-0.6, -0.3, 3.0]]))


def test_nonprefix_masks_cannot_cross_a_reset():
    learner, batch = fixture()
    batch[-1][0] = torch.tensor([False, True])
    with pytest.raises(ValueError, match="valid prefix"):
        learner.window_loss(*batch)


@pytest.mark.parametrize("mutate_model", [False, True])
def test_current_model_probe_does_not_fit_and_detects_parameter_mutation(tmp_path, monkeypatch, mutate_model):
    from instinctlab.tasks.parkour.supermimic import native_physics_audit

    learner, batch = fixture()
    # Native rollout advances the leading 29 previous-action context entries.
    learner.model = WorldModel(3, 2, 30, hidden=(32, 32))
    with torch.no_grad():
        learner.model.net[-1].weight.normal_(std=0.01)
    state, _, actions, context, _, _ = batch
    context = torch.zeros(2, 2, 30)
    learner.env = SimpleNamespace(action_manager=SimpleNamespace(action=actions[:, 0]), device="cpu")
    learner.recorder = SimpleNamespace(state=lambda: state[:, 0])
    learner.joint_ids = list(range(29))
    learner.context = lambda previous: context[:, 0]
    before = {name: value.clone() for name, value in learner.model.state_dict().items()}
    # autograd.grad in the probe must preserve preexisting training gradients.
    for parameter in learner.model.parameters():
        parameter.grad = torch.ones_like(parameter)

    @torch.no_grad()
    def surrogate_branches(env, task, controls, direction):
        # Wrapper regression only: surrogate branches exercise the existing
        # finite-difference comparison without claiming real Newton evidence.
        outcomes = {}
        branches = [("zero_a", 0), ("zero_0", 0)]
        for epsilon in (0.05, 0.15):
            branches.extend([(f"plus_{epsilon}", epsilon), (f"minus_{epsilon}", -epsilon)])
        for name, epsilon in branches:
            applied = controls.clone()
            applied[:, 0] += epsilon * direction
            outcomes[name] = dict(
                states=learner.rollout(state[:, 0], applied, context[:, 0]),
                done=torch.zeros(2, 2, dtype=torch.bool),
            )
        if mutate_model:
            next(learner.model.parameters()).add_(1)
        return dict(outcomes=outcomes, restore_max_state_errors=torch.zeros(len(branches), 2))

    monkeypatch.setattr(native_physics_audit, "collect_newton_action_probes", surrogate_branches)
    if mutate_model:
        with pytest.raises(AssertionError):
            native_physics_audit.probe_native_physics(learner, tmp_path)
        assert not (tmp_path / "physical_action_probes.pt").exists()
        return
    report = native_physics_audit.probe_native_physics(learner, tmp_path)
    assert report["model_updates"] == report["policy_updates"] == 0
    assert report["model_and_normalizers_unchanged"]
    assert (tmp_path / "physical_action_probes.pt").exists()
    for name, value in learner.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    for parameter in learner.model.parameters():
        torch.testing.assert_close(parameter.grad, torch.ones_like(parameter), rtol=0, atol=0)
    for readout in report["gradient_probe"].values():
        assert all(p["relative_error"] < 0.01 for p in readout["perturbations"])


def test_native_auxiliary_supervision_excludes_padding_and_reaches_output_weights():
    learner, batch = fixture()
    learner.model = WorldModel(3, 2, 7, hidden=(32, 32), aux_dim=58)
    learner.auxiliary_floor = torch.cat((torch.ones(29), torch.full((29,), 10.0)))
    torque = torch.full((2, 2, 29), 2.0)
    acceleration = torch.full((2, 2, 29), 30.0)
    # These labels are actual last-substep readouts, independent of qvel differences.
    labels = learner.auxiliary_targets(SimpleNamespace(applied_torque=torque, joint_acceleration=acceleration))
    with pytest.raises(ValueError, match="torque/qacc"):
        learner.window_loss(*batch)
    loss, metrics = learner.window_loss(*batch, auxiliary=labels)
    torch.testing.assert_close(metrics["auxiliary"], torch.tensor(9.75))
    labels[0, 1] = float("nan")
    padded_loss, _ = learner.window_loss(*batch, auxiliary=labels)
    torch.testing.assert_close(loss, padded_loss, rtol=0, atol=0)
    padded_loss.backward()
    gradient = learner.model.net[-1].weight.grad[-58:]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0


def test_contact_targets_use_force_history_before_reset():
    learner, _ = fixture()
    history = torch.zeros(2, 3, 2, 3)
    history[0, 0, 0, 2] = 2
    history[1, 2, 1, 2] = 1  # Strictly greater than the physical target threshold.
    normal = history.clone()
    normal[0].zero_()
    timers = torch.tensor([[0.0, 0.02], [0.0, 0.0]])
    contact = learner.contact_targets(
        SimpleNamespace(force_history=history, normal_force_history=normal, contact_time=timers)
    )
    torch.testing.assert_close(contact, torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6]))


def test_open_loop_previous_action_advances_without_future_action_leakage():
    learner, batch = fixture()
    contexts = []

    def step(state, action, context, scan):
        contexts.append(context.detach().clone())
        p, rotation, v, w, q, qd = state
        following = (p, rotation, v, w, q + action * 0.01 + context[:, :29] * 0.02, qd)
        return following, torch.zeros(len(action), 2), None

    learner.model.step = step
    actions = batch[2].requires_grad_(True)
    context = torch.arange(60.0)[None].expand(2, -1)
    states = learner.rollout(batch[0][:, 0], actions, context)
    torch.testing.assert_close(contexts[0], context)
    torch.testing.assert_close(contexts[1][:, :29], actions[:, 0])
    torch.testing.assert_close(contexts[1][:, 29:], context[:, 29:])
    first = torch.autograd.grad(states[:, 0, 13:42].sum(), actions, retain_graph=True)[0]
    second = torch.autograd.grad(states[:, 1, 13:42].sum(), actions)[0]
    assert first[:, 1].abs().sum() == 0
    torch.testing.assert_close(second[:, 0], torch.full((2, 29), 0.03))
    torch.testing.assert_close(second[:, 1], torch.full((2, 29), 0.01))
