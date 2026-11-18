"""One F update and restored physical action probes in the native scene.

An untrained/one-update model is not expected to have accurate derivatives.
This audit measures that gap instead of treating connectivity as correctness.
"""

import torch

from .action_gradient_audit import compare_probe_derivatives, first_action_derivative
from .collection import DynamicsTerrainImporter, rasterize_terrain
from .native_physics import NativePhysics
from .newton_probes import collect_newton_action_probes


def configure_native_physics_audit(cfg):
    cfg.scene.terrain.class_type = DynamicsTerrainImporter
    return cfg


def native_height_field(env):
    """One raster and footprint shared by real termination and model queries."""
    terrain = env.scene.terrain
    if hasattr(terrain, "native_height_field"):
        return terrain.native_height_field
    x = torch.arange(-0.8, 0.85, 0.1, device=env.device)
    y = torch.arange(-0.5, 0.55, 0.1, device=env.device)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    offsets = torch.stack((xx.flatten(), yy.flatten()), -1)
    terrain.native_height_field, _ = rasterize_terrain(terrain.dynamics_mesh, env.device, 0.05, offsets)
    return terrain.native_height_field


def native_scan_out_of_bounds(env):
    """Real timeout when the full privileged critic footprint leaves the raster."""
    robot = env.scene["robot"]
    _, valid = native_height_field(env).scan(robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch)
    return ~valid


def make_native_physics(env, recorder, **kwargs):
    return NativePhysics(env, recorder, native_height_field(env), **kwargs)


class NativePhysicsProbeTask:
    """Read live physical states/rewards for the existing Newton probe collector."""

    physical_size = 71

    def __init__(self, physics):
        self.physics, self.env = physics, physics.env
        self.joint_ids = physics.joint_ids

    def real_state(self):
        reward = self.env.reward_buf
        terms = torch.stack(list(reward.values()), -1) if isinstance(reward, dict) else reward[:, None]
        return torch.cat((self.physics.recorder.state(), terms), -1)

    def real_context(self):
        return self.physics.context(self.env.action_manager.action)

    def parts(self, state):
        return dict(task_reward_terms=state[..., self.physical_size :])


def fit_and_probe_native_physics(physics, records, out):
    if len(records) < 8:
        raise ValueError("Physical audit needs at least eight real control transitions.")
    transitions, contexts = zip(*records[:8])
    stack = lambda name: torch.stack([getattr(t, name) for t in transitions], 1)  # noqa: E731
    state, target, actions = stack("state"), stack("next_state"), stack("action")
    context = torch.stack(contexts, 1)
    contact = torch.stack([physics.contact_targets(t) for t in transitions], 1)
    done = stack("terminated") | stack("truncated")
    mask = torch.cat((torch.ones_like(done[:, :1]), (~done[:, :-1]).cumprod(1).bool()), 1)
    auxiliary = torch.stack([physics.auxiliary_targets(t) for t in transitions], 1)
    loss, metrics = physics.window_loss(state, target, actions, context, contact, mask, auxiliary)
    optimizer = torch.optim.Adam(physics.model.parameters(), lr=1e-4)
    counts = dict(physical=0)

    def count_step(*args, **kwargs):
        counts["physical"] += 1

    optimizer.register_step_post_hook(count_step)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(physics.model.parameters(), 1.0)
    if not torch.isfinite(grad):
        raise FloatingPointError("Nonfinite native F gradient.")
    optimizer.step()
    print("NATIVE_PHYSICS_ONE_UPDATE", float(loss.detach()), flush=True)

    probe_report = probe_native_physics(physics, out)
    torch.save(
        dict(
            format="native_physical_diagnostic_v3",
            model=physics.model.state_dict(),
            root_frame="native_root_link",
            quaternion_order="wxyz",
            joint_names=physics.recorder.joint_names,
            contact_body_names=physics.contact_body_names,
            contact_target_names=physics.contact_target_names,
            context_names=physics.context_names,
            context_widths=physics.context_widths,
            auxiliary_names=physics.auxiliary_names,
            material_contract=physics.materials.contract,
            raster_resolution=physics.terrain.resolution,
            updates=counts,
            scope="Diagnostic only; simulator, replay, actor and perception are not included.",
        ),
        out / "diagnostic_physical_model.pt",
    )
    return dict(
        loss=float(loss.detach()),
        grad_norm=float(grad),
        metrics={name: float(value) for name, value in metrics.items()},
        actual_updates=counts,
        input_dimension=physics.model.in_dim,
        output_contacts=physics.model.num_contact,
        visual_dimension=physics.model.z_dim,
        auxiliary_dimension=physics.model.aux_dim,
        gradient_probe=probe_report["gradient_probe"],
        model_maturity="One diagnostic update only; no claim of learned physical gradient accuracy.",
        probe_model_and_normalizers_unchanged=True,
    )


def probe_native_physics(physics, out):
    """Compare the current F with restored real action perturbations; no fitting.

    Run at a final checkpoint boundary. Real probe branches bypass reset/events;
    noise-floor and termination masks determine which derivatives are informative.
    """
    env = physics.env
    task = NativePhysicsProbeTask(physics)
    actions = env.action_manager.action.detach()[:, None].expand(-1, 2, -1).clone()
    direction = torch.randn_like(actions[:, 0])
    direction /= direction.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    initial = physics.recorder.state()
    initial_context = task.real_context()

    def readout(states):
        # Only root-origin velocity and joint positions. No direct action reward.
        return torch.cat((states[..., 7:10], states[..., 13:42]), -1)

    before = {name: value.detach().clone() for name, value in physics.model.state_dict().items()}
    derivative = first_action_derivative(
        lambda controls: readout(physics.rollout(initial, controls, initial_context)), actions, direction
    )
    probes = collect_newton_action_probes(env, task, actions, direction)
    comparison = dict(
        root_origin_velocity=compare_probe_derivatives(probes, lambda states: states[..., 7:10], derivative[..., :3]),
        joint_positions=compare_probe_derivatives(probes, lambda states: states[..., 13:42], derivative[..., 3:]),
    )
    for name, value in physics.model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    torch.save(probes, out / "physical_action_probes.pt")
    return dict(
        gradient_probe=comparison,
        model_and_normalizers_unchanged=True,
        model_updates=0,
        policy_updates=0,
        scope="Local physical action derivatives only; not a gait-quality or model-readiness certificate.",
    )
