"""Short physical perturbation probes with explicit actuator-queue restoration.

PhysX contact solver warm-start caches are not exposed by scene state. Two
unperturbed repeats quantify that replay floor; results below it are inconclusive.
These probes measure open-loop dynamics only, without reset, reward or curriculum.
"""

import torch
from dataclasses import fields, is_dataclass

import warp as wp


def _saved_arrays(value):
    if value is None:
        return []
    if isinstance(value, wp.array):
        return [(wp.to_torch(value), wp.to_torch(value).clone())]
    if is_dataclass(value):
        result = []
        for field in fields(value):
            result.extend(_saved_arrays(getattr(value, field.name)))
        return result
    raise TypeError(f"unsupported actuator state field {type(value)}")


@torch.no_grad()
def collect_physical_probes(env, state, context, action, horizon=4):
    """Restore identical physical/actuator state for +/- actions and zero repeats."""
    raw = env.unwrapped
    robot = raw.scene["robot"]
    adapter = robot.newton_actuator_adapter
    runtime = robot.actuators._control._actuator_runtime
    scene_state = raw.scene.get_state()
    states_a = list(adapter._states_a) if adapter is not None else []
    states_b = list(adapter._states_b) if adapter is not None else []
    saved = []
    for actuator_state in states_a + states_b:
        saved.extend(_saved_arrays(actuator_state))
    graph_index = runtime._native_actuator_graph_index if runtime is not None else None
    if raw._physics_handles_decimation:
        raise ValueError("physical probes currently require per-substep PhysX control")

    def restore():
        raw.scene.reset_to(scene_state)
        if adapter is not None:
            adapter._states_a, adapter._states_b = list(states_a), list(states_b)
            runtime._native_actuator_graph_index = graph_index
        for target, original in saved:
            target.copy_(original)

    def read_state():
        group = raw.observation_manager.compute_group("wm_state")
        if isinstance(group, dict):
            return torch.cat([v.reshape(raw.num_envs, -1) for v in group.values()], -1).clone()
        return group.clone()

    actions = action[:, None].expand(-1, horizon, -1).clone()
    direction = torch.randn_like(action)
    direction /= direction.norm(dim=-1, keepdim=True)
    results = {}
    restore_errors = []
    try:
        for label, epsilon in (
            ("zero_a", 0.0),
            ("plus_015", 0.15),
            ("minus_015", -0.15),
            ("zero_b", 0.0),
            ("plus_005", 0.05),
            ("minus_005", -0.05),
        ):
            restore()
            restore_errors.append((read_state() - state).abs().amax(-1))
            reached = []
            for step in range(horizon):
                applied = actions[:, step] + (epsilon * direction if step == 0 else 0)
                raw.action_manager.process_action(applied)
                for _ in range(raw.cfg.decimation):
                    raw.action_manager.apply_action()
                    raw.scene.write_data_to_sim()
                    raw.sim.step(render=False)
                    raw.scene.update(dt=raw.physics_dt)
                reached.append(read_state())
            results[label] = torch.stack(reached, 1).cpu()
    finally:
        restore()
    return {
        "state": state.cpu(),
        "context": context.cpu(),
        "actions": actions.cpu(),
        "direction": direction.cpu(),
        "outcomes": results,
        "restore_max_state_error": torch.stack(restore_errors).cpu(),
        "note": (
            "Physical state and applicable native-actuator queues restored; PhysX solver cache is not restored. "
            "Zero repeats measure replay uncertainty. No events, termination or auto-reset during probes."
        ),
    }
