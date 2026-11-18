"""Disposable, zero-training Newton/MuJoCo Warp local action probes.

The snapshot is specific to the current all-implicit, single-state solver path.
Zero-action repeats measure residual replay uncertainty even after restoring
native solver arrays. Never treat differences below that floor as derivatives.
"""

import numpy as np
import random
import torch
from contextlib import contextmanager
from dataclasses import is_dataclass

import warp as wp


@contextmanager
def record_newton_substeps(env, task):
    """Record physics transitions after scene refresh, before any auto-reset.

    A temporary observer for the GPU single-state, implicit-PD path with Python
    decimation and one solver step per physics call. The caller clears/consumes
    the yielded list at each control boundary. No state is restored: ordinary
    ``env.step`` still advances physics and owns termination/reset handling.
    """
    raw = env.unwrapped
    manager = raw.sim.backend_manager
    robot = raw.scene["robot"]
    if (
        type(manager).__name__ != "NewtonMJWarpManager"
        or not manager._use_single_state
        or manager._solver.use_mujoco_cpu
        or manager._adapter is not None
        or manager._num_substeps != 1
        or raw._physics_handles_decimation
        or any(type(a).__name__ != "ImplicitActuator" for a in robot.actuators.values())
    ):
        raise ValueError("substep recording requires GPU single-state implicit PD with Python physics decimation")
    action_term = raw.action_manager.get_term("joint_pos")
    original_step, original_update = raw.sim.step, raw.scene.update
    records, pending = [], None

    def physical_state():
        return task.real_state()[..., : task.physical_size].detach().clone()

    def physics_step(*args, **kwargs):
        nonlocal pending
        if pending is not None:
            raise RuntimeError("physics advanced without the preceding scene refresh")
        context = task.real_context().detach().clone()
        pending = {
            "state": physical_state(),
            "context": context,
            "root_com": task.context_parts(context)["base_com"].clone(),
            "action": raw.action_manager.action.detach().clone(),
            "processed_action": action_term.processed_actions.detach().clone(),
            "position_target": robot.data.joint_pos_target.torch[:, task.joint_ids].detach().clone(),
            "timestamp": float(robot.data._sim_timestamp),
        }
        result = original_step(*args, **kwargs)
        pending["solver_qpos"] = wp.to_torch(manager._solver.mjw_data.qpos).clone()
        pending["solver_qvel"] = wp.to_torch(manager._solver.mjw_data.qvel).clone()
        pending["solver_ctrl"] = wp.to_torch(manager._solver.mjw_data.ctrl).clone()
        return result

    def scene_update(dt, *args, **kwargs):
        nonlocal pending
        result = original_update(dt, *args, **kwargs)
        if pending is not None:
            if abs(dt - raw.physics_dt) > 1e-12:
                raise ValueError("recorded scene refresh interval differs from one physics step")
            pending["next_state"] = physical_state()
            pending["next_timestamp"] = float(robot.data._sim_timestamp)
            records.append(pending)
            pending = None
        return result

    raw.sim.step, raw.scene.update = physics_step, scene_update
    try:
        yield records
        if pending is not None:
            raise RuntimeError("substep recorder exited before the final scene refresh")
    finally:
        raw.sim.step, raw.scene.update = original_step, original_update


def snapshot_probe_owners(owners):
    """Preserve native buffers, dataclass caches and scalar clocks in named roots.

    Arbitrary object references are not recursively traversed. The caller lists
    all native owners explicitly; configuration and immutable resources stay put.
    Buffer object references are restored too, since action terms can rebind them.
    """
    arrays, references, scalars, visited = {}, [], [], set()

    def visit(owner, path):
        if id(owner) in visited:
            return
        visited.add(id(owner))
        entries = owner.items() if isinstance(owner, dict) else vars(owner).items()
        for name, value in list(entries):
            label = f"{path}.{name}"
            if isinstance(value, (wp.array, torch.Tensor)):
                if id(value) not in arrays:
                    saved = wp.clone(value) if isinstance(value, wp.array) else value.clone()
                    arrays[id(value)] = (value, saved, label)
                references.append((owner, name, value))
            elif isinstance(value, (int, float, bool, np.integer, np.floating)):
                scalars.append((owner, name, value))
            elif name != "cfg" and (
                isinstance(value, dict) or is_dataclass(value) or type(value).__name__ == "ProxyArray"
            ):
                visit(value, label)

    for name, owner in owners:
        if owner is not None:
            visit(owner, name)
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state(), torch.cuda.get_rng_state_all())

    @torch.no_grad()
    def restore():
        for owner, name, value in references + scalars:
            current = owner[name] if isinstance(owner, dict) else getattr(owner, name)
            if current is value:
                continue
            if isinstance(owner, dict):
                owner[name] = value
            else:
                setattr(owner, name, value)
        for value, saved, _ in arrays.values():
            if isinstance(value, wp.array):
                wp.copy(value, saved)
            else:
                value.copy_(saved)
        random.setstate(rng[0])
        np.random.set_state(rng[1])
        torch.random.set_rng_state(rng[2])
        torch.cuda.set_rng_state_all(rng[3])

    return restore, {"array_fields": [entry[2] for entry in arrays.values()], "scalar_fields": len(scalars)}


@torch.no_grad()
def collect_newton_action_probes(env, task, actions, direction, epsilons=(0.05, 0.15), contact_body_ids=None):
    """Native rewards/states without events or auto-reset; perturb first action.

    Restore the entire starting state before every branch and on exit. Terminal
    flags are retained; callers must exclude subsequent invalid continuation.
    """
    raw = env.unwrapped
    manager = raw.sim.backend_manager
    robot = raw.scene["robot"]
    if type(manager).__name__ != "NewtonMJWarpManager" or not manager._use_single_state:
        raise ValueError("probe requires Newton MuJoCo Warp single-state solver")
    if manager._solver.use_mujoco_cpu or robot.newton_actuator_adapter is not None:
        raise ValueError("probe supports GPU solver with implicit actuators and no native adapter")
    if any(type(actuator).__name__ != "ImplicitActuator" for actuator in robot.actuators.values()):
        raise ValueError("probe requires all-implicit actuator groups")
    if actions.shape != (raw.num_envs, actions.shape[1], 29) or direction.shape != (raw.num_envs, 29):
        raise ValueError("probe action sequence/direction shape differs")
    initial, context = task.real_state().clone(), task.real_context().clone()
    owners = [
        ("environment", raw),
        ("simulation", raw.sim),
        ("manager", manager),
        ("scene_data_backend", manager._scene_data_backend),
        ("state0", manager.get_state_0()),
        ("state1", manager.get_state_1()),
        ("control", manager.get_control()),
        ("solver", manager._solver),
        ("mjw_data", manager._solver.mjw_data),
        ("contacts", manager.get_contacts()),
        ("robot", robot),
        ("robot_data", robot._data),
        ("actuators", robot.actuators),
        ("action_manager", raw.action_manager),
        ("reward_manager", raw.reward_manager),
        ("termination_manager", raw.termination_manager),
    ]
    owners += [(f"action.{name}", term) for name, term in raw.action_manager._terms.items()]
    owners += [(f"command.{name}", term) for name, term in raw.command_manager._terms.items()]
    owners += [(f"native_contact.{i}", sensor) for i, sensor in enumerate(manager._newton_contact_sensors.values())]
    for name, sensor in raw.scene.sensors.items():
        owners.extend([(f"sensor.{name}", sensor), (f"sensor_data.{name}", sensor._data)])
    restore, manifest = snapshot_probe_owners(owners)
    outcomes, restore_errors = {}, []
    branches = [("zero_a", 0.0)]
    for i, epsilon in enumerate(epsilons):
        branches.extend([(f"plus_{epsilon}", epsilon), (f"minus_{epsilon}", -epsilon), (f"zero_{i}", 0.0)])
    try:
        for label, epsilon in branches:
            restore()
            restored = task.real_state()
            restore_errors.append((restored - initial).abs().amax(-1).cpu())
            torch.testing.assert_close(restored, initial, rtol=0, atol=0)
            reached, rewards, terminated, foot_contacts = [], [], [], []
            for step in range(actions.shape[1]):
                applied = actions[:, step] + (epsilon * direction if step == 0 else 0)
                raw.action_manager.process_action(applied)
                steps_per_call = raw.cfg.decimation if raw._physics_handles_decimation else 1
                for _ in range(raw.cfg.decimation // steps_per_call):
                    raw._sim_step_counter += steps_per_call
                    raw.action_manager.apply_action()
                    raw.scene.write_data_to_sim()
                    raw.sim.step(render=False)
                    raw.scene.update(dt=raw.physics_dt * steps_per_call)
                raw.episode_length_buf += 1
                raw.common_step_counter += 1
                done = raw.termination_manager.compute()
                raw.reward_manager.compute(dt=raw.step_dt)
                raw.command_manager.compute(dt=raw.step_dt)
                state = task.real_state().clone()
                reached.append(state.cpu())
                rewards.append(task.parts(state)["task_reward_terms"].cpu())
                terminated.append(done.clone().cpu())
                if contact_body_ids is not None:
                    contact = raw.scene["contact_forces"].data.current_contact_time.torch[:, contact_body_ids] > 0
                    foot_contacts.append(contact.clone().cpu())
            outcomes[label] = {
                "states": torch.stack(reached, 1),
                "rewards": torch.stack(rewards, 1),
                "done": torch.stack(terminated, 1),
            }
            if contact_body_ids is not None:
                outcomes[label]["foot_contact"] = torch.stack(foot_contacts, 1)
    finally:
        restore()
    torch.testing.assert_close(task.real_state(), initial, rtol=0, atol=0)
    return {
        "initial": initial.cpu(),
        "context": context.cpu(),
        "actions": actions.cpu(),
        "direction": direction.cpu(),
        "outcomes": outcomes,
        "restore_max_state_errors": torch.stack(restore_errors),
        "snapshot": manifest,
        "backend_handles_decimation": raw._physics_handles_decimation,
        "model_updates": 0,
        "policy_updates": 0,
        "scope": "local open-loop physical action probe, no events/auto-reset; zero repeats required for acceptance",
    }
