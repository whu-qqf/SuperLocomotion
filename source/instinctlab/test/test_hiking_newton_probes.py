"""Probe restoration must recover caches and rebound action buffers."""

import numpy as np
import random
import torch
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import warp as wp

from isaaclab.utils.warp import ProxyArray

from instinctlab.tasks.parkour.supermimic.newton_probes import record_newton_substeps, snapshot_probe_owners


def substep_fixture():
    live, published = torch.zeros(1, 3), torch.zeros(1, 3)
    solver = SimpleNamespace(
        use_mujoco_cpu=False,
        mjw_data=SimpleNamespace(
            qpos=wp.zeros((1, 7), device="cpu"),
            qvel=wp.zeros((1, 6), device="cpu"),
            ctrl=wp.zeros((1, 29), device="cpu"),
        ),
    )
    manager = type("NewtonMJWarpManager", (), {})()
    manager._use_single_state, manager._adapter, manager._num_substeps, manager._solver = True, None, 1, solver
    data = SimpleNamespace(_sim_timestamp=0.0, joint_pos_target=SimpleNamespace(torch=torch.ones(1, 29)))
    robot = SimpleNamespace(data=data, actuators={"all": type("ImplicitActuator", (), {})()})

    def step():
        live.add_(1)

    def update(dt):
        data._sim_timestamp += dt
        published.copy_(live)

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.update = update
    term = SimpleNamespace(processed_actions=torch.ones(1, 29))
    raw = SimpleNamespace(
        sim=SimpleNamespace(step=step, backend_manager=manager),
        scene=scene,
        _physics_handles_decimation=False,
        physics_dt=0.005,
        action_manager=SimpleNamespace(action=torch.zeros(1, 29), get_term=lambda name: term),
    )
    task = SimpleNamespace(
        real_state=lambda: published,
        physical_size=3,
        real_context=lambda: torch.tensor([[0.1, 0.2, 0.3]]),
        context_parts=lambda c: {"base_com": c},
        joint_ids=list(range(29)),
    )
    return SimpleNamespace(unwrapped=raw), task, live, published


def test_substep_observer_records_after_refresh_and_preserves_pre_reset_target():
    env, task, _, published = substep_fixture()
    raw = env.unwrapped
    old_step, old_update = raw.sim.step, raw.scene.update
    with record_newton_substeps(env, task) as records:
        for _ in range(2):
            raw.sim.step()
            raw.scene.update(dt=0.005)
        # A control-boundary auto-reset must not overwrite the physical target.
        published.fill_(99)
        raw.scene.update(dt=0.0)
    assert len(records) == 2
    torch.testing.assert_close(records[0]["state"], torch.zeros(1, 3))
    torch.testing.assert_close(records[0]["next_state"], torch.ones(1, 3))
    torch.testing.assert_close(records[1]["state"], torch.ones(1, 3))
    torch.testing.assert_close(records[1]["next_state"], torch.full((1, 3), 2.0))
    assert records[1]["next_timestamp"] == 0.01
    assert raw.sim.step is old_step and raw.scene.update is old_update


def test_substep_observer_restores_hooks_on_failure_and_rejects_missing_refresh():
    env, task, _, _ = substep_fixture()
    raw = env.unwrapped
    old_step, old_update = raw.sim.step, raw.scene.update
    with pytest.raises(RuntimeError, match="before the final scene refresh"):
        with record_newton_substeps(env, task):
            raw.sim.step()
    assert raw.sim.step is old_step and raw.scene.update is old_update
    with pytest.raises(RuntimeError, match="original failure"):
        with record_newton_substeps(env, task):
            raise RuntimeError("original failure")
    assert raw.sim.step is old_step and raw.scene.update is old_update


def test_snapshot_restores_native_caches_aliases_rebindings_and_random_streams():
    @dataclass(frozen=True)
    class FrozenSelection:
        indices: object

    @dataclass
    class Cache:
        data: object
        timestamp: float

    native = wp.array([1.0, 2.0], dtype=wp.float32, device="cpu")
    tensor = torch.tensor([3.0, 4.0])
    proxy = ProxyArray(native)
    owner = SimpleNamespace(
        native=native,
        action=tensor,
        cache=Cache(proxy, 1.0),
        selection=FrozenSelection(native),
        step=7,
        nested={"tensor": tensor},
    )
    restore, manifest = snapshot_probe_owners([("owner", owner)])
    expected = (random.random(), np.random.rand(), torch.rand(3))
    owner.action = torch.zeros(2)
    owner.cache.timestamp = 20.0
    owner.step = 99
    owner.nested["tensor"] = torch.ones(2)
    tensor.zero_()
    native.fill_(8)
    restore()
    actual = (random.random(), np.random.rand(), torch.rand(3))
    assert owner.action is tensor and owner.nested["tensor"] is tensor
    assert owner.cache.timestamp == 1.0 and owner.step == 7
    torch.testing.assert_close(tensor, torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(proxy.torch, torch.tensor([1.0, 2.0]))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])
    assert len(manifest["array_fields"]) == 2
    restore()
    torch.testing.assert_close(proxy.torch, torch.tensor([1.0, 2.0]))
