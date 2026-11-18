"""Exported P/actor preserves real inference and rejects incompatible clocks."""

import torch

import pytest
from instinctlab_perception.checkpoint import load_hiking_controller, save_hiking_controller
from instinctlab_perception.events import PerceptionRuntime
from instinctlab_perception.policy import make_hiking_policy
from instinctlab_perception.rssm import PerceptionCfg, PerceptionModel


def test_controller_export_matches_policy_and_camera_action_clock_after_partial_reset(tmp_path):
    torch.manual_seed(24)
    perception = PerceptionModel(PerceptionCfg(deter=16, stoch=2, categories=4, embed=16))
    policy = make_hiking_policy(perception.cfg.feature_dim)
    joints = [f"joint_{i}" for i in range(29)]
    path = tmp_path / "controller.pt"
    save_hiking_controller(path, perception, policy, 0.02, joints)
    packet = torch.load(path, weights_only=True)
    assert set(packet) == {
        "format",
        "perception_cfg",
        "observation_layout",
        "control_dt",
        "joint_names",
        "perception",
        "policy",
    }
    loaded, runtime = load_hiking_controller(path, 2, "cpu", 0.02, joints)
    original_runtime = PerceptionRuntime(perception, 2)
    assert all(not p.requires_grad for p in loaded.parameters())
    assert all(not p.requires_grad for p in runtime.model.parameters())
    with torch.no_grad():
        for step in range(12):
            depth, proprio, history = torch.rand(2, 1, 18, 32), torch.randn(2, 64), torch.randn(2, 768)
            captured = runtime.control_step.remainder(5) == 0
            expected, original_tick = original_runtime.before_action(depth, proprio, captured)
            actual, tick = runtime.before_action(depth, proprio, captured)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            expected_action = policy(history, expected).clamp(-20, 20)
            action = loaded(history, actual).clamp(-20, 20)
            torch.testing.assert_close(action, expected_action, rtol=0, atol=0)
            if tick is not None:
                for name in vars(tick):
                    torch.testing.assert_close(getattr(tick, name), getattr(original_tick, name), rtol=0, atol=0)
            runtime.after_action(action)
            original_runtime.after_action(expected_action)
            if step == 3:
                runtime.reset(torch.tensor([0]))
                original_runtime.reset(torch.tensor([0]))
    torch.testing.assert_close(runtime.actions, original_runtime.actions, rtol=0, atol=0)
    assert runtime.episode_id.tolist() == [1, 0]


def test_controller_rejects_reordered_joints_and_different_control_period(tmp_path):
    model = PerceptionModel(PerceptionCfg(deter=16, stoch=2, categories=4, embed=16))
    joints = [f"joint_{i}" for i in range(29)]
    path = tmp_path / "controller.pt"
    save_hiking_controller(path, model, make_hiking_policy(model.cfg.feature_dim), 0.02, joints)
    with pytest.raises(ValueError, match="ordering"):
        load_hiking_controller(path, 2, "cpu", 0.02, joints[::-1])
    with pytest.raises(ValueError, match="period"):
        load_hiking_controller(path, 2, "cpu", 0.01, joints)
