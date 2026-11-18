"""Same-Step autoreset must not change the collected physics targets."""

import torch
from types import SimpleNamespace

import pytest

from isaaclab.managers import RecorderTermCfg

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.tasks.parkour.supermimic.native_transitions import HikingTransitionRecorder


def fixture():
    def proxy(value):
        return SimpleNamespace(torch=value)

    data = SimpleNamespace(
        root_link_pos_w=proxy(torch.zeros(2, 3)),
        root_link_quat_w=proxy(torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)),
        root_link_lin_vel_w=proxy(torch.ones(2, 3)),
        root_link_ang_vel_w=proxy(torch.zeros(2, 3)),
        joint_pos=proxy(torch.arange(29.0).repeat(2, 1)),
        joint_vel=proxy(torch.zeros(2, 29)),
        applied_torque=proxy(torch.full((2, 29), 2.0)),
        joint_acc=proxy(torch.full((2, 29), 3.0)),
    )
    robot = SimpleNamespace(
        data=data,
        find_joints=lambda names, preserve_order: (list(reversed(range(29))), names),
    )
    contacts = SimpleNamespace(
        net_forces_w_history=proxy(torch.ones(2, 3, 4, 3)),
        net_normal_forces_w_history=proxy(torch.full((2, 3, 4, 3), 0.5)),
        current_air_time=proxy(torch.zeros(2, 4)),
        current_contact_time=proxy(torch.ones(2, 4)),
    )
    env = SimpleNamespace(
        scene={"robot": robot, "contact_forces": SimpleNamespace(data=contacts)},
        action_manager=SimpleNamespace(action=torch.ones(2, 29), prev_action=torch.full((2, 29), 0.5)),
        episode_length_buf=torch.tensor([7, 3]),
        reset_terminated=torch.tensor([True, False]),
        reset_time_outs=torch.tensor([False, False]),
        reward_buf={"rewards": torch.tensor([1.0, 2.0])},
    )
    return env, HikingTransitionRecorder(RecorderTermCfg(class_type=HikingTransitionRecorder), env)


def test_pre_reset_targets_survive_reset_and_live_buffer_rebinding():
    env, recorder = fixture()
    recorder.record_pre_step()
    env.scene["robot"].data.root_link_pos_w.torch.add_(0.25)
    recorder.record_post_step()
    # Same-Step autoreset mutates/rebinds these buffers before env.step returns.
    env.scene["robot"].data.root_link_pos_w.torch[0] = 10
    env.action_manager.action[0].zero_()
    env.reward_buf["rewards"].zero_()
    env.reset_terminated = torch.zeros(2, dtype=torch.bool)
    env.episode_length_buf[0] = 0
    env.scene["contact_forces"].data.current_contact_time.torch[0] = 0
    env.scene["contact_forces"].data.net_normal_forces_w_history.torch.zero_()
    env.scene["robot"].data.applied_torque.torch.zero_()
    env.scene["robot"].data.joint_acc.torch.zero_()
    transition = recorder.take_transition()
    assert transition.state.shape == (2, 71)
    torch.testing.assert_close(transition.state[:, :3], torch.zeros(2, 3))
    torch.testing.assert_close(transition.next_state[:, :3], torch.full((2, 3), 0.25))
    torch.testing.assert_close(transition.action, torch.ones(2, 29))
    torch.testing.assert_close(transition.previous_action, torch.full((2, 29), 0.5))
    assert transition.episode_step.tolist() == [7, 3]
    assert transition.terminated.tolist() == [True, False]
    assert transition.rewards["rewards"].tolist() == [1.0, 2.0]
    assert transition.contact_time[0].tolist() == [1.0] * 4
    torch.testing.assert_close(transition.normal_force_history, torch.full((2, 3, 4, 3), 0.5))
    torch.testing.assert_close(transition.applied_torque, torch.full((2, 29), 2.0))
    torch.testing.assert_close(transition.joint_acceleration, torch.full((2, 29), 3.0))


def test_state_uses_policy_joint_order_link_origin_velocity_and_wxyz():
    _, recorder = fixture()
    state = recorder.state()
    torch.testing.assert_close(state[:, 3:7], torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1))
    torch.testing.assert_close(state[:, 7:10], torch.ones(2, 3))
    torch.testing.assert_close(state[:, 13:42], torch.arange(28.0, -1, -1).repeat(2, 1))
    assert recorder.joint_names == list(G1_29DOF_POLICY_JOINT_ORDER_V1)


def test_recorder_rejects_silent_transition_loss_and_unmatched_callbacks():
    _, recorder = fixture()
    with pytest.raises(RuntimeError, match="No completed"):
        recorder.take_transition()
    with pytest.raises(RuntimeError, match="no matching"):
        recorder.record_post_step()
    recorder.record_pre_step()
    with pytest.raises(RuntimeError, match="Consume"):
        recorder.record_pre_step()
    recorder.record_post_step()
    with pytest.raises(RuntimeError, match="Consume"):
        recorder.record_pre_step()
    recorder.take_transition()
    recorder.record_pre_step()
    recorder.record_post_step()
    assert recorder.take_transition().action.shape == (2, 29)
