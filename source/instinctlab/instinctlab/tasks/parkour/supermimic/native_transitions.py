"""Capture native Hiking transitions before Same-Step autoreset mutates buffers.

The native manager calls record_post_step after rewards/terminations and before
reset. No observations or HDF5 episodes are stored by this recorder.
"""

import torch
from dataclasses import dataclass

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.managers.recorder_manager import DatasetExportMode, RecorderManagerBaseCfg
from isaaclab.utils import configclass

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1


@dataclass
class NativeTransition:
    state: torch.Tensor
    next_state: torch.Tensor
    action: torch.Tensor
    previous_action: torch.Tensor
    episode_step: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    rewards: torch.Tensor | dict[str, torch.Tensor]
    force_history: torch.Tensor
    normal_force_history: torch.Tensor
    air_time: torch.Tensor
    contact_time: torch.Tensor
    applied_torque: torch.Tensor
    joint_acceleration: torch.Tensor


class HikingTransitionRecorder(RecorderTerm):
    """One consumable transition per control step, never a post-reset target.

    State has 71 values: native root-link position, wxyz quaternion, world
    link-origin linear/angular velocity, then policy-order joint positions and
    velocities. The root is the native URDF root, NOT implicitly MJ's pelvis.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.robot = env.scene["robot"]
        self.joint_ids, self.joint_names = self.robot.find_joints(
            list(G1_29DOF_POLICY_JOINT_ORDER_V1), preserve_order=True
        )
        if list(self.joint_names) != list(G1_29DOF_POLICY_JOINT_ORDER_V1):
            raise ValueError("Native transition joint order differs from policy order.")
        self.pending = None
        self.transition = None

    def state(self):
        d = self.robot.data
        return (
            torch.cat(
                (
                    d.root_link_pos_w.torch,
                    d.root_link_quat_w.torch[:, [3, 0, 1, 2]],
                    d.root_link_lin_vel_w.torch,
                    d.root_link_ang_vel_w.torch,
                    d.joint_pos.torch[:, self.joint_ids],
                    d.joint_vel.torch[:, self.joint_ids],
                ),
                -1,
            )
            .detach()
            .clone()
        )

    def record_pre_step(self):
        if self.pending is not None or self.transition is not None:
            raise RuntimeError("Consume the previous native transition before stepping again.")
        manager = self._env.action_manager
        self.pending = (
            self.state(),
            manager.action.detach().clone(),
            manager.prev_action.detach().clone(),
            self._env.episode_length_buf.detach().clone(),
        )
        return None, None

    def record_post_step(self):
        if self.pending is None:
            raise RuntimeError("Native post-step capture has no matching pre-step.")
        env = self._env
        state, action, previous_action, age = self.pending
        rewards = env.reward_buf
        rewards = (
            {name: value.detach().clone() for name, value in rewards.items()}
            if isinstance(rewards, dict)
            else rewards.detach().clone()
        )
        sensor = env.scene["contact_forces"].data
        self.transition = NativeTransition(
            state=state,
            next_state=self.state(),
            action=action,
            previous_action=previous_action,
            episode_step=age,
            terminated=env.reset_terminated.detach().clone(),
            truncated=env.reset_time_outs.detach().clone(),
            rewards=rewards,
            force_history=sensor.net_forces_w_history.torch.detach().clone(),
            normal_force_history=sensor.net_normal_forces_w_history.torch.detach().clone(),
            air_time=sensor.current_air_time.torch.detach().clone(),
            contact_time=sensor.current_contact_time.torch.detach().clone(),
            applied_torque=self.robot.data.applied_torque.torch[:, self.joint_ids].detach().clone(),
            joint_acceleration=self.robot.data.joint_acc.torch[:, self.joint_ids].detach().clone(),
        )
        self.pending = None
        return None, None

    def take_transition(self):
        if self.transition is None:
            raise RuntimeError("No completed native transition is available.")
        result, self.transition = self.transition, None
        return result


@configclass
class NativeHikingRecorderCfg(RecorderManagerBaseCfg):
    dataset_export_mode = DatasetExportMode.EXPORT_NONE
    export_in_record_pre_reset = False
    transition = RecorderTermCfg(class_type=HikingTransitionRecorder)
