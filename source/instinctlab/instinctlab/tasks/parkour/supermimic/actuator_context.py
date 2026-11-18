"""Read actual actuator delay semantics in the pinned IsaacLab snapshot."""

import torch

import warp as wp

from isaaclab.actuators import ImplicitActuator


class HikingActuatorContext:
    """Read implicit PD or project native delay/queue age into policy joint order.

    The shoe task uses implicit PD, with no command delay. Explicit PD may use
    Newton execution under PhysX; those ``robot.actuators`` entries are Newton
    controllers, not DelayedPDActuator.
    Queue state is read from the bridge's current ping-pong state, never guessed
    from the episode length. This snapshot-specific access is isolated here.
    """

    def __init__(self, robot, joint_order):
        self.robot = robot
        self.adapter = robot.newton_actuator_adapter
        self.policy_ids = torch.tensor([robot.joint_names.index(n) for n in joint_order], device=robot.device)
        self.shape = (robot.num_instances, robot.num_joints)
        caps = torch.zeros(self.shape, device=robot.device)
        if self.adapter is None:
            if not all(isinstance(a, ImplicitActuator) for a in robot.actuators.values()):
                raise TypeError("unsupported actuator runtime: expected implicit or native PD")
            self.kind = "implicit_pd_no_delay"
            self.caps = caps[0, self.policy_ids]
            return
        self.kind = "native_pd"
        for actuator in self.adapter.actuators:
            if actuator.delay is not None:
                caps.flatten()[wp.to_torch(actuator.indices).long()] = actuator.delay.buf_depth
        self.caps = caps[0, self.policy_ids]

    def read(self):
        lag = torch.zeros(self.shape, device=self.robot.device)
        age = torch.zeros_like(lag)
        if self.adapter is None:
            return lag[:, self.policy_ids], age[:, self.policy_ids]
        for actuator, state in zip(self.adapter.actuators, self.adapter._states_a):
            if actuator.delay is None:
                continue
            indices = wp.to_torch(actuator.indices).long()
            lag.flatten()[indices] = wp.to_torch(actuator.delay.delay_steps).float()
            age.flatten()[indices] = wp.to_torch(state.delay_state.num_pushes).float()
        return lag[:, self.policy_ids], age[:, self.policy_ids].clamp_max(self.caps)
