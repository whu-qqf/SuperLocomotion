"""Native link poses and origin twists derived from root and policy-order joints."""

import torch
import xml.etree.ElementTree as ET
from pathlib import Path

import pytorch_kinematics as pk

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_plane_cfg import TRACKED_BODIES
from instinctlab.tasks.shadowing.supermimic.world_model_frames import _mul, _rotate


def matrix_quaternion(matrix):
    """Best-conditioned quaternion branch; unused square roots have finite gradients."""
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.flatten(-2).unbind(-1)
    diagonal = torch.stack((1 + m00 + m11 + m22, 1 + m00 - m11 - m22, 1 - m00 + m11 - m22, 1 - m00 - m11 + m22), -1)
    magnitude = diagonal.clamp_min(1e-8).sqrt()
    candidates = torch.stack(
        (
            torch.stack((diagonal[..., 0], m21 - m12, m02 - m20, m10 - m01), -1),
            torch.stack((m21 - m12, diagonal[..., 1], m10 + m01, m02 + m20), -1),
            torch.stack((m02 - m20, m10 + m01, diagonal[..., 2], m12 + m21), -1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, diagonal[..., 3]), -1),
        ),
        -2,
    ) / (2 * magnitude[..., None].clamp_min(0.1))
    index = magnitude.argmax(-1)[..., None, None].expand(*magnitude.shape[:-1], 1, 4)
    selected = candidates.gather(-2, index).squeeze(-2)
    return torch.cat((selected[..., 1:], selected[..., :1]), -1)


class NativeLinkKinematics:
    """Analytic revolute-joint Jacobians at link origins, with world-frame twists."""

    def __init__(self, urdf: Path, device, tracked_bodies=TRACKED_BODIES):
        self.tracked_bodies = tuple(tracked_bodies)
        text = urdf.read_text()
        self.chain = pk.build_chain_from_urdf(text).to(device=device, dtype=torch.float32)
        robot = ET.fromstring(text)
        joints = {joint.attrib["name"]: joint for joint in robot.findall("joint")}
        origin = robot.find(f"link[@name='{self.chain._root.name}']/inertial/origin")
        self.root_com = torch.tensor([float(x) for x in origin.attrib["xyz"].split()], device=device)
        parents = {joint.find("child").attrib["link"]: joint for joint in joints.values()}
        names = self.chain.get_joint_parameter_names()
        if any(joints[name].attrib["type"] not in ("revolute", "continuous") for name in names):
            raise ValueError("native link projection requires revolute joints")
        self.order = torch.tensor(
            [list(G1_29DOF_POLICY_JOINT_ORDER_V1).index(name) for name in names], device=device, dtype=torch.long
        )
        self.child_names = [joints[name].find("child").attrib["link"] for name in names]
        self.tracked_frame_indices = self.chain.get_frame_indices(*self.tracked_bodies)
        self.child_frame_indices = self.chain.get_frame_indices(*self.child_names)
        self.axes = torch.tensor(
            [[float(x) for x in joints[name].find("axis").attrib["xyz"].split()] for name in names],
            device=device,
        )
        ancestors = []
        for name in self.tracked_bodies:
            active = set()
            while name in parents:
                joint = parents[name]
                active.add(joint.attrib["name"])
                name = joint.find("parent").attrib["link"]
            ancestors.append([name in active for name in names])
        self.ancestors = torch.tensor(ancestors, device=device, dtype=torch.float32)

    def links(self, parts, root_com=None):
        joints = parts["joint_pos"]
        lead = joints.shape[:-1]
        # The dictionary FK API calls GPU scalar .item(); Python-list indexing
        # also creates host tensors. Keep the entire rollout on the device so
        # both forward and backward can be captured in a CUDA graph.
        transforms = self.chain.forward_kinematics_tensor(joints.reshape(-1, joints.shape[-1])[:, self.order])
        link = transforms.index_select(0, self.tracked_frame_indices).transpose(0, 1)
        child = transforms.index_select(0, self.child_frame_indices).transpose(0, 1)
        axes = torch.einsum("bjik,jk->bji", child[..., :3, :3], self.axes)
        rate_axes = axes * parts["joint_vel"].reshape(-1, joints.shape[-1])[:, self.order, None]
        offset = link[:, :, None, :3, 3] - child[:, None, :, :3, 3]
        linear = (
            torch.linalg.cross(rate_axes[:, None].expand_as(offset), offset) * self.ancestors[None, :, :, None]
        ).sum(2)
        angular = torch.einsum("bji,lj->bli", rate_axes, self.ancestors)
        local_pos = link[..., :3, 3].reshape(*lead, len(self.tracked_bodies), 3)
        local_quat = matrix_quaternion(link[..., :3, :3])
        local_quat = local_quat.reshape(*lead, len(self.tracked_bodies), 4)
        root_quat = parts["root_quat_w"][..., None, :].expand(*local_quat.shape)
        displacement = _rotate(root_quat, local_pos)
        root_angular = parts["root_ang_vel_w"][..., None, :].expand_as(displacement)
        com = self.root_com if root_com is None else root_com
        com = _rotate(parts["root_quat_w"], com.expand_as(parts["root_pos_w"]))
        origin_velocity = parts["root_lin_vel_w"] - torch.linalg.cross(parts["root_ang_vel_w"], com)
        return {
            "body_link_pos_w": parts["root_pos_w"][..., None, :] + displacement,
            "body_link_quat_w": _mul(root_quat, local_quat),
            "body_link_lin_vel_w": (
                origin_velocity[..., None, :]
                + torch.linalg.cross(root_angular, displacement)
                + _rotate(root_quat, linear.reshape_as(displacement))
            ),
            "body_link_ang_vel_w": root_angular + _rotate(root_quat, angular.reshape_as(displacement)),
        }

    def project(self, state, parameterization, root_com=None):
        parts = parameterization._unpack(state)
        parts.update(self.links(parts, root_com))
        return parameterization._pack(parts)
