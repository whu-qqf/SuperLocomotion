"""Native Hiking state/context reader and the shared physical SuperMimic F.

Visual features condition F; predicting their increments is a legacy option.
Raw images and their reconstruction loss are owned by the visual runner.
Native root-link coordinates and policy joint order come from the pre-reset
recorder. Depth-only F never samples the privileged terrain raster.
"""

import hashlib
import torch
import torch.nn.functional as F
from pathlib import Path

from instinctlab_supermimic.networks import WorldModel
from instinctlab_supermimic.physical import flatten_state, unflatten_state

from isaaclab.actuators import ImplicitActuator

from .materials import NewtonShapeMaterials
from .stair_kinematics import NativeLinkKinematics


def shoe_sole_points(state, kinematics):
    """World shoe-underside proxies shared by training and measured evaluation."""
    position, rotation, _, _, joints, _ = state
    transforms = kinematics.chain.forward_kinematics(joints[:, kinematics.order])
    links = torch.stack([transforms[name].get_matrix() for name in kinematics.tracked_bodies], 1)
    points = joints.new_tensor([[x, y, -0.058] for x in (0.132, -0.054) for y in (-0.018, 0.018)])
    local = torch.einsum("bfij,pj->bfpi", links[..., :3, :3], points) + links[..., None, :3, 3]
    return torch.einsum("bij,bfpj->bfpi", rotation, local) + position[:, None, None, :]


class NativePhysics:
    def __init__(
        self,
        env,
        recorder,
        terrain,
        hidden=(2048, 2048, 2048),
        foot_loss_weight=0.0,
        z_dim=0,
        model_heightmap=True,
        predict_latent=True,
        terrain_features=False,
        mean_contact_loss=False,
    ):
        if foot_loss_weight < 0:
            raise ValueError("Foot supervision weight must be nonnegative.")
        self.foot_loss_weight = foot_loss_weight
        self.terrain_features = terrain_features
        self.mean_contact_loss = mean_contact_loss
        self.env, self.robot, self.recorder, self.terrain = env, env.scene["robot"], recorder, terrain
        if self.robot.newton_actuator_adapter is not None or not all(
            isinstance(actuator, ImplicitActuator) for actuator in self.robot.actuators.values()
        ):
            raise ValueError("Native physical integration currently requires implicit PD without a delayed queue.")
        self.joint_ids = recorder.joint_ids
        self.materials = NewtonShapeMaterials(env, self.robot)
        self.action_term = env.action_manager.get_term("joint_pos")
        self.context_names = (
            "previous_action",
            "action_scale",
            "action_offset",
            "joint_stiffness",
            "joint_damping",
            "body_mass",
            "body_inertia",
            "body_com_pos_b",
            "shape_friction",
            "shape_restitution",
        )
        values = self.context_parts(env.action_manager.action)
        self.context_widths = [value.shape[-1] for value in values]
        sensor = env.scene["contact_forces"]
        self.contact_body_names = list(sensor.body_names)
        contact_count = sensor.data.net_forces_w_history.torch.shape[2]
        if contact_count != len(self.contact_body_names):
            raise ValueError("Native contact labels differ from the sensor force-history axis.")
        self.foot_contact_ids = [self.contact_body_names.index(f"{side}_ankle_roll_link") for side in ("left", "right")]
        self.contact_target_names = (
            [f"total_history/{name}" for name in self.contact_body_names]
            + [f"normal_history/{name}" for name in self.contact_body_names]
            + [f"instantaneous/{self.contact_body_names[i]}" for i in self.foot_contact_ids]
        )
        self.model = WorldModel(
            68 if terrain_features else len(terrain.local_xy) if model_heightmap else 0,
            len(self.contact_target_names),
            sum(self.context_widths),
            z_dim=z_dim,
            hidden=hidden,
            aux_dim=58,
            predict_latent=predict_latent,
        ).to(env.device)
        self.foot_kinematics = NativeLinkKinematics(
            Path(self.robot.cfg.spawn.asset_path),
            env.device,
            ("left_ankle_roll_link", "right_ankle_roll_link"),
        )

        self.auxiliary_names = ("applied_torque", "last_substep_joint_acceleration")
        self.auxiliary_floor = torch.cat((torch.ones(29), torch.full((29,), 0.05 / env.physics_dt))).to(env.device)
        self.error_floor = torch.cat(
            (
                torch.full((3,), 3e-3),
                torch.full((3,), 5e-3),
                torch.full((3,), 2e-2),
                torch.full((3,), 5e-2),
                torch.full((29,), 3e-3),
                torch.full((29,), 5e-2),
            )
        ).to(env.device)

    def checkpoint_contract(self):
        """Old real transitions may only be replayed against the same terrain."""
        digest = hashlib.sha256()
        for name in ("heights", "valid", "origin_xy", "local_xy"):
            value = getattr(self.terrain, name).detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.numpy().tobytes())
        return dict(
            root_frame="native_root_link",
            quaternion_order="wxyz",
            joint_names=list(self.recorder.joint_names),
            urdf_sha256=hashlib.sha256(Path(self.robot.cfg.spawn.asset_path).read_bytes()).hexdigest(),
            contact_targets=self.contact_target_names,
            context_names=self.context_names,
            context_widths=self.context_widths,
            auxiliary_names=self.auxiliary_names,
            terrain_resolution=self.terrain.resolution,
            terrain_sha256=digest.hexdigest(),
            world_model_heightmap=self.model.scan_dim > 0,
        )

    def context_parts(self, previous_action):
        d, ids = self.robot.data, self.joint_ids
        scale = torch.as_tensor(self.action_term._scale, device=previous_action.device).expand_as(previous_action)
        offset = torch.as_tensor(self.action_term._offset, device=previous_action.device).expand_as(previous_action)
        friction, restitution = self.materials.values()
        return (
            previous_action,
            scale,
            offset,
            d.joint_stiffness.torch[:, ids],
            d.joint_damping.torch[:, ids],
            d.body_mass.torch.flatten(1),
            d.body_inertia.torch.flatten(1),
            d.body_com_pos_b.torch.flatten(1),
            friction,
            restitution,
        )

    def context(self, previous_action):
        """Call before env.step: reset may change action offsets or parameters."""
        return torch.cat(self.context_parts(previous_action), -1).detach().clone()

    def model_scan(self, state):
        """Visual F gets no privileged height features; critic keeps scan()."""
        if getattr(self, "terrain_features", False):
            features, valid = self.terrain_features_with_validity(state)
            if not valid.all():
                raise ValueError("Terrain model query left the supported native raster.")
            return features
        if self.model.scan_dim == 0:
            return state[0].new_empty((*state[0].shape[:-1], 0))
        return self.scan(state)

    def terrain_features_with_validity(self, state):
        """TGI base/sole layout, adapted to the native G1 shoe geometry (68 values)."""
        position, rotation, _, _, joints, _ = state
        yaw = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        base = position.new_tensor(
            [[x, y] for x in (-0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9) for y in (-0.4, -0.2, 0.0, 0.2, 0.4)]
        )
        x, y = base.unbind(-1)
        base_xy = position[:, None, :2] + torch.stack((c * x - s * y, s * x + c * y), -1)
        base_h, base_valid = self.terrain.query(base_xy)
        center_h, center_valid = self.terrain.query(position[:, None, :2])
        transforms = self.foot_kinematics.chain.forward_kinematics(joints[:, self.foot_kinematics.order])
        links = torch.stack([transforms[name].get_matrix() for name in self.foot_kinematics.tracked_bodies], 1)
        sole = position.new_tensor(
            [[x, y, -0.058] for x in (-0.054, -0.0075, 0.039, 0.0855, 0.132) for y in (-0.018, 0.0, 0.018)]
        )
        local = torch.einsum("bfij,pj->bfpi", links[..., :3, :3], sole) + links[..., None, :3, 3]
        points = torch.einsum("bij,bfpj->bfpi", rotation, local) + position[:, None, None, :]
        foot_h, foot_valid = self.terrain.query(points[..., :2])
        center = points[:, :, 7]
        foot_center_h = foot_h[:, :, 7:8]
        feet = torch.cat((center[..., 2:3] - foot_center_h, foot_h - foot_center_h), -1).flatten(1)
        features = torch.cat((position[:, 2:3] - center_h, base_h - center_h, feet), -1)
        valid = center_valid.all(-1) & base_valid.all(-1) & foot_valid.flatten(1).all(-1)
        return features, valid

    def scan(self, state):
        """Heading-aligned privileged terrain input, differentiable in position."""
        height, valid = self.scan_with_validity(state)
        if not valid.all():
            raise ValueError("Physical rollout left the supported native terrain raster.")
        return height

    def scan_with_validity(self, state):
        """Return per-environment support, without treating padding as terrain."""
        position, rotation = state[:2]
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        c, s = yaw.cos()[..., None], yaw.sin()[..., None]
        x, y = self.terrain.local_xy.unbind(-1)
        xy = position[..., None, :2] + torch.stack((c * x - s * y, s * x + c * y), -1)
        height, valid = self.terrain.query(xy)
        supported = valid.all(dim=-1)
        if getattr(self, "terrain_features", False):
            _, model_valid = self.terrain_features_with_validity(state)
            supported = supported & model_valid
        return (position[..., 2:3] - height).clamp(max=5.0), supported

    def contact_targets(self, transition):
        total = transition.force_history.norm(dim=-1).amax(dim=1) > 1.0
        normal = transition.normal_force_history.norm(dim=-1).amax(dim=1) > 1.0
        instantaneous = transition.contact_time[:, self.foot_contact_ids] > 0.0
        return torch.cat((total, normal, instantaneous), -1).float()

    @staticmethod
    def auxiliary_targets(transition):
        return torch.cat((transition.applied_torque, transition.joint_acceleration), -1)

    @torch.no_grad()
    def observe(self, transition, context):
        """Update statistics from true pre-reset transitions, never predictions."""
        state, next_state = unflatten_state(transition.state), unflatten_state(transition.next_state)
        self.model.in_norm.update(self.model.inputs(state, transition.action, context, self.model_scan(state)))
        self.model.out_norm.update(self.model.targets(state, next_state))
        self.model.aux_norm.update(self.auxiliary_targets(transition))

    def window_loss(
        self,
        state,
        next_state,
        actions,
        context,
        contacts,
        mask,
        auxiliary=None,
        latent=None,
        next_latent=None,
        latent_observation=None,
    ):
        """Multi-step physical regression; terminal targets remain supervised.

        Inputs are N,T,..., mask is true through the first terminal transition
        and false thereafter. Callers construct windows within one episode.
        """
        if state.ndim != 3 or mask.shape != state.shape[:2] or actions.shape[:2] != mask.shape:
            raise ValueError("Native physical windows must have matching N,T axes.")
        valid = mask.bool()
        if not valid.any() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Physical mask must be a nonempty valid prefix per window.")
        if self.model.aux_dim and (auxiliary is None or auxiliary.shape != (*mask.shape, self.model.aux_dim)):
            raise ValueError("Native physical readouts need matching torque/qacc supervision.")
        if self.model.z_dim and latent_observation is None and (latent is None or next_latent is None):
            raise ValueError("Visual dynamics require current-encoder latent targets.")
        current_z = None if latent is None else latent[:, 0].detach()
        current = unflatten_state(state[:, 0])
        scale = torch.maximum(self.model.out_norm.std(), self.error_floor)
        errors, contact_errors, auxiliary_errors, foot_errors, latent_errors = [], [], [], [], []
        for step in range(actions.shape[1]):
            active = valid[:, step]
            if not active.any():
                break
            # Remove terminated rows before evaluation. Invalid padding may
            # contain NaNs and must never enter the model, scan or normalizer.
            selected = tuple(value[active] for value in current)
            observation = (
                latent_observation(selected, active, step)
                if latent_observation is not None
                else None if current_z is None else current_z[active]
            )
            predicted, logits, predicted_z, readouts = self.model.step_with_aux(
                selected,
                actions[active, step],
                context[active, step],
                self.model_scan(selected),
                observation,
            )
            if current_z is not None:
                latent_errors.append(
                    ((predicted_z - next_latent[active, step].detach()) / self.model.z_std).square().mean(-1).sum()
                )
                current_z = current_z.index_copy(0, active.nonzero(as_tuple=True)[0], predicted_z)
            target = unflatten_state(next_state[active, step])
            error = (self.model.state_error(predicted, target) / scale).square().mean(-1)
            # Supervise the composed root/joint geometry, not only its separate
            # coordinates. Two centimetres is a spatial scale, independent of
            # the replay's joint-delta normalizer and contact class frequency.
            if self.foot_loss_weight > 0:
                foot_error = (
                    ((self.foot_points(predicted) - self.foot_points(target)) / 0.02).square().mean((-3, -2, -1))
                )
                foot_errors.append(foot_error.sum())
            bce = F.binary_cross_entropy_with_logits(logits, contacts[active, step], reduction="none").sum(-1)
            if getattr(self, "mean_contact_loss", False):
                bce = bce / logits.shape[-1]
            errors.append(error.sum())
            contact_errors.append(bce.sum())
            if self.model.aux_dim:
                aux_scale = torch.maximum(self.model.aux_norm.std(), self.auxiliary_floor)
                auxiliary_errors.append(((readouts - auxiliary[active, step]) / aux_scale).square().mean(-1).sum())
            # Functional scatter keeps recurrent action gradients intact.
            ids = active.nonzero(as_tuple=True)[0]
            current = tuple(old.index_copy(0, ids, new) for old, new in zip(current, predicted))
        physical = torch.stack(errors).sum() / state.shape[0]
        contact = torch.stack(contact_errors).sum() / state.shape[0]
        aux_loss = torch.stack(auxiliary_errors).sum() / state.shape[0] if auxiliary_errors else physical.new_zeros(())
        feet = torch.stack(foot_errors).sum() / state.shape[0] if foot_errors else physical.new_zeros(())
        metrics = dict(
            physical=physical.detach(),
            contact=contact.detach(),
            auxiliary=aux_loss.detach(),
            valid=int(valid.sum()),
        )
        latent_loss = torch.stack(latent_errors).sum() / state.shape[0] if latent_errors else physical.new_zeros(())
        if latent_errors:
            metrics["latent"] = latent_loss.detach()
        if foot_errors:
            metrics["feet"] = feet.detach()
        contact_weight = 0.5 if getattr(self, "mean_contact_loss", False) else 0.1
        return physical + contact_weight * contact + aux_loss + self.foot_loss_weight * feet + latent_loss, metrics

    def foot_points(self, state):
        """World shoe underside proxies; root is the native URDF link origin.

        Front/rear points constrain toe/heel pitch as well as ankle position.
        These are geometric training readouts, not exact collision distances.
        """
        return shoe_sole_points(state, self.foot_kinematics)

    def rollout(self, initial_state, actions, initial_context):
        """Open-loop physical states; previous action advances with the rollout."""
        state, context = unflatten_state(initial_state), initial_context
        predicted = []
        for step in range(actions.shape[1]):
            action = actions[:, step]
            state, _, _ = self.model.step(state, action, context, self.model_scan(state))
            predicted.append(flatten_state(state))
            context = torch.cat((action, context[..., 29:]), -1)
        return torch.stack(predicted, 1)
