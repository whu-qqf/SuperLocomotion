"""Native Hiking task readouts for physical imagination.

Reward functions and resolved entity selections come from the live reward
manager. A tensor-only view supplies their native data contract; no reward
scalar is learned. Hard sensor gates retain their native forward values.
"""

import torch
from pathlib import Path
from types import SimpleNamespace

import warp as wp
from instinctlab_supermimic.physical import invariant_features
from instinctlab_supermimic.rotations import mv, quat_to_mat, yaw_mat

from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.warp import ProxyArray, raycast_mesh

from .stair_kinematics import NativeLinkKinematics, matrix_quaternion

POLICY_TERMS = ("base_ang_vel", "projected_gravity", "velocity_commands", "joint_pos", "joint_vel", "actions")
POLICY_WIDTHS = (3, 3, 3, 29, 29, 29)
AMP_TERMS = ("projected_gravity", "joint_pos_rel", "joint_vel", "base_lin_vel", "base_ang_vel")
AMP_WIDTHS = (3, 29, 29, 3, 3)
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def tensor_view(value):
    """Native MDP functions use ProxyArray.torch; keep the autograd tensor."""
    return SimpleNamespace(torch=value)


def flatten_history(frames, widths):
    """Oldest-first frames to native term-major flattened histories."""
    if frames.ndim != 3 or frames.shape[-1] != sum(widths):
        raise ValueError("Native observation history has incompatible frame dimensions.")
    return torch.cat([term.flatten(1) for term in frames.split(widths, -1)], -1)


def single_stance_time(timers):
    """Native value for exclusive timers; continuous relaxation for imagination.

    The force-history contact indicator cannot define these modes because it
    can remain positive after liftoff. Reset (all-zero) timers contribute zero.
    """
    air, contact = timers[..., :2], timers[..., 2:]
    mode = contact / (air + contact).clamp_min(1e-8)
    single = mode[..., 0] * (1 - mode[..., 1]) + mode[..., 1] * (1 - mode[..., 0])
    return single * (mode * contact + (1 - mode) * air).amin(-1)


class NativeTaskHeads:
    def __init__(self, env, physics):
        self.env, self.physics = env, physics
        self.robot = env.scene["robot"]
        self.joint_ids = physics.joint_ids
        self.native_order = torch.argsort(torch.as_tensor(self.joint_ids, device=env.device))
        self.body_names = list(self.robot.body_names)
        urdf = Path(self.robot.cfg.spawn.asset_path)
        self.kin = NativeLinkKinematics(urdf, env.device, self.body_names)
        self.root_id = self.body_names.index(self.kin.chain._root.name)
        self.foot_ids = [self.body_names.index(name) for name in FEET]
        self.foot_contact_ids = [physics.contact_body_names.index(name) for name in FEET]
        self.dt = env.step_dt
        if getattr(env.cfg, "terrain_locomotion", False):
            physics.terrain.build_support_field()
        self.command_term = env.command_manager.get_term("base_velocity")
        if not self.command_term.cfg.only_positive_lin_vel_x:
            raise ValueError("Native Hiking task heads require only_positive_lin_vel_x=True.")
        self.reward_cfg = {
            name: env.reward_manager.get_term_cfg(name, "rewards")
            for name in env.reward_manager.active_terms["rewards"]
        }
        self.termination_cfg = {
            name: env.termination_manager.get_term_cfg(name)
            for name in env.termination_manager.active_terms
            if not env.termination_manager.get_term_cfg(name).time_out
        }
        if set(self.termination_cfg) != {"base_contact", "bad_orientation", "root_height"}:
            raise ValueError("Native task heads require the three original Hiking physical terminations.")
        if self.termination_cfg["base_contact"].params["threshold"] != 1.0:
            raise ValueError("Native F normal-history contact labels use a 1 N threshold.")
        if getattr(env.cfg.rewards.rewards, "combine_method", "sum") != "sum":
            raise ValueError("Native task heads implement Hiking's sum reward group.")
        self.policy_scale, self.policy_noise = self._scales(env.cfg.observations.policy, POLICY_TERMS, POLICY_WIDTHS)
        self.amp_scale, _ = self._scales(env.cfg.observations.amp_policy, AMP_TERMS, AMP_WIDTHS)
        self.vp_sensor = env.scene[self.reward_cfg["volume_points_penetration"].params["sensor_cfg"].name]
        self.vp_ids = [self.body_names.index(name) for name in self.vp_sensor.body_names]
        self.scanners = [env.scene[name] for name in ("left_height_scanner", "right_height_scanner")]
        self.foot_meshes = []
        for sensor in self.scanners:
            if sensor.cfg.ray_alignment != "yaw":
                raise ValueError("Native feet task readout requires yaw-aligned scanners.")
            if len(sensor.cfg.mesh_prim_paths) != 1:
                raise ValueError("Native feet task readout requires a single static terrain mesh.")
            key = (sensor.cfg.mesh_prim_paths[0], sensor.device)
            if key not in sensor.meshes:
                raise ValueError("Native Hiking feet scanners must use the ground-only LegacyRayCaster preset.")
            self.foot_meshes.append(sensor.meshes[key])

    def _scales(self, group, names, widths):
        scale, noise = [], []
        for name, width in zip(names, widths):
            term = getattr(group, name)
            s = 1.0 if term.scale is None else term.scale
            scale.append(torch.as_tensor(s, device=self.env.device).expand(width))
            n = term.noise
            if n is not None and n.n_min != -n.n_max:
                raise ValueError("Native Hiking observation readout requires symmetric uniform noise.")
            noise.append(torch.as_tensor(0.0 if n is None else n.n_max, device=self.env.device).expand(width))
        scale = torch.cat(scale)
        return scale, torch.cat(noise) * scale

    @torch.no_grad()
    def context(self):
        """Snapshot per-row native constants before stepping/reset/resampling.

        Joint fields are in native articulation order. Body COMs are local to
        their own links, not subtree or whole-robot COMs.
        """
        data = self.robot.data
        names = (
            "default_joint_pos",
            "default_joint_vel",
            "soft_joint_pos_limits",
            "soft_joint_vel_limits",
            "joint_effort_limits",
            "joint_stiffness",
            "body_com_pos_b",
            "GRAVITY_VEC_W",
        )
        context = {name: getattr(data, name).torch.detach().clone() for name in names}
        if "goal_progress" in self.reward_cfg:
            c = self.command_term
            context.update(
                goal_target=c.pos_command_w.detach().clone(),
                progress_start_position=c.progress_start_pos_w.detach().clone(),
                goal_standing=c.is_standing_env.detach().clone(),
                goal_random=c.random_velocity_indices.detach().clone(),
            )
        return context

    @torch.no_grad()
    def command(self):
        """Latched command and its generating parameters, before env.step()."""
        c = self.command_term
        return dict(
            target=c.pos_command_w.detach().clone(),
            maximum=c.max_command_b.detach().clone(),
            standing=c.is_standing_env.detach().clone(),
            random=c.random_velocity_indices.detach().clone(),
            random_velocity=torch.stack((c.random_lin_vel_x, c.random_lin_vel_y, c.random_ang_vel_z), -1)
            .detach()
            .clone(),
            velocity=c.command.detach().clone(),
            time_left=c.time_left.detach().clone(),
            origin=self.env.scene.env_origins.detach().clone(),
        )

    def velocity_command(self, state, command):
        """Original native pose-to-velocity rule, without target resampling."""
        p, rotation = state[:2]
        c = self.command_term.cfg
        target = command["target"] - p
        local = mv(yaw_mat(rotation).transpose(-1, -2), target)
        vx = (local[..., 0] * c.velocity_control_stiffness).clamp(min=0)
        vx = torch.minimum(vx, command["maximum"][..., 0])
        vy = local[..., 1] * c.velocity_control_stiffness
        vy = torch.maximum(torch.minimum(vy, command["maximum"][..., 1]), -command["maximum"][..., 1])
        yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
        wz = (wrap_to_pi(torch.atan2(target[..., 1], target[..., 0]) - yaw) * c.heading_control_stiffness).clamp(
            *c.ranges.ang_vel_z
        )
        velocity = torch.stack((vx, vy, wz), -1)
        velocity = velocity * (torch.linalg.vector_norm(target[..., :2], dim=-1) > c.target_dis_threshold)[..., None]
        linear = (
            velocity[..., :2] * (torch.linalg.vector_norm(velocity[..., :2], dim=-1) > c.lin_vel_threshold)[..., None]
        )
        angular = velocity[..., 2:] * (velocity[..., 2:].abs() > c.ang_vel_threshold)
        velocity = torch.cat((linear, angular), -1)
        velocity = torch.where(command["standing"][..., None], torch.zeros_like(velocity), velocity)
        return torch.where(command["random"][..., None], command["random_velocity"], velocity)

    def data(self, state, context, auxiliary=None):
        """Native root/link poses and COM velocities from shared 71-dim state."""
        p, rotation, v, w, q, qd = state
        parts = dict(
            root_pos_w=p,
            root_quat_w=matrix_quaternion(rotation),
            root_lin_vel_w=v,
            root_ang_vel_w=w,
            joint_pos=q,
            joint_vel=qd,
        )
        links = self.kin.links(parts, root_com=torch.zeros_like(p))
        body_rotation = quat_to_mat(links["body_link_quat_w"][..., [3, 0, 1, 2]])
        body_com_shift = mv(body_rotation, context["body_com_pos_b"])
        body_velocity = links["body_link_lin_vel_w"] + torch.linalg.cross(
            links["body_link_ang_vel_w"], body_com_shift, dim=-1
        )
        root_com_shift = mv(rotation, context["body_com_pos_b"][..., self.root_id, :])
        root_velocity = v + torch.linalg.cross(w, root_com_shift, dim=-1)
        data = dict(context)
        data.update(
            root_pos_w=p,
            root_quat_w=parts["root_quat_w"],
            root_lin_vel_b=mv(rotation.transpose(-1, -2), root_velocity),
            root_ang_vel_b=mv(rotation.transpose(-1, -2), w),
            projected_gravity_b=mv(
                rotation.transpose(-1, -2), torch.nn.functional.normalize(context["GRAVITY_VEC_W"], dim=-1)
            ),
            heading_w=torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0]),
            joint_pos=q[..., self.native_order],
            joint_vel=qd[..., self.native_order],
            body_pos_w=links["body_link_pos_w"],
            body_quat_w=links["body_link_quat_w"],
            body_lin_vel_w=body_velocity,
            body_ang_vel_w=links["body_link_ang_vel_w"],
            **links,
        )
        if auxiliary is not None:
            if auxiliary.shape[-1] != 58:
                raise ValueError("Native reward readouts require 29 torques and 29 last-substep accelerations.")
            torque, acceleration = auxiliary.split(29, -1)
            data.update(applied_torque=torque[..., self.native_order], joint_acc=acceleration[..., self.native_order])
        return SimpleNamespace(**{name: tensor_view(value) for name, value in data.items()})

    def proprio(self, state, previous_action, velocity, context):
        d = self.data(state, context)
        frame = torch.cat(
            (
                d.root_ang_vel_b.torch,
                d.projected_gravity_b.torch,
                velocity,
                (d.joint_pos.torch - context["default_joint_pos"])[..., self.joint_ids],
                (d.joint_vel.torch - context["default_joint_vel"])[..., self.joint_ids],
                previous_action,
            ),
            -1,
        )
        return frame * self.policy_scale

    def amp_frame(self, state, context):
        d = self.data(state, context)
        return (
            torch.cat(
                (
                    d.projected_gravity_b.torch,
                    (d.joint_pos.torch - context["default_joint_pos"])[..., self.joint_ids],
                    (d.joint_vel.torch - context["default_joint_vel"])[..., self.joint_ids],
                    d.root_lin_vel_b.torch,
                    d.root_ang_vel_b.torch,
                ),
                -1,
            )
            * self.amp_scale
        )

    def advance_feet_timers(self, timers, instantaneous_contact):
        """Instantaneous contact is distinct from the history contact head."""
        air, contact = timers[..., :2], timers[..., 2:]
        return torch.cat(
            ((air + self.dt) * (1 - instantaneous_contact), (contact + self.dt) * instantaneous_contact), -1
        )

    @torch.compiler.disable
    def foot_heights(self, data, sensor_context):
        """Original mesh hits and local face-plane derivatives (no raster).

        The initialized ray starts already include the native sensor offset.
        Newton's site frame has no additional cfg offset applied to its pose.
        Drift is snapshotted; rewards query the end-of-control state, matching
        native lazy sensor refresh. Face changes remain nondifferentiable.
        """
        heights = []
        rotations = quat_to_mat(data.body_quat_w.torch[..., self.foot_ids, :][..., [3, 0, 1, 2]])
        for i, sensor in enumerate(self.scanners):
            spec = sensor_context[i]
            rotation = yaw_mat(rotations[..., i, :, :])
            position = data.body_pos_w.torch[..., self.foot_ids[i], :] + spec["drift"]
            drift = mv(rotation, spec["ray_cast_drift"])
            position = position + torch.cat((drift[..., :2], torch.zeros_like(drift[..., 2:])), -1)
            starts = position[..., None, :] + mv(rotation[..., None, :, :], spec["starts"])
            directions = spec["directions"]
            hits, _, normals, _ = raycast_mesh(
                starts.detach(),
                directions.detach(),
                self.foot_meshes[i],
                max_dist=sensor.cfg.max_distance,
                return_normal=True,
            )
            valid = torch.isfinite(hits).all(-1)
            normals = torch.where(valid[..., None], normals, torch.zeros_like(normals))
            denominator = (normals * directions).sum(-1)
            supported = valid & (denominator.abs() > 1e-6)
            denominator = torch.where(supported, denominator, torch.ones_like(denominator))
            displacement = starts - starts.detach()
            dt = -(normals * displacement).sum(-1) / denominator
            dz = displacement[..., 2] + dt * directions[..., 2]
            z = torch.where(valid, hits[..., 2], torch.zeros_like(hits[..., 2]))
            z = z + torch.where(supported, dz, torch.zeros_like(dz)) + spec["ray_cast_drift"][..., 2:3]
            # Native feet_at_plane replaces misses by zero, including no drift.
            heights.append(torch.where(valid, z, torch.zeros_like(z)))
        return heights

    @torch.no_grad()
    def sensor_context(self):
        return [
            dict(
                starts=s.ray_starts.torch.detach().clone(),
                directions=s.ray_directions.torch.detach().clone(),
                drift=s.drift.torch.detach().clone(),
                ray_cast_drift=s.ray_cast_drift.torch.detach().clone(),
            )
            for s in self.scanners
        ]

    @torch.compiler.disable
    def volume_points(self, data):
        ids = self.vp_ids
        p = data.body_link_pos_w.torch[..., ids, :]
        rotation = quat_to_mat(data.body_link_quat_w.torch[..., ids, :][..., [3, 0, 1, 2]])
        offset = mv(rotation[..., None, :, :], self.vp_sensor._volume_points_pattern)
        points = p[..., None, :] + offset
        velocity = data.body_link_lin_vel_w.torch[..., ids, None, :] + torch.linalg.cross(
            data.body_link_ang_vel_w.torch[..., ids, None, :], offset, dim=-1
        )
        query = ProxyArray(wp.from_torch(points.detach().contiguous(), dtype=wp.vec3f))
        best = torch.zeros_like(points)
        depth = torch.zeros_like(points[..., 0])
        for obstacle in self.vp_sensor._virtual_obstacles.values():
            candidate = obstacle.get_points_penetration_offset(query).torch
            magnitude = torch.linalg.vector_norm(candidate, dim=-1)
            selected = magnitude > depth
            best = torch.where(selected[..., None], candidate, best)
            depth = torch.maximum(depth, magnitude)
        # Native offsets point inward toward the cylinder axis, i.e. grad(depth).
        direction = best / depth.clamp_min(1e-12)[..., None]
        projection = ((points - points.detach()) * direction).sum(-1)
        # Preserve the original vector-valued sensor contract and exact depth.
        penetration = best + direction * projection[..., None]
        return SimpleNamespace(penetration_offset=tensor_view(penetration), points_vel_w=tensor_view(velocity))

    def _reward_view(
        self,
        state,
        action,
        previous_action,
        velocity,
        context,
        auxiliary,
        force_history,
        normal_force_history,
        air_time,
        contact_time,
        terminated,
        sensor_context,
        previous_position=None,
    ):
        data = self.data(state, context, auxiliary)
        scene = {
            "robot": SimpleNamespace(
                data=data, actuators=NativeTaskActuators(context["joint_stiffness"], data.applied_torque)
            )
        }
        scene["contact_forces"] = SimpleNamespace(
            data=SimpleNamespace(
                net_forces_w_history=tensor_view(force_history),
                net_normal_forces_w_history=tensor_view(normal_force_history),
                current_air_time=tensor_view(air_time),
                current_contact_time=tensor_view(contact_time),
            )
        )
        heights = self.foot_heights(data, sensor_context)
        for name, z in zip(("left_height_scanner", "right_height_scanner"), heights):
            hits = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z), -1)
            scene[name] = SimpleNamespace(data=SimpleNamespace(ray_hits_w=tensor_view(hits)))
        scene[self.reward_cfg["volume_points_penetration"].params["sensor_cfg"].name] = SimpleNamespace(
            data=self.volume_points(data)
        )
        return SimpleNamespace(
            scene=NativeTaskScene(scene),
            step_dt=self.dt,
            action_manager=SimpleNamespace(action=action, prev_action=previous_action),
            command_manager=SimpleNamespace(
                get_command=lambda name: velocity,
                get_term=lambda name: SimpleNamespace(
                    pos_command_w=context["goal_target"],
                    progress_start_pos_w=(
                        context["progress_start_position"] if previous_position is None else previous_position
                    ),
                    is_standing_env=context["goal_standing"],
                    random_velocity_indices=context["goal_random"],
                ),
            ),
            termination_manager=SimpleNamespace(terminated=terminated),
        )

    def _weighted_rewards(self, view):
        return {name: cfg.func(view, **cfg.params) * cfg.weight * self.dt for name, cfg in self.reward_cfg.items()}

    def reward(
        self,
        state,
        action,
        previous_action,
        velocity,
        context,
        auxiliary,
        force_history,
        normal_force_history,
        air_time,
        contact_time,
        terminated,
        sensor_context,
    ):
        """Exact native hard-gate rewards on pre-reset real physical readouts."""
        if getattr(self.env.cfg, "tgi_aligned", False):
            from .native_rewards import tgi_core_reward

            return tgi_core_reward(self, state, action, previous_action, velocity, context)
        view = self._reward_view(
            state,
            action,
            previous_action,
            velocity,
            context,
            auxiliary,
            force_history,
            normal_force_history,
            air_time,
            contact_time,
            terminated,
            sensor_context,
        )
        terms = self._weighted_rewards(view)
        return sum(terms.values()), terms

    def soft_reward(
        self,
        state,
        action,
        previous_action,
        velocity,
        context,
        auxiliary,
        contacts,
        timers,
        termination_probability,
        sensor_context,
        previous_position=None,
    ):
        """Same values at hard contacts, with continuous contact-gate relaxation.

        The F contact layout is total-history B, normal-history B, then two
        instantaneous foot modes. Only the last two advance timers, outside
        this method. Terminations exclude timeouts. Physical quantities retain
        their original reward functions, weights and dt scaling.
        """
        if getattr(self.env.cfg, "tgi_aligned", False):
            from .native_rewards import tgi_core_reward

            return tgi_core_reward(self, state, action, previous_action, velocity, context)
        bodies = len(self.physics.contact_body_names)
        if contacts.shape[-1] != 2 * bodies + 2:
            raise ValueError("Native imagined contacts need total, normal and instantaneous channels.")
        total, normal = contacts[..., :bodies], contacts[..., bodies : 2 * bodies]
        zeros = torch.zeros_like(total)
        forces = torch.stack((zeros, zeros, 2 * total), -1).unsqueeze(1)
        normal_forces = torch.stack((zeros, zeros, 2 * normal), -1).unsqueeze(1)
        ids = torch.as_tensor(self.foot_contact_ids, device=contacts.device)
        air = torch.zeros_like(total).index_copy(-1, ids, timers[..., :2])
        contact = torch.zeros_like(total).index_copy(-1, ids, timers[..., 2:])
        view = self._reward_view(
            state,
            action,
            previous_action,
            velocity,
            context,
            auxiliary,
            forces,
            normal_forces,
            air,
            contact,
            termination_probability > 0.5,
            sensor_context,
            previous_position,
        )
        terms = self._weighted_rewards(view)
        data = view.scene["robot"].data
        raw = {"is_alive": 1 - termination_probability}
        threshold = self.reward_cfg["feet_air_time"].params["vel_threshold"]
        moving = (velocity[..., :2].norm(dim=-1) > threshold) | (velocity[..., 2].abs() > threshold)
        raw["feet_air_time"] = single_stance_time(timers) * moving
        for name in ("feet_slide", "feet_flat_ori", "feet_at_plane"):
            cfg = self.reward_cfg[name]
            sensor_cfg = cfg.params.get("sensor_cfg", cfg.params.get("contact_sensor_cfg"))
            body_ids = cfg.params["asset_cfg"].body_ids
            contact_weights = total[..., sensor_cfg.body_ids]
            if name == "feet_slide":
                if cfg.params.get("threshold", 0.1) != 1.0:
                    raise ValueError("Native F total-history contact labels use a 1 N threshold.")
                value = data.body_lin_vel_w.torch[..., body_ids, :2].norm(dim=-1)
                if cfg.params.get("ang_vel_penalty", False):
                    value = value + data.body_ang_vel_w.torch[..., body_ids, :2].norm(dim=-1)
                raw[name] = (value * contact_weights).sum(-1)
            elif name == "feet_flat_ori":
                rotations = quat_to_mat(data.body_quat_w.torch[..., body_ids, :][..., [3, 0, 1, 2]])
                gravity = mv(
                    rotations.transpose(-1, -2),
                    torch.nn.functional.normalize(context["GRAVITY_VEC_W"], dim=-1)[..., None, :],
                )
                raw[name] = (gravity[..., :2].norm(dim=-1) * contact_weights).sum(-1)
            else:
                hits = [
                    view.scene[n].data.ray_hits_w.torch[..., 2] for n in ("left_height_scanner", "right_height_scanner")
                ]
                ground = torch.stack(hits, -2)
                heights = data.body_pos_w.torch[..., body_ids, 2:3]
                offset = cfg.params.get("height_offset", 0.035)
                raw[name] = ((heights - ground - offset).clamp(0, 0.3) * contact_weights[..., None]).sum((-2, -1))
        cfg = self.reward_cfg["undesired_contacts"]
        if cfg.params["threshold"] != 1.0:
            raise ValueError("Native F normal-history contact labels use a 1 N threshold.")
        raw["undesired_contacts"] = normal[..., cfg.params["sensor_cfg"].body_ids].sum(-1)
        for name, value in raw.items():
            terms[name] = value * self.reward_cfg[name].weight * self.dt
        return sum(terms.values()), terms

    def locomotion_readout(self, state, context, contact, velocity):
        """Privileged training geometry; never actor or visual F input."""
        from instinctlab_supermimic.locomotion import locomotion_terms

        points = self.physics.foot_points(state)
        ground, valid = self.physics.terrain.query(points[..., :2], exact=True)
        clearance_center = points.mean(-2)
        clearance_ground, clearance_valid = self.physics.terrain.query(clearance_center[..., :2], exact=True)
        data = self.data(state, context)
        foot_velocity = data.body_link_lin_vel_w.torch[..., self.foot_ids, :]
        return locomotion_terms(
            contact,
            foot_velocity,
            points,
            ground,
            valid,
            velocity,
            self.dt,
            support_field=self.physics.terrain.query_support(points[..., :2]),
            weights=getattr(self.env.cfg, "locomotion_reward_weights", None),
            tgi_recipe=getattr(self.env.cfg, "tgi_aligned", False),
            clearance_ground=clearance_ground,
            clearance_valid=clearance_valid,
        )

    def critic_features(self, state, command, timers, physical_context):
        target = mv(yaw_mat(state[1]).transpose(-1, -2), command["target"] - state[0])
        target = torch.cat((target[..., :2], torch.linalg.vector_norm(target[..., :2], dim=-1, keepdim=True)), -1)
        return torch.cat(
            (
                invariant_features(state),
                self.velocity_command(state, command),
                target,
                command["maximum"],
                command["standing"][..., None].float(),
                command["random"][..., None].float(),
                timers,
                physical_context,
                self.physics.scan(state),
            ),
            -1,
        )

    def soft_termination(self, state, command, contacts, height_width=0.02, tilt_width=0.04):
        """Smooth physical termination probability; native timeouts are excluded."""
        bodies = len(self.physics.contact_body_names)
        normal = contacts[..., bodies : 2 * bodies]
        ids = self.termination_cfg["base_contact"].params["sensor_cfg"].body_ids
        base_alive = (1 - normal[..., ids]).prod(-1)
        rotation = state[1]
        tilt = torch.atan2(rotation[..., 2, :2].norm(dim=-1), rotation[..., 2, 2])
        angle = self.termination_cfg["bad_orientation"].params["limit_angle"]
        upright = 1 - torch.sigmoid((tilt - angle) / tilt_width)
        height = state[0][..., 2] - command["origin"][..., 2].clamp(max=0)
        minimum = self.termination_cfg["root_height"].params["minimum_height"]
        high = 1 - torch.sigmoid((minimum - height) / height_width)
        return 1 - base_alive * upright * high


class NativeTaskScene(dict):
    """Native scene exposes the same sensor objects by [] and .sensors[]."""

    @property
    def sensors(self):
        return self


class NativeTaskActuators(dict):
    """Native MDP reads collection effort and per-actuator stiffness."""

    def __init__(self, stiffness, applied_effort):
        super().__init__(implicit_pd=SimpleNamespace(joint_indices=slice(None), stiffness=stiffness))
        self.applied_effort = applied_effort
