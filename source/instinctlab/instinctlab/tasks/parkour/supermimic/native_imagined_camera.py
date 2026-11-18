"""Training-only depth queries at imagined G1 poses, without advancing Newton.

Mesh handles are shared with the native camera. Transforms, ray buffers and
postprocessing belong to this query object; no live sensor buffer is written.
The camera lookup deliberately has no state/action derivative.
"""

import copy
import torch
from pathlib import Path
from types import SimpleNamespace

import warp as wp
from instinctlab_supermimic.quaternion_xyzw import multiply as _mul
from instinctlab_supermimic.quaternion_xyzw import rotate as _rotate

from isaaclab.sensors.ray_caster import kernels
from isaaclab.utils.warp import ProxyArray

from instinctlab.sensors.grouped_ray_caster.grouped_kernels import raycast_mesh_table_min_distance_kernel
from instinctlab.sensors.noisy_camera.noisy_camera import NoisyCameraMixin

from .stair_kinematics import NativeLinkKinematics


class ImaginedDepthSequence:
    """Own one replay-seeded 36-frame history per imagined trajectory.

    Geometry is detached by render(); encoding stays differentiable during
    model fitting. Only active episode prefixes are queried or encoded.
    """

    def __init__(self, camera, perception, images, env_ids):
        if images.shape != (len(env_ids), 36, 18, 32):
            raise ValueError("Imagined depth requires the complete native 36-frame history.")
        self.camera, self.perception = camera, perception
        self.images, self.env_ids = images.float().clone(), env_ids

    def __call__(self, state, active, step):
        from instinctlab_supermimic.physical import flatten_state

        images = self.images[active]
        if step:
            image = self.camera.render(flatten_state(state), self.env_ids[active])["image"]
            images = torch.cat((images[:, 1:], image[:, None]), 1)
            self.images[active] = images
        return self.perception.encode(images[:, ::5])


def _tensor(value):
    return value.torch if hasattr(value, "torch") else wp.to_torch(value)


class NativeImaginedDepthCamera:
    """Static scene plus the imagined robot's own meshes, indexed by source env.

    Repeated env indices are allowed: imagination samples several trajectories
    from the same physical environment. The current Hiking postprocessing is
    stateless crop/blur/normalization; stateful image noise needs replay context
    before it can be used by this interface.
    """

    def __init__(self, env):
        sensor, robot = env.scene["camera"], env.scene["robot"]
        # Force lazy initialization before taking an immutable geometry snapshot.
        _ = sensor.data
        if sensor.cfg.data_types != ["distance_to_image_plane"]:
            raise ValueError("Imagined camera requires the native image-plane depth output.")
        if any(hasattr(noise.func, "reset") for noise in sensor.noise_pipeline):
            raise ValueError("Stateful image noise needs explicit replay state for imagined queries.")
        self.device = env.device
        self.cfg = copy.deepcopy(sensor.cfg)
        self.image_shape, self.num_rays = sensor.image_shape, sensor.num_rays
        self.kin = NativeLinkKinematics(Path(robot.cfg.spawn.asset_path), self.device, robot.body_names)
        self.meshes = tuple(sensor.meshes.values())  # Keep shared Warp meshes alive.
        # uint64 CUDA indexing is not supported by all PyTorch builds. Mesh
        # handles keep their bits in int64 storage until wrapped for Warp.
        self.mesh_ids = _tensor(sensor._mesh_ids_wp).view(torch.int64).clone()
        self.mesh_positions = _tensor(sensor._mesh_positions_w).clone()
        self.mesh_quaternions = _tensor(sensor._mesh_orientations_w).clone()
        indices = _tensor(sensor._mesh_body_indices).cpu().tolist()
        paths = sensor._scene_data_provider.backend.transform_paths
        names = list(robot.body_names)
        self.mesh_links = torch.tensor(
            [[-1 if index < 0 else names.index(paths[index].rsplit("/", 1)[-1]) for index in row] for row in indices],
            device=self.device,
        )
        sites = _tensor(sensor._sensor_site_indices).long()
        owners = _tensor(sensor.backend.model.shape_body)[sites].cpu().tolist()
        if any(index < 0 for index in owners):
            raise ValueError("Imagined camera must mount on a robot body.")
        self.mount_links = torch.tensor(
            [names.index(sensor.backend.model.body_label[index].rsplit("/", 1)[-1]) for index in owners],
            device=self.device,
        )
        self.site_transforms = _tensor(sensor.backend.model.shape_transform)[sites].clone()
        self.offset_positions = _tensor(sensor._offset_pos_wp).clone()
        self.offset_quaternions = _tensor(sensor._offset_quat_wp).clone()
        self.drift = _tensor(sensor.drift).clone()
        self.ray_drift = _tensor(sensor.ray_cast_drift).clone()
        self.ray_starts = _tensor(sensor.ray_starts).clone()
        self.ray_directions = _tensor(sensor.ray_directions).clone()
        self.clip_enabled = sensor._depth_clip_enabled
        self.clip_fill = sensor._depth_clip_fill_value
        self.buffers = {}

    def _allocate(self, batch):
        def array(shape, dtype):
            return wp.zeros(shape, dtype=dtype, device=self.device)

        b = SimpleNamespace(
            mask=wp.ones(batch, dtype=wp.bool, device=self.device),
            position=array(batch, wp.vec3f),
            quaternion=array(batch, wp.quatf),
            starts=array((batch, self.num_rays), wp.vec3f),
            directions=array((batch, self.num_rays), wp.vec3f),
            hits=array((batch, self.num_rays), wp.vec3f),
            distance=array((batch, self.num_rays), wp.float32),
            normal=array((1, 1), wp.vec3f),
            face=array((1, 1), wp.int32),
            mesh=array((1, 1), wp.int16),
            depth=array((batch, *self.image_shape, 1), wp.float32),
        )
        b.processing = object.__new__(NoisyCameraMixin)
        b.processing.cfg = copy.deepcopy(self.cfg)
        b.processing.device, b.processing.num_instances = self.device, batch
        b.processing._data = SimpleNamespace(output={"distance_to_image_plane": ProxyArray(b.depth)})
        b.processing.build_noise_pipeline()
        # Terminal masking changes batch size during model fitting. Retaining
        # every size would eventually keep hundreds of full ray buffers alive.
        self.buffers = {batch: b}
        return b

    @torch.no_grad()
    def render(self, state, env_ids):
        """Return raw NHWC depth and processed NHW images from detached states."""
        if state.ndim != 2 or state.shape[-1] != 71 or env_ids.shape != state.shape[:1]:
            raise ValueError("Expected B x 71 native states and B source environment indices.")
        batch = len(state)
        b = self.buffers.get(batch) or self._allocate(batch)
        parts = dict(
            root_pos_w=state[:, :3],
            root_quat_w=state[:, [4, 5, 6, 3]],
            root_lin_vel_w=state[:, 7:10],
            root_ang_vel_w=state[:, 10:13],
            joint_pos=state[:, 13:42],
            joint_vel=state[:, 42:71],
        )
        links = self.kin.links(parts)
        positions, quaternions = links["body_link_pos_w"], links["body_link_quat_w"]
        rows = torch.arange(batch, device=state.device)
        mount = self.mount_links[env_ids]
        site = self.site_transforms[env_ids]
        site_position = positions[rows, mount] + _rotate(quaternions[rows, mount], site[:, :3])
        site_quaternion = _mul(quaternions[rows, mount], site[:, 3:])
        transforms = torch.cat((site_position, site_quaternion), -1).contiguous()
        slot = self.mesh_links[env_ids]
        mesh_position = torch.where(
            (slot >= 0)[..., None], positions[rows[:, None], slot.clamp_min(0)], self.mesh_positions[env_ids]
        ).contiguous()
        mesh_quaternion = torch.where(
            (slot >= 0)[..., None], quaternions[rows[:, None], slot.clamp_min(0)], self.mesh_quaternions[env_ids]
        ).contiguous()

        def selected(value, dtype):
            value = value[env_ids].contiguous()
            return wp.from_torch(value, dtype=dtype)

        wp.launch(
            kernels.update_ray_caster_kernel,
            dim=(batch, self.num_rays),
            inputs=[
                wp.from_torch(transforms, dtype=wp.transformf),
                b.mask,
                selected(self.offset_positions, wp.vec3f),
                selected(self.offset_quaternions, wp.quatf),
                selected(self.drift, wp.vec3f),
                selected(self.ray_drift, wp.vec3f),
                selected(self.ray_starts, wp.vec3f),
                selected(self.ray_directions, wp.vec3f),
                int(kernels.ALIGNMENT_BASE),
            ],
            outputs=[b.position, b.quaternion, b.starts, b.directions],
            device=self.device,
        )
        wp.launch(
            kernels.fill_ray_hits_distance_inf_kernel,
            dim=(batch, self.num_rays),
            inputs=[b.mask, False],
            outputs=[b.hits, b.distance, b.normal],
            device=self.device,
        )
        wp.launch(
            raycast_mesh_table_min_distance_kernel,
            dim=(batch, self.num_rays),
            inputs=[
                b.mask,
                selected(self.mesh_ids, wp.uint64),
                b.starts,
                b.directions,
                wp.from_torch(mesh_position, dtype=wp.vec3f),
                wp.from_torch(mesh_quaternion, dtype=wp.quatf),
                float(kernels.CAMERA_RAYCAST_MAX_DIST),
                float(self.cfg.min_distance),
                0,
                0,
                0,
            ],
            outputs=[b.hits, b.distance, b.normal, b.face, b.mesh],
            device=self.device,
        )
        wp.launch(
            kernels.compute_distance_to_image_plane_to_image_masked_kernel,
            dim=(batch, self.num_rays),
            inputs=[
                b.mask,
                b.quaternion,
                b.distance,
                b.directions,
                int(self.image_shape[1]),
                bool(self.clip_enabled),
                float(self.cfg.max_distance),
                self.clip_fill,
            ],
            outputs=[b.depth],
            device=self.device,
        )
        raw = wp.to_torch(b.depth).clone()
        processed = b.processing.apply_noise_pipeline(raw, env_ids=rows)
        return dict(
            raw=raw,
            image=processed[..., 0],
            position=wp.to_torch(b.position).clone(),
            quaternion=wp.to_torch(b.quaternion).clone(),
        )
