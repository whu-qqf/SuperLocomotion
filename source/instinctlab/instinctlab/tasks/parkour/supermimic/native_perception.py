"""Native Hiking camera interface for the shared visual RSSM.

This does not register a training task: the native physical learner still needs
migration. It retains the real Hiking camera pose, rays and noise processing.
"""

import torch

import warp as wp

from isaaclab.managers import SceneEntityCfg

from instinctlab.envs.mdp.observations.exteroception import visualizable_image
from instinctlab.sensors.noisy_camera.newton_ray_caster_cameras import NewtonNoisyGroupedRayCasterCamera


class ClockedNewtonDepthCamera(NewtonNoisyGroupedRayCasterCamera):
    def _initialize_impl(self):
        super()._initialize_impl()
        self.frame_generation = torch.zeros(self.num_instances, dtype=torch.long, device=self.device)

    def _update_buffers_impl(self, env_mask):
        super()._update_buffers_impl(env_mask)
        self.frame_generation[wp.to_torch(env_mask).bool()] += 1


def configure_native_perception(cfg):
    """Use with the resolved Newton Hiking cfg, before constructing the scene."""
    cfg.scene.camera.class_type = ClockedNewtonDepthCamera
    cfg.scene.camera.update_period = 0.1
    cfg.scene.camera.data_histories = {}
    for group in (cfg.observations.policy, cfg.observations.critic):
        group.depth_image.func = visualizable_image
        group.depth_image.params = dict(sensor_cfg=SceneEntityCfg("camera"), data_type="distance_to_image_plane_noised")
    # The same constraint capacity already required by existing native audits.
    cfg.sim.physics.solver_cfg.njmax = 2048
    return cfg


def observable_proprio(obs, history=8):
    """Native term-major policy layout; exclude commands and past actions."""
    widths = (3, 3, 3, 29, 29, 29)
    frames, offset = [], 0
    for width in widths:
        frames.append(obs[:, offset : offset + history * width].view(len(obs), history, width)[:, -1])
        offset += history * width
    if offset != 768:
        raise ValueError("Unexpected native proprio history shape.")
    return torch.cat([frames[i] for i in (0, 1, 3, 4)], -1)
