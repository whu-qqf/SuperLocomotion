"""Recorded-terrain reset, timeline and reference commands for the native route."""

import torch
from pathlib import Path

from instinctlab_supermimic.rotations import mv, quat_to_mat

from instinctlab.tasks.parkour.mdp.commands.pose_velocity_command import PoseVelocityCommand
from instinctlab.tasks.shadowing.supermimic.terrain_reference import ReferenceTerrainImporter, configure_paired_terrain

from .paired_motion import PairedMotionLibrary, PairedMotionPlayer


class PairedTerrainImporter(ReferenceTerrainImporter):
    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        self.dynamics_mesh = self.reference_mesh


def paired_player_for_env(env):
    if not hasattr(env, "_paired_player"):
        library = PairedMotionLibrary(
            env.cfg.paired_reference_directory, Path(env.cfg.scene.robot.spawn.asset_path), env.device
        )
        # configure_paired_terrain uses one column, and terrain IDs are row IDs.
        terrain_ids = torch.arange(env.num_envs, device=env.device) % len(torch.unique(library.terrain_ids))
        origins = env.scene.terrain.terrain_origins[terrain_ids, 0]
        env.scene.terrain.terrain_levels[:] = terrain_ids
        env.scene.terrain.terrain_types[:] = 0
        env.scene.terrain.env_origins[:] = origins
        env.scene.env_origins[:] = origins
        env._paired_player = PairedMotionPlayer(library, terrain_ids, origins)
        env._paired_start_frame = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    return env._paired_player


def paired_environment_targets(env, offset=0, reset=False):
    """Read physical-time labels; pre-reset post-step queries include elapsed controls."""
    player = paired_player_for_env(env)
    elapsed = torch.zeros_like(env.episode_length_buf) if reset else env.episode_length_buf
    frames = env._paired_start_frame + elapsed + offset
    return player.library.query(player.state["clip_id"], frames[:, None], player.origins, player.terrain_ids)


def reset_paired_reference(env, env_ids, random_phase=True, minimum_remaining=32):
    player = paired_player_for_env(env)
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    if isinstance(ids, slice):
        ids = torch.arange(env.num_envs, device=env.device)[ids]
    player.reset(ids, random_phase=random_phase, minimum_remaining=minimum_remaining)
    env._paired_start_frame[ids] = player.state["frame"][ids]
    target = player.library.query(
        player.state["clip_id"][ids], player.state["frame"][ids, None], player.origins[ids], player.terrain_ids[ids]
    )
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints(player.library.joint_names, preserve_order=True)
    robot.write_joint_state_to_sim_index(
        position=target["joint_pos"][:, 0], velocity=target["joint_vel"][:, 0], joint_ids=joint_ids, env_ids=ids
    )
    pose = torch.cat((target["root_pos"][:, 0], target["root_quat"][:, 0, [1, 2, 3, 0]]), -1)
    robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
    velocity = torch.cat((target["root_velocity"][:, 0], target["root_angular_velocity"][:, 0]), -1)
    robot.write_root_link_velocity_to_sim_index(root_velocity=velocity, env_ids=ids)


def paired_reference_ended(env):
    player = paired_player_for_env(env)
    return env._paired_start_frame + env.episode_length_buf >= player.library.lengths[player.state["clip_id"]] - 1


def paired_scan_out_of_bounds(env):
    """Reset as soon as the model's entire terrain footprint becomes unsupported."""
    from .native_physics_audit import native_height_field

    robot = env.scene["robot"]
    _, valid = native_height_field(env).scan(robot.data.root_link_pos_w.torch, robot.data.root_link_quat_w.torch)
    return ~valid


class PairedReferenceCommand(PoseVelocityCommand):
    """Keep native observation interfaces, sourcing stage-one speed from the clip."""

    def _init_valid_targets(self):
        # This stage follows measured trajectories, not sampled flat-patch goals.
        self.valid_targets = None

    def _set_reference(self, env_ids, reset=False):
        target = paired_environment_targets(self._env, reset=reset)
        rotation = quat_to_mat(target["root_quat"][:, 0])
        velocity = mv(rotation.transpose(-1, -2), target["root_velocity"][:, 0])
        angular = mv(rotation.transpose(-1, -2), target["root_angular_velocity"][:, 0])
        command = torch.cat((velocity[:, :2], angular[:, 2:3]), -1)
        self.vel_command_b[env_ids] = command[env_ids]
        self.random_velocity_indices[env_ids] = True
        self.is_standing_env[env_ids] = False
        self.random_lin_vel_x[env_ids] = command[env_ids, 0]
        self.random_lin_vel_y[env_ids] = command[env_ids, 1]
        self.random_ang_vel_z[env_ids] = command[env_ids, 2]
        self.pos_command_w[env_ids] = target["root_pos"][env_ids, 0]
        self.max_command_b[env_ids] = command[env_ids].abs()
        self.pos_command_b[env_ids] = 0

    def _resample_command(self, env_ids):
        self._set_reference(env_ids, reset=True)
        self.progress_start_pos_w[env_ids] = self.robot.data.root_link_pos_w.torch[env_ids]

    def _update_command(self):
        self._set_reference(slice(None))


def configure_paired_environment(cfg, directory, horizon=32, random_phase=True):
    """Stage-one reference tracking; keep Hiking's original command class elsewhere."""
    from isaaclab.managers import EventTermCfg, TerminationTermCfg

    configure_paired_terrain(cfg, directory)
    cfg.scene.terrain.class_type = PairedTerrainImporter
    cfg.paired_reference_directory = str(Path(directory).resolve())
    cfg.scene.terrain.max_init_terrain_level = None
    cfg.curriculum = None  # Fixed terrain identity for every reset and replay row.
    cfg.events.reset_base = None
    cfg.events.reset_robot_joints = None
    cfg.events.reset_paired_reference = EventTermCfg(
        func=reset_paired_reference, mode="reset", params=dict(random_phase=random_phase, minimum_remaining=horizon)
    )
    cfg.terminations.paired_reference_end = TerminationTermCfg(func=paired_reference_ended, time_out=True)
    cfg.terminations.paired_scan_out_of_bounds = TerminationTermCfg(func=paired_scan_out_of_bounds, time_out=True)
    cfg.commands.base_velocity.class_type = PairedReferenceCommand
    cfg.commands.base_velocity.velocity_ranges = None
    cfg.commands.base_velocity.random_velocity_terrain = None
    cfg.commands.base_velocity.resampling_time_range = (1e6, 1e6)
    return cfg
