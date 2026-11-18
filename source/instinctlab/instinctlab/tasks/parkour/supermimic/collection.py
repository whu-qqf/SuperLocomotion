"""Collect episode-safe real trajectories and their fixed terrain representation."""

from __future__ import annotations

import argparse
import copy
import json
import numpy as np
import torch
from pathlib import Path

from instinctlab.terrains.terrain_importer import TerrainImporter


class DynamicsTerrainImporter(TerrainImporter):
    """Retain the exact imported mesh only for the dynamics collection experiment."""

    def import_mesh(self, name, mesh):
        super().import_mesh(name, mesh)
        self.dynamics_mesh = mesh


def rasterize_terrain(mesh, device, resolution, local_xy):
    from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh

    from .terrain import TerrainHeightField

    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=device)
    origin = torch.tensor(mesh.bounds[0, :2], dtype=torch.float32, device=device)
    counts = np.ceil((mesh.bounds[1, :2] - mesh.bounds[0, :2]) / resolution).astype(int) + 1
    x = origin[0] + torch.arange(int(counts[0]), device=device) * resolution
    y = origin[1] + torch.arange(int(counts[1]), device=device) * resolution
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    starts = torch.stack((xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), 20.0)), -1)
    heights = []
    for chunk in starts.split(65536):
        directions = torch.zeros_like(chunk)
        directions[:, 2] = -1
        heights.append(raycast_mesh(chunk, directions, wp_mesh)[0][:, 2])
    heights = torch.cat(heights).reshape(len(y), len(x))
    valid = torch.isfinite(heights)
    terrain = TerrainHeightField(torch.where(valid, heights, 0), valid, origin, resolution, local_xy).to(device)
    return terrain, wp_mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--motion-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--terrain-seed", type=int, default=42)
    parser.add_argument("--raster-resolution", type=float, default=0.025)
    parser.add_argument("--gradient-probes", action="store_true")
    args = parser.parse_args()

    import gymnasium as gym

    from instinct_rl.runners import OnPolicyRunner

    from isaaclab.app import launch_simulation
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.utils.warp import raycast_mesh

    from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
    from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_plane_cfg import SuperMimicObsExtrasCfg
    from instinctlab.tasks.shadowing.supermimic.world_model_frames import (
        HeadingFrameParameterization,
        _heading,
        _rotate,
    )
    from instinctlab.utils.config import set_cfg_joint_order
    from instinctlab.utils.instinct_rl_cli import resolve_train_cfg

    from .actuator_context import HikingActuatorContext
    from .dynamics import HikingDynamicsContext

    torch.manual_seed(args.seed)
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    task = "Instinct-Parkour-Target-Amp-G1-v0"
    cfg, agent_cfg = resolve_train_cfg(task, [])
    cfg.scene.num_envs = args.num_envs
    cfg.scene.terrain.class_type = DynamicsTerrainImporter
    cfg.scene.terrain.terrain_generator.num_rows = 2
    cfg.scene.terrain.terrain_generator.num_cols = 10
    cfg.scene.terrain.terrain_generator.seed = args.terrain_seed
    cfg.scene.terrain.max_init_terrain_level = 1
    cfg.seed = args.seed
    cfg.compute_final_obs = True
    motion = Path(args.motion_directory).resolve()
    motion_cfg = cfg.scene.motion_reference.motion_buffers["run_walk"]
    motion_cfg.path = str(motion)
    motion_cfg.filtered_motion_selection_filepath = str(motion / "parkour_motion_without_run.yaml")
    extra = SuperMimicObsExtrasCfg()
    cfg.observations.wm_state = copy.deepcopy(extra.wm_state)
    cfg.observations.sim_context = copy.deepcopy(extra.sim_context)
    order = list(G1_29DOF_POLICY_JOINT_ORDER_V1)
    for name in ("joint_pos", "joint_vel"):
        getattr(cfg.observations.wm_state, name).params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), order
        )
    cfg.observations.sim_context.joint_offset.params["asset_cfg"] = set_cfg_joint_order(SceneEntityCfg("robot"), order)
    cfg.log_dir = str(output)

    with launch_simulation(cfg, {"visualizer": "none"}):
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        env = InstinctRlVecEnvWrapper(gym.make(task, cfg=cfg))
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device="cuda:0")
        runner.load(args.checkpoint)
        policy = runner.get_inference_policy(device="cuda:0")
        obs_format = env.get_obs_format()
        parameterization = HeadingFrameParameterization(obs_format["wm_state"])
        obs, info = env.get_observations()
        offset = 0
        for name, shape in obs_format["policy"].items():
            size = int(np.prod(shape))
            if name == "actions":
                action_slice = slice(offset, offset + size)
                break
            offset += size
        else:
            raise KeyError("Hiking policy action history missing")

        axis = torch.linspace(-1.2, 1.2, 17, device=env.device)
        xx, yy = torch.meshgrid(axis, axis, indexing="xy")
        local_xy = torch.stack((xx.flatten(), yy.flatten()), -1)
        mesh = env.unwrapped.scene.terrain.dynamics_mesh
        terrain, wp_mesh = rasterize_terrain(mesh, env.device, args.raster_resolution, local_xy)
        delay_context = HikingActuatorContext(env.unwrapped.scene["robot"], order)
        caps = delay_context.caps.tolist()
        names = order
        layout = {
            "sizes": {
                "terrain": len(local_xy),
                "physics": info["observations"]["sim_context"].shape[-1],
                "actuator_lag": len(names),
                "actuator_age": len(names),
                "action_history": action_slice.stop - action_slice.start,
            },
            "history_length": 8,
            "actuator_names": names,
            "age_caps": caps,
            "decimation": cfg.decimation,
            "actuator_runtime": delay_context.kind,
        }
        if max(caps) > cfg.decimation:
            raise ValueError("actuator delay exceeds one control interval; a substep target queue is required")
        context_builder = HikingDynamicsContext(terrain, parameterization, layout)
        records = {k: [] for k in ("state", "action", "context", "next_state", "done", "reward", "contact_force")}
        scan_errors = []
        supports = []
        with torch.no_grad():
            for step in range(args.steps):
                pack = info["observations"]
                state = pack["wm_state"].clone()
                parts = parameterization._unpack(state)
                scan, supported = terrain.scan(parts["root_pos_w"], parts["root_quat_w"])
                lag, age = delay_context.read()
                context = torch.cat((scan, pack["sim_context"], lag, age, obs[:, action_slice]), -1).clone()
                # Independent exact mesh rays measure raster approximation at real poses.
                local = torch.cat((local_xy, torch.zeros_like(local_xy[:, :1])), -1)
                starts = parts["root_pos_w"][:, None] + _rotate(_heading(parts["root_quat_w"])[:, None], local[None])
                starts[..., 2] = 20.0
                directions = torch.zeros_like(starts)
                directions[..., 2] = -1
                exact = parts["root_pos_w"][:, 2:3] - raycast_mesh(starts, directions, wp_mesh)[0][..., 2]
                supports.append(supported.cpu())
                scan_errors.append((scan - exact).abs().cpu())
                force = env.unwrapped.scene["contact_forces"].data.net_forces_w.torch.norm(dim=-1).clone()
                action = policy(obs) + 0.1 * torch.randn(args.num_envs, env.num_actions, device=env.device)
                obs, reward, done, info = env.step(action)
                reached = info["observations"]["wm_state"].clone()
                if done.any():
                    reached[done.bool()] = info["final_observations"]["wm_state"][done.bool()]
                values = (state, action, context, reached, done, reward, force)
                for key, value in zip(records, values):
                    if not torch.isfinite(value).all():
                        raise ValueError(f"non-finite {key} at step {step}")
                    records[key].append(value.cpu())
                if step % 64 == 0:
                    print(f"HIKING: collected {step + 1}/{args.steps}", flush=True)

        data = {key: torch.stack(value) for key, value in records.items()}
        data.update(
            state_format=dict(obs_format["wm_state"]),
            context_layout=layout,
            joint_names=order,
            body_names=list(cfg.observations.wm_state.body_link_pos_w.params["asset_cfg"].body_names),
            physics_format=dict(obs_format["sim_context"]),
            terrain={k: v.cpu() for k, v in terrain.state_dict().items()},
            terrain_resolution=args.raster_resolution,
            terrain_seed=args.terrain_seed,
            collection_seed=args.seed,
            terrain_supported=torch.stack(supports),
            policy_checkpoint=str(Path(args.checkpoint).resolve()),
            task=task,
            step_dt=env.unwrapped.step_dt,
        )
        errors = torch.stack(scan_errors)
        finite = torch.isfinite(errors)
        metrics = {
            "transitions": args.steps * args.num_envs,
            "terminal_transitions": int(data["done"].sum()),
            "raster_scan_supported_fraction": float(data["terrain_supported"].float().mean()),
            "raster_vs_exact_scan_mae_m": float(errors[finite].mean()),
            "raster_vs_exact_scan_p95_m": float(errors[finite].quantile(0.95)),
            "exact_scan_finite_fraction": float(finite.float().mean()),
            "actuator_names": names,
            "context_size": context_builder.size,
            "checkpoint": data["policy_checkpoint"],
            "actuator_runtime": delay_context.kind,
        }
        data["collection_metrics"] = metrics
        torch.save(data, output / "transitions.pt")
        (output / "collection_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
        print(json.dumps(metrics, indent=2), flush=True)
        if args.gradient_probes:
            from .probes import collect_physical_probes

            state = info["observations"]["wm_state"].clone()
            parts = parameterization._unpack(state)
            scan, _ = terrain.scan(parts["root_pos_w"], parts["root_quat_w"])
            lag, age = delay_context.read()
            context = torch.cat((scan, info["observations"]["sim_context"], lag, age, obs[:, action_slice]), -1)
            probes = collect_physical_probes(env, state, context, policy(obs).detach())
            torch.save(probes, output / "physical_probes.pt")
            print("HIKING: physical perturbation probes saved", flush=True)
        env.close()
