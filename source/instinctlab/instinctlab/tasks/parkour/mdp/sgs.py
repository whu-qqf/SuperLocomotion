"""Success-guided terrain sampling for native Hiking.

The sampler keeps an outcome window for each terrain row, terrain column,
spawn patch and travel direction. Sampling remains proportional to the
configured terrain proportions while concentrating probability near a target
success rate, as in TGI.
"""

from __future__ import annotations

import math
import torch


class SuccessGuidedSampler:
    def __init__(
        self,
        num_configs: int,
        device,
        window: int = 16,
        target: float = 0.8,
        kappa: float = 5.0,
        temperature: float = 2.0,
        group: torch.Tensor | None = None,
        group_prior: torch.Tensor | None = None,
        prev: torch.Tensor | None = None,
        neighbor: float = 0.25,
    ):
        if num_configs < 1 or window < 1 or not 0 < target < 1:
            raise ValueError("SGS needs positive sizes and a target strictly between zero and one.")
        self.n, self.window = num_configs, window
        self.group, self.group_prior = group, group_prior
        self.target, self.kappa, self.temperature = target, kappa, max(temperature, 1.0)
        self.outcomes = torch.zeros(num_configs, window, device=device)
        self.count = torch.zeros(num_configs, dtype=torch.long, device=device)
        self.prev, self.neighbor = prev, neighbor
        self.levels = 0
        if prev is not None:
            chain = torch.arange(num_configs, device=device)
            while bool((chain >= 0).any()):
                chain = torch.where(chain >= 0, prev[chain.clamp_min(0)], chain)
                self.levels += 1
                if self.levels > num_configs:
                    raise ValueError("SGS predecessor graph contains a cycle.")

    def success_rate(self):
        filled = self.count.clamp(max=self.window)
        return self.outcomes.sum(-1) / filled.clamp(min=1)

    def effective_success(self):
        p = self.success_rate()
        if self.prev is None or self.neighbor <= 0:
            return p
        for _ in range(self.levels):
            previous = torch.where(self.prev >= 0, p[self.prev.clamp(min=0)], torch.zeros_like(p))
            updated = torch.maximum(p, self.neighbor * previous)
            p = updated
        return p

    def probs(self):
        p = self.effective_success()
        weights = (p + 1e-8) ** (self.kappa * self.target) * (1 - p + 1e-8) ** (self.kappa * (1 - self.target))
        logits = torch.log(weights.clamp_min(1e-8) + 1e-8) / self.temperature
        if self.group is None:
            return torch.softmax(logits, 0)
        e = torch.exp(logits - logits.max())
        z = torch.zeros_like(self.group_prior).index_add_(0, self.group, e)
        return self.group_prior[self.group] * e / z[self.group].clamp_min(1e-12)

    def sample(self, count: int):
        return torch.multinomial(self.probs(), count, replacement=True)

    def record(self, config_ids, success):
        ids, order = torch.sort(config_ids, stable=True)
        positions = torch.arange(len(ids), device=ids.device)
        first = torch.full((self.n,), len(ids), dtype=torch.long, device=ids.device)
        first.scatter_reduce_(0, ids, positions, reduce="amin", include_self=True)
        slots = (self.count[ids] + positions - first[ids]) % self.window
        totals = torch.bincount(ids, minlength=self.n)
        keep = positions - first[ids] >= totals[ids] - self.window
        self.outcomes[ids[keep], slots[keep]] = success[order][keep].float()
        self.count.index_add_(0, ids, torch.ones_like(ids))

    def state_dict(self):
        return {"outcomes": self.outcomes.clone(), "count": self.count.clone()}

    def load_state_dict(self, state):
        if state["outcomes"].shape != self.outcomes.shape or state["count"].shape != self.count.shape:
            raise ValueError("SGS checkpoint configuration dimensions differ.")
        self.outcomes.copy_(state["outcomes"].to(self.outcomes))
        self.count.copy_(state["count"].to(self.count))

    def stats(self):
        probability = self.probs()
        visited = self.count > 0
        return {
            "effective_configs": float(torch.exp(-(probability * probability.clamp_min(1e-30).log()).sum())),
            "mean_success": float(self.success_rate()[visited].mean()) if visited.any() else 0.0,
            "visited": float(visited.float().mean()),
        }


def _ids(env_ids, device, num_envs):
    if env_ids is None:
        env_ids = slice(None)
    if isinstance(env_ids, slice):
        return torch.arange(num_envs, device=device)[env_ids]
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def _state(env, spawns=4, directions=4, window=16, target=0.8, neighbor=0.25):
    state = getattr(env, "_hiking_sgs", None)
    terrain = env.scene.terrain
    if state is not None:
        return state
    if int(math.isqrt(spawns)) ** 2 != spawns or min(spawns, directions, window) < 1:
        raise ValueError("SGS spawn count must be a positive square; directions/window must be positive.")
    origins = terrain.terrain_origins
    rows, cols = origins.shape[:2]
    generator = terrain.cfg.terrain_generator
    proportions = torch.tensor(
        [float(item.proportion) for item in generator.sub_terrains.values()], device=env.device, dtype=torch.float32
    )
    proportions = proportions / proportions.sum()
    cumulative = torch.cumsum(proportions, 0)
    columns = torch.arange(cols, device=env.device)
    type_ids = torch.stack([torch.where(c / cols + 0.001 < cumulative)[0][0] for c in columns])
    represented = torch.bincount(type_ids, minlength=len(proportions)) > 0
    proportions = proportions * represented
    proportions /= proportions.sum()
    k = spawns * directions
    cell = torch.arange(rows * cols * k, device=env.device) // k % cols
    config_ids = torch.arange(rows * cols * k, device=env.device)
    previous = torch.where(config_ids >= cols * k, config_ids - cols * k, torch.full_like(config_ids, -1))
    state = {
        "rows": rows,
        "cols": cols,
        "spawns": spawns,
        "directions": directions,
        "sampler": SuccessGuidedSampler(
            rows * cols * k,
            env.device,
            window=window,
            target=target,
            group=type_ids[cell],
            group_prior=proportions,
            prev=previous,
            neighbor=neighbor,
        ),
        "config_ids": torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device),
        "terrain_size": float(min(generator.size)),
        "start_xy": torch.zeros(env.num_envs, 2, device=env.device),
        "commanded_distance": torch.zeros(env.num_envs, device=env.device),
        "travel": torch.zeros(env.num_envs, device=env.device),
    }
    env._hiking_sgs = state
    return state


def _episode_success(env, env_ids, state):
    command = env.command_manager.get_term("base_velocity")
    robot = env.scene["robot"]
    current = robot.data.root_pos_w.torch[env_ids, :2]
    distance = (current - state["start_xy"][env_ids]).norm(dim=-1)
    time_left = (env.max_episode_length - env.episode_length_buf[env_ids]).clamp_min(0) * env.step_dt
    commanded = state["commanded_distance"][env_ids] + command.command[env_ids, :2].norm(dim=-1) * time_left
    walked = distance > (0.8 * commanded).clamp(max=state["terrain_size"] / 2)
    survived = env.episode_length_buf[env_ids] >= env.max_episode_length - 1
    return torch.where(commanded >= 1.0, walked, survived)


def sgs_terrain_levels(
    env,
    env_ids,
    spawns: int = 4,
    directions: int = 4,
    window: int = 16,
    target: float = 0.8,
    neighbor: float = 0.25,
):
    """Update terrain assignment from episode success and sample the next SGS configuration."""
    state = _state(env, spawns, directions, window, target, neighbor)
    ids = _ids(env_ids, env.device, env.num_envs)
    old = state["config_ids"][ids]
    if (old >= 0).any():
        valid = (old >= 0) & (env.episode_length_buf[ids] > 0)
        state["sampler"].record(old[valid], _episode_success(env, ids[valid], state))
    sampled = state["sampler"].sample(len(ids))
    state["config_ids"][ids] = sampled
    k = spawns * directions
    cells = sampled // k
    levels, columns = cells // state["cols"], cells % state["cols"]
    terrain = env.scene.terrain
    terrain.terrain_levels[ids] = levels
    terrain.terrain_types[ids] = columns
    terrain.env_origins[ids] = terrain.terrain_origins[levels, columns]
    return float(levels.float().mean())


def sgs_reset_spawn(env, env_ids, asset_cfg=None):
    """Apply sampled spawn patch and travel direction after the standard reset event."""
    state = getattr(env, "_hiking_sgs", None)
    if state is None:
        return
    ids = _ids(env_ids, env.device, env.num_envs)
    sub = state["config_ids"][ids] % (state["spawns"] * state["directions"])
    n = int(round(math.sqrt(state["spawns"])))
    patch = sub // state["directions"]
    off = torch.stack((patch // n, patch % n), -1).float()
    span = env.cfg.events.reset_base.params["pose_range"]
    span_xy = torch.tensor((span["x"][1] - span["x"][0], span["y"][1] - span["y"][0]), device=env.device) * n
    data = env.scene["robot"].data
    pose = torch.cat((data.root_pos_w.torch[ids], data.root_quat_w.torch[ids]), -1).clone()
    pose[:, :2] += ((off + 0.5) / n - 0.5) * span_xy
    direction = sub % state["directions"]
    yaw = (direction.float() + torch.rand(len(ids), device=env.device)) * (2 * math.pi / state["directions"]) - math.pi
    from isaaclab.utils.math import quat_from_euler_xyz

    pose[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
    state["travel"][ids] = yaw
    state["start_xy"][ids] = pose[:, :2]
    state["commanded_distance"][ids] = 0
    env.scene["robot"].write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)


def sgs_pre_step(env):
    """Integrate the actual command before physics and before same-step autoreset."""
    state = getattr(env, "_hiking_sgs", None)
    if state is not None:
        command = env.command_manager.get_command("base_velocity")
        state["commanded_distance"] += command[:, :2].norm(dim=-1) * env.step_dt


def configure_sgs(cfg):
    """Replace terrain promotion and partition the existing spawn area without widening it."""
    from isaaclab.managers import CurriculumTermCfg, EventTermCfg

    cfg.curriculum.terrain_levels = CurriculumTermCfg(
        func=sgs_terrain_levels,
        params=dict(spawns=4, directions=4, window=16, target=0.8, neighbor=0.25),
    )
    for axis in ("x", "y"):
        lo, hi = cfg.events.reset_base.params["pose_range"][axis]
        cfg.events.reset_base.params["pose_range"][axis] = (lo / 2, hi / 2)
    cfg.events.sgs_reset_spawn = EventTermCfg(func=sgs_reset_spawn, mode="reset")
    cfg.sgs_curriculum = "tgi_success_distance_spawn4_direction4_window16_target08_v1"
    return cfg
