"""Native camera inference with the exported P/actor, without constructing F."""

import gymnasium as gym
import json
import torch
from pathlib import Path

from instinctlab_perception.checkpoint import load_hiking_controller

from isaaclab.app import launch_simulation
from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

from .native_perception import configure_native_perception, observable_proprio


@torch.no_grad()
def rollout_controller(vec, actor, runtime, obs, steps, transition_recorder=None, reference_actor=None):
    """Run exported inference; optional recorder consumption supports roundtrip audits.

    The standalone play process passes no recorder or reference actor. They are
    used only by the same-scene checkpoint audit, with no extra observations.
    """
    if steps < 1:
        raise ValueError("Native controller rollout needs a positive step budget.")
    env = vec.unwrapped
    camera = env.scene["camera"]
    seen = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    captures, ticks, resets, maximum = 0, 0, 0, 0.0
    for _ in range(steps):
        torch.testing.assert_close(runtime.control_step, env.episode_length_buf.long(), rtol=0, atol=0)
        generation = camera.frame_generation
        captured = generation != seen
        feature, tick = runtime.before_action(obs[:, 768:].view(-1, 1, 18, 32), observable_proprio(obs), captured)
        captures += int(captured.sum())
        ticks += 0 if tick is None else len(tick.env)
        seen.copy_(generation)
        action = actor(obs[:, :768], feature).clamp(-20, 20)
        if not torch.isfinite(action).all():
            raise FloatingPointError("Nonfinite exported native controller action.")
        if reference_actor is not None:
            torch.testing.assert_close(action, reference_actor(obs[:, :768], feature).clamp(-20, 20), rtol=0, atol=0)
        maximum = max(maximum, float(action.abs().max()))
        obs, _, done, _ = vec.step(action)
        if transition_recorder is not None:
            transition = transition_recorder.take_transition()
            torch.testing.assert_close(transition.action, action, rtol=0, atol=0)
            torch.testing.assert_close(done.bool(), transition.terminated | transition.truncated, rtol=0, atol=0)
        runtime.after_action(action)
        ids = done.nonzero(as_tuple=True)[0]
        if len(ids):
            runtime.reset(ids)
            seen[ids] = -1
            resets += len(ids)
    return dict(
        envs=env.num_envs,
        controls=steps,
        resets=resets,
        camera_captures=captures,
        perception_ticks=ticks,
        control_dt=env.step_dt,
        maximum_absolute_action=maximum,
        trained_modules_loaded=["perception", "policy"],
        optimizer_updates=0,
        reference_actor_equal=reference_actor is not None,
        scope="Inference interface check; not evidence of trained gait quality.",
    )


def main():
    parser = cli.build_play_parser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--seed", type=int, default=157)
    parser.add_argument("--flat_walk", action="store_true", help="Use the same flat ground and commands as training.")
    args, overrides = cli.parse_cli(parser, require_run_name=False)
    if not args.checkpoint or args.steps < 1:
        raise ValueError("Native controller inference requires --checkpoint and a positive --steps budget.")
    cfg, _ = cli.resolve_train_cfg(args.task, overrides)
    cfg = configure_native_perception(cfg)
    if args.flat_walk:
        from .native_rewards import configure_native_tracking_rewards
        from .native_training import configure_flat_walk

        configure_flat_walk(cfg)
        configure_native_tracking_rewards(cfg)
    cfg.recorders = RecorderManagerBaseCfg()
    cfg.scene.num_envs = args.num_envs or 8
    cfg.seed = args.seed
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    cli.pre_launch_video_config(cfg, args, step_offset=args.video_start_step, prefix="native_dual")
    with launch_simulation(cfg, args):
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        cli.apply_video_recording(cfg, args, str(output))
        vec = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        env = vec.unwrapped
        try:
            obs, _ = vec.reset()
            joint_names = env.action_manager.get_term("joint_pos")._joint_names
            actor, runtime = load_hiking_controller(args.checkpoint, env.num_envs, env.device, env.step_dt, joint_names)
            report = rollout_controller(vec, actor, runtime, obs, args.steps)
            report.update(task=args.task, physical_model_constructed=False)
            (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
            print("NATIVE_CONTROLLER_PASS", json.dumps(report), flush=True)
        finally:
            vec.close()
