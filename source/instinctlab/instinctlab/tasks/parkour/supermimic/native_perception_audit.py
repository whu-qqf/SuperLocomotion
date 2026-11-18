"""Native camera/RSSM check; optionally one F update and physical action probes."""

import gymnasium as gym
import hashlib
import json
import torch
from pathlib import Path

from instinct_rl.modules.moe import MoeLayer
from instinctlab_perception.events import PerceptionRuntime
from instinctlab_perception.policy import PerceptivePolicy
from instinctlab_perception.replay import PerceptionReplay, collate_ticks, reconstruct_prefixes
from instinctlab_perception.rssm import PerceptionModel

from isaaclab.app import launch_simulation
from isaaclab.managers import TerminationTermCfg

import instinctlab.tasks  # noqa: F401
from instinctlab.utils import instinct_rl_cli as cli

from .native_perception import configure_native_perception, observable_proprio
from .native_transitions import NativeHikingRecorderCfg


def partial_timeout(env):
    """One audit-only timeout to exercise Same-Step reset at a known control."""
    return (torch.arange(env.num_envs, device=env.device) == 0) & (env.common_step_counter == 8)


def digest(module):
    h = hashlib.sha256()
    for p in module.parameters():
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    parser = cli.build_train_parser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--validation_steps", type=int, default=32)
    parser.add_argument("--check_physics", action="store_true")
    parser.add_argument("--check_task", action="store_true")
    args, overrides = cli.parse_cli(parser, require_run_name=False)
    if args.validation_steps < 8:
        raise ValueError("Native interface audit needs at least eight controls to exercise the partial timeout.")
    cfg, _ = cli.resolve_train_cfg(args.task, overrides)
    cfg = configure_native_perception(cfg)
    cfg.recorders = NativeHikingRecorderCfg()
    if args.check_task:
        from .native_task_audit import NativeTaskAuditRecorderCfg

        cfg.recorders = NativeTaskAuditRecorderCfg()
    cfg.terminations.audit_timeout = TerminationTermCfg(func=partial_timeout, time_out=True)
    if args.check_physics or args.check_task:
        from .native_physics_audit import configure_native_physics_audit

        configure_native_physics_audit(cfg)
    cfg.scene.num_envs = args.num_envs or 128
    cfg.seed = 157
    torch.manual_seed(157)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    with launch_simulation(cfg, args):
        from instinctlab.utils.wrappers.instinct_rl.vecenv_wrapper import InstinctRlVecEnvWrapper

        vec = InstinctRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
        env = vec.unwrapped
        try:
            obs, _ = vec.reset()
            p = PerceptionModel().to(env.device)
            actor = PerceptivePolicy(MoeLayer(896, 4, 29, "elu", [256, 128, 64])).to(env.device)
            runtime = PerceptionRuntime(p, env.num_envs)
            replay = PerceptionReplay(env.num_envs)
            camera = env.scene["camera"]
            recorder = env.recorder_manager._terms["transition"]
            physics, physical_records = None, []
            if args.check_physics or args.check_task:
                from .native_physics_audit import fit_and_probe_native_physics, make_native_physics

                physics = make_native_physics(env, recorder)
                physical_initial = digest(physics.model)
            task_audit = env.recorder_manager._terms["task_parity"] if args.check_task else None
            if task_audit is not None:
                task_audit.bind(physics)
            seen = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
            initial = dict(perception=digest(p), policy=digest(actor))
            resets, captures, ticks, captured_transitions = 0, 0, 0, 0
            reset_target_differences = []
            for step in range(args.validation_steps):
                torch.testing.assert_close(runtime.control_step, env.episode_length_buf.long(), rtol=0, atol=0)
                assert obs.shape == (env.num_envs, 768 + 18 * 32), obs.shape
                generation = camera.frame_generation
                captured = generation != seen
                with torch.no_grad():
                    belief, tick = runtime.before_action(
                        obs[:, 768:].view(-1, 1, 18, 32), observable_proprio(obs), captured
                    )
                    action = actor(obs[:, :768], belief).clamp(-20, 20)
                seen.copy_(generation)
                captures += int(captured.sum())
                if tick is not None:
                    replay.add(tick)
                    ticks += len(tick.env)
                    assert (tick.capture_step[tick.observed] <= tick.control_step[tick.observed]).all()
                current_episode = runtime.episode_id.clone()
                physical_context = physics.context(env.action_manager.action) if physics is not None else None
                obs, _, done, _ = vec.step(action)
                transition = recorder.take_transition()
                torch.testing.assert_close(transition.action, action, rtol=0, atol=0)
                torch.testing.assert_close(done.bool(), transition.terminated | transition.truncated, rtol=0, atol=0)
                assert torch.isfinite(transition.next_state).all()
                captured_transitions += len(transition.action)
                if physics is not None:
                    physics.observe(transition, physical_context)
                    physical_records.append((transition, physical_context))
                runtime.after_action(transition.action)
                ids = done.bool().nonzero(as_tuple=True)[0]
                if len(ids):
                    # These target tensors must survive the reset performed in step().
                    reset_target_differences.append(
                        float((transition.next_state[ids] - recorder.state()[ids]).abs().amax())
                    )
                    # Native environment auto-reset is owned by its step method.
                    replay.finish(ids.cpu(), current_episode[ids].cpu())
                    runtime.reset(ids)
                    seen[ids] = -1
                    resets += len(ids)
                if (step + 1) % 8 == 0:
                    print("NATIVE_PERCEPTION_CONTROLS", step + 1, flush=True)
                if step == 7:
                    assert done[0], "The audit-only timeout did not exercise native autoreset."
                    assert runtime.episode_id[0] > current_episode[0]
            assert reset_target_differences and max(reset_target_differences) > 1e-5
            if task_audit is not None:
                (out / "task_parity.json").write_text(json.dumps(task_audit.result(), indent=2) + "\n")
                if task_audit.failures:
                    torch.save(task_audit.failures, out / "task_parity_failures.pt")
                    raise AssertionError(f"Native task parity failed: {list(task_audit.failures)}")
            assert digest(actor) == initial["policy"] and digest(p) == initial["perception"]
            samples = replay.sample(4, 8)
            start = reconstruct_prefixes(p, [s.prefix for s in samples])
            batch = collate_ticks([s.target for s in samples], p.cfg, env.device)
            opt = torch.optim.Adam(p.parameters(), lr=1e-4)
            loss, metrics, _ = p.sequence_loss(batch, start, sample=False)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(p.parameters(), 1.0)
            assert torch.isfinite(grad)
            opt.step()
            assert digest(p) != initial["perception"] and digest(actor) == initial["policy"]
            physical_report = None
            if args.check_physics:
                assert digest(physics.model) == physical_initial
                physical_report = fit_and_probe_native_physics(physics, physical_records, out)
                assert digest(physics.model) != physical_initial
                assert digest(actor) == initial["policy"]
            result = dict(
                task=args.task,
                envs=env.num_envs,
                controls=args.validation_steps,
                resets=resets,
                actual_camera_captures=captures,
                perception_ticks=ticks,
                pre_reset_transitions=captured_transitions,
                reset_target_max_difference=max(reset_target_differences),
                audit_partial_timeout_control=8,
                camera_period=0.1,
                control_dt=env.step_dt,
                perception_updates=1,
                policy_updates=0,
                physical_model_updates=physical_report["actual_updates"]["physical"] if physical_report else 0,
                physical=physical_report,
                task_readout_parity=task_audit.result() if task_audit is not None else None,
                grad_norm=float(grad),
                perception_metrics={k: float(v) for k, v in metrics.items()},
                native_terrain_preserved=True,
                native_camera_used=True,
                scope="Native perception/optional F diagnostic only; full policy/critic/style loop not yet migrated.",
            )
            (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
            print("NATIVE_PERCEPTION_PASS", json.dumps(result), flush=True)
        finally:
            vec.close()
