"""Compare imagined task readouts to native buffers before Same-Step reset."""

import copy
import torch

from instinctlab_supermimic.physical import unflatten_state

from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass

from .native_task import AMP_TERMS, POLICY_TERMS, NativeTaskHeads
from .native_transitions import NativeHikingRecorderCfg


class NativeTaskParityRecorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.heads = None
        self.errors = {}
        self.failures = {}
        self.controls = 0
        self.pending = None

    def bind(self, physics):
        self.heads = NativeTaskHeads(self._env, physics)

    def compare(self, name, reconstructed, native, atol=2e-4, rtol=2e-4):
        if not torch.isfinite(reconstructed).all() or not torch.isfinite(native).all():
            raise FloatingPointError(f"Nonfinite native task parity readout: {name}.")
        error = float((reconstructed - native).abs().max())
        self.errors[name] = max(error, self.errors.get(name, 0.0))
        try:
            torch.testing.assert_close(reconstructed, native, atol=atol, rtol=rtol)
        except AssertionError as error:
            if name not in self.failures:
                self.failures[name] = dict(
                    reconstructed=reconstructed.detach().cpu().clone(),
                    native=native.detach().cpu().clone(),
                    message=str(error),
                )

    @torch.no_grad()
    def record_pre_step(self):
        if self.heads is None:
            raise RuntimeError("Bind the native task audit after env.reset and before stepping.")
        heads = self.heads
        context, command, sensors = heads.context(), heads.command(), heads.sensor_context()
        state = unflatten_state(self._env.recorder_manager._terms["transition"].state())
        # reset() samples a target without necessarily running _update_command.
        # Compare the generating rule on a copy; preserve the actual latch for
        # rewards, especially the first control after the initial env.reset().
        generated = copy.copy(heads.command_term)
        generated.vel_command_b = generated.vel_command_b.clone()
        generated.pos_command_b = generated.pos_command_b.clone()
        generated.progress_start_pos_w = generated.progress_start_pos_w.clone()
        generated._update_command()
        self.compare("velocity_command_rule", heads.velocity_command(state, command), generated.command)
        self.pending = context, command, sensors
        return None, None

    @torch.no_grad()
    def record_post_step(self):
        env, heads = self._env, self.heads
        if self.pending is None:
            raise RuntimeError("Native task parity has no matching pre-step snapshot.")
        context, command, sensors = self.pending
        transition = env.recorder_manager._terms["transition"].transition
        if transition is None:
            raise RuntimeError("Native task parity must run after the transition recorder and before reset.")
        state = unflatten_state(transition.next_state)
        auxiliary = torch.cat((transition.applied_torque, transition.joint_acceleration), -1)
        data = heads.data(state, context, auxiliary)
        native = env.scene["robot"].data
        for name in (
            "root_lin_vel_b",
            "root_ang_vel_b",
            "projected_gravity_b",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_lin_vel_w",
            "body_link_lin_vel_w",
            "body_ang_vel_w",
        ):
            # Newton propagates twists using differences of world positions;
            # our FK uses root-local displacements. Float32 rounding across
            # Hiking's large terrain affects linear velocities by fractions
            # of a mm/s. Keep the tighter tolerance for all other readouts.
            atol = 5e-4 if name in ("body_lin_vel_w", "body_link_lin_vel_w") else 2e-4
            self.compare(name, getattr(data, name).torch, getattr(native, name).torch, atol=atol)
        # Evaluate observation functions directly: omit corruption/history so
        # this comparison cannot be confused by independently sampled noise.
        frame = heads.proprio(state, transition.action, command["velocity"], context)
        expected = []
        for name in POLICY_TERMS:
            cfg = getattr(env.observation_manager.cfg.policy, name)
            value = cfg.func(env, **cfg.params)
            expected.append(value * (1.0 if cfg.scale is None else cfg.scale))
        self.compare("policy_frame", frame, torch.cat(expected, -1))
        frame = heads.amp_frame(state, context)
        expected = []
        for name in AMP_TERMS:
            cfg = getattr(env.observation_manager.cfg.amp_policy, name)
            value = cfg.func(env, **cfg.params)
            expected.append(value * (1.0 if cfg.scale is None else cfg.scale))
        self.compare("amp_frame", frame, torch.cat(expected, -1))
        total, terms = heads.reward(
            state,
            transition.action,
            transition.previous_action,
            command["velocity"],
            context,
            auxiliary,
            transition.force_history,
            transition.normal_force_history,
            transition.air_time,
            transition.contact_time,
            transition.terminated,
            sensors,
        )
        for name, value in terms.items():
            original = env.reward_manager._termwise_reward_buf["rewards"][name] * env.step_dt
            self.compare(f"reward/{name}", value, original)
        self.compare("reward_total", total, transition.rewards["rewards"])
        self.controls += 1
        self.pending = None
        return None, None

    def result(self):
        if self.controls == 0:
            raise RuntimeError("Native task parity has not observed any real transitions.")
        return dict(
            passed=not self.failures,
            failed_readouts=list(self.failures),
            controls=self.controls,
            reward_terms=len(self.heads.reward_cfg),
            maximum_absolute_errors=self.errors,
            tolerances=dict(default_atol=2e-4, body_linear_velocity_atol=5e-4, rtol=2e-4),
            reward_source="Native reward-manager functions, weights and resolved entity selections.",
            foot_height_source="Original terrain mesh; local face-plane derivatives, not a height raster.",
            scope="True-state readout parity only; does not certify learned F predictions or policy quality.",
        )


@configclass
class NativeTaskAuditRecorderCfg(NativeHikingRecorderCfg):
    task_parity = RecorderTermCfg(class_type=NativeTaskParityRecorder)
