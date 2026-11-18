"""Verify native resume and exported inference without rebuilding the scene."""

import json
import torch

from instinctlab_perception.checkpoint import load_hiking_controller

from .native_inference import rollout_controller
from .native_runner import NativeDualRunner


def check_native_checkpoint(runner, model_path, controller_path, output):
    """No optimizer steps: fresh collector, eight resumed controls, twelve play controls.

    A new runner resets the live environment. Run only after the requested
    training budget and final checkpoint, since the old collector is then stale.
    This checks same-scene recovery, not a fresh-process terrain regeneration.
    """
    expected_counts = runner.update_counts.copy()
    resumed = NativeDualRunner(runner.vec, runner.cfg)
    resumed.load(model_path)
    if resumed.update_counts != expected_counts or resumed.iteration != runner.iteration:
        raise AssertionError("Native resume did not restore optimizer counters and iteration.")
    for name in runner.modules:
        expected, restored = runner.modules[name].state_dict(), resumed.modules[name].state_dict()
        for key in expected:
            torch.testing.assert_close(restored[key], expected[key], rtol=0, atol=0)
    segment = resumed.replay.segment
    collection = resumed.collect(8)
    if resumed.replay.segment != segment or resumed.update_counts != expected_counts:
        raise AssertionError("Resumed collection changed its segment or optimizer counters.")
    evaluation = resumed.evaluate_models()
    if resumed.update_counts != expected_counts:
        raise AssertionError("Resumed model evaluation changed optimizer counters.")
    resumed.refresh_runtime()
    seed = resumed.replay.sample_starts(
        resumed.perception_replay, resumed.perception, 4, resumed.cfg.policy.horizon, resumed.env.step_dt
    )
    if not torch.isfinite(seed.perception_state.deter).all():
        raise FloatingPointError("Resumed native replay produced a nonfinite current-P seed.")
    obs, _ = runner.vec.reset()
    actor, runtime = load_hiking_controller(
        controller_path, resumed.env.num_envs, resumed.env.device, resumed.env.step_dt, resumed.recorder.joint_names
    )
    inference = rollout_controller(runner.vec, actor, runtime, obs, 12, resumed.recorder, resumed.actor)
    if resumed.update_counts != expected_counts:
        raise AssertionError("Exported controller inference changed optimizer counters.")
    report = dict(
        passed=True,
        restored_iteration=resumed.iteration,
        collection=collection,
        optimizer_counts_unchanged=expected_counts,
        resume_segment=segment,
        model_evaluation=evaluation,
        inference=inference,
        scope="Same-scene resume and export check; no gait-quality or fresh-process claim.",
    )
    (output / "checkpoint_roundtrip.json").write_text(json.dumps(report, indent=2) + "\n")
    print("NATIVE_CHECKPOINT_ROUNDTRIP_PASS", json.dumps(report), flush=True)
    return report
