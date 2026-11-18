"""Model evaluation masks terminals and reports foot/contact errors separately."""

import math
import torch
from types import SimpleNamespace

from instinctlab.tasks.parkour.supermimic.native_latent import NativeLatentRunner


def test_latent_model_evaluation_uses_valid_samples_and_instantaneous_contacts():
    state = torch.zeros(2, 32, 71)
    state[..., 3] = 1
    mask = torch.ones(2, 32, dtype=torch.bool)
    mask[1, 8:] = False
    target = state.clone()
    target[~mask] = torch.nan
    contacts = torch.zeros(2, 32, 6)
    contacts[:, :, -2] = 1  # only the left instantaneous foot is in stance
    contacts[~mask] = torch.nan
    images = torch.zeros(2, 32, 1)
    images[~mask] = torch.nan
    batch = dict(
        state=state,
        next_state=target,
        contacts=contacts,
        mask=mask,
        depth=images,
        next_depth=images,
        actions=torch.zeros(2, 32, 29),
        context=torch.zeros(2, 32, 1),
    )

    def step(current, action, context, scan, latent):
        position, *other = current
        position = position + position.new_tensor([0.1, 0, 0])
        logits = position.new_tensor([0.0, 0.0, 0.0, 0.0, 100.0, -100.0]).expand(len(position), -1)
        return (position, *other), logits, latent

    def state_error(predicted, expected):
        result = predicted[0].new_zeros(len(predicted[0]), 70)
        result[:, :3] = predicted[0] - expected[0]
        return result

    physics = SimpleNamespace(
        model=SimpleNamespace(step=step, state_error=state_error),
        model_scan=lambda current: current[0].new_empty(len(current[0]), 0),
        foot_points=lambda current: current[0][:, None, None, :].expand(-1, 2, 4, -1),
    )
    runner = SimpleNamespace(
        cfg=SimpleNamespace(imagined_camera=False),
        iteration=100,
        train_envs=(0,),
        holdout_envs=(1,),
        physics=physics,
        perception=SimpleNamespace(encode=lambda image: image.new_zeros(len(image), 32)),
        model_batch=lambda *args, **kwargs: batch,
    )
    result = NativeLatentRunner.evaluate_models(runner)
    for split in ("train", "holdout"):
        assert result[split]["8"]["valid"] == 2
        assert result[split]["16"]["valid"] == result[split]["32"]["valid"] == 1
        last = result[split]["32"]
        assert math.isclose(last["feet_rmse_m"], 3.2 / math.sqrt(3), rel_tol=1e-5)
        assert math.isclose(last["root_rmse_m"], last["feet_rmse_m"], rel_tol=1e-6)
        assert last["joint_rmse_rad"] == 0
        assert last["contact_brier"] == 0
        assert last["contact_true_positive"] == 1
        assert last["contact_false_positive"] == last["contact_false_negative"] == 0
        assert last["measured_contact_fraction"] == last["predicted_contact_fraction"] == 0.5
        assert last["contact_strata"]["stance"]["foot_samples"] == 1
        assert last["contact_strata"]["swing"]["foot_samples"] == 1
        assert last["contact_strata"]["touchdown"]["foot_samples"] == 0
        assert last["contact_strata"]["touchdown"]["feet_rmse_m"] is None
