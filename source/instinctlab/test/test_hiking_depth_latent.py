"""Visual dynamics must be recurrent, current-encoder supervised and reset safe."""

import torch
from dataclasses import replace

import pytest
from instinctlab_supermimic.depth import (
    DEPTH_SHAPE,
    DepthAutoencoder,
    DepthHistory,
    DepthLatentPolicy,
    load_depth_controller,
)
from instinctlab_supermimic.networks import WorldModel

from instinctlab.tasks.parkour.supermimic.native_imagination import frozen_parameters, imagine_native
from instinctlab.tasks.parkour.supermimic.native_latent import NativeLatentRunner


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 export requires CUDA")
def test_bf16_controller_export_matches_training_precision(tmp_path, monkeypatch):
    from instinctlab_supermimic import networks

    if not torch.cuda.is_bf16_supported():
        pytest.skip("Native BF16 unavailable")
    monkeypatch.setattr(networks, "AMP_DTYPE", torch.bfloat16)
    encoder = DepthAutoencoder().cuda()
    actor = DepthLatentPolicy(normalize_inputs=True).cuda()
    path = tmp_path / "controller.pt"
    torch.save(
        dict(
            format="hiking_depth_latent_controller_v2",
            depth=encoder.encoder.state_dict(),
            actor=actor.state_dict(),
            depth_shape=DEPTH_SHAPE,
            depth_stride=5,
        ),
        path,
    )
    controller = load_depth_controller(path, "cuda")
    proprio, image = torch.randn(32, 768, device="cuda"), torch.rand(32, 18, 32, device="cuda")
    with torch.no_grad():
        expected = actor(proprio, encoder.encode(image[:, None].expand(-1, 8, -1, -1)))
        actual = controller(proprio, image)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_encoder_reconstruction_and_actor_gradient_ownership():
    torch.manual_seed(72)
    model, actor = DepthAutoencoder(), DepthLatentPolicy()
    depth = torch.rand(2, *DEPTH_SHAPE)
    latent = model.encode(depth)
    proprio = torch.randn(2, 768, requires_grad=True)
    actor(proprio, latent).square().mean().backward()
    assert proprio.grad.norm() > 0
    assert all(p.grad is None for p in model.parameters())
    assert next(actor.parameters()).grad.norm() > 0
    before = next(model.encoder.parameters()).detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    model.reconstruction_loss(depth).backward()
    assert next(model.encoder.parameters()).grad.norm() > 0
    assert next(model.decoder.parameters()).grad.norm() > 0
    optimizer.step()
    assert not torch.equal(before, next(model.encoder.parameters()))


def test_depth_history_terminal_then_reset_and_export(tmp_path):
    encoder, actor = DepthAutoencoder(), DepthLatentPolicy()
    history = DepthHistory(torch.zeros(2, 18, 32))
    for index in range(1, 37):
        history.advance(torch.full((2, 18, 32), index / 40))
    terminal = history.stack()
    assert terminal[0, :, 0, 0].tolist() == torch.arange(1, 37, 5).div(40).tolist()
    history.reset(torch.tensor([0]), torch.ones(2, 18, 32))
    assert (history.stack()[0] == 1).all()
    torch.testing.assert_close(terminal[1], history.stack()[1])
    assert terminal[0, -1, 0, 0] < 1  # saved terminal did not become reset pixels
    path = tmp_path / "controller.pt"
    torch.save(
        dict(
            format="hiking_depth_latent_controller_v1",
            depth=encoder.encoder.state_dict(),
            actor=actor.state_dict(),
            depth_shape=DEPTH_SHAPE,
            depth_stride=5,
        ),
        path,
    )
    controller = load_depth_controller(path)
    proprio, image = torch.randn(2, 768), torch.rand(2, 18, 32)
    expected = actor(proprio, encoder.encode(image[:, None].expand(-1, 8, -1, -1)))
    torch.testing.assert_close(controller(proprio, image), expected, rtol=0, atol=0)
    controller(proprio, torch.zeros_like(image), reset_ids=torch.tensor([0]))
    assert (controller.history.stack()[0] == 0).all()


def latent_physics_fixture():
    from test_hiking_native_physics import fixture

    physics, inputs = fixture()
    physics.model = WorldModel(3, 2, 7, z_dim=32, hidden=(32, 32))
    runner = object.__new__(NativeLatentRunner)
    runner.physics, runner.perception = physics, DepthAutoencoder()
    runner.update_counts = dict(model=0)
    state, target, actions, context, contacts, mask = inputs
    batch = dict(
        state=state,
        next_state=target,
        actions=actions,
        context=context,
        contacts=contacts,
        mask=mask,
        depth=torch.rand(2, 2, *DEPTH_SHAPE),
        next_depth=torch.rand(2, 2, *DEPTH_SHAPE),
    )
    return runner, batch


def test_latent_targets_current_encoder_and_masked_padding():
    runner, batch = latent_physics_fixture()
    loss, _ = runner.model_loss(batch, update_statistics=False)
    batch["depth"][0, 1] = float("nan")
    batch["next_depth"][0, 1] = float("nan")
    batch["next_state"][0, 1] = float("nan")
    actual, metrics = runner.model_loss(batch, update_statistics=False)
    torch.testing.assert_close(actual, loss)
    assert torch.isfinite(actual)
    # Dynamics alone must train F, never the encoder through its targets.
    z = runner.perception.encode(torch.nan_to_num(batch["depth"]))
    z1 = runner.perception.encode(torch.nan_to_num(batch["next_depth"]))
    physical = {k: v for k, v in batch.items() if k not in ("depth", "next_depth")}
    dynamics, _ = runner.physics.window_loss(**physical, latent=z, next_latent=z1)
    dynamics.backward()
    assert all(p.grad is None for p in runner.perception.parameters())
    assert runner.physics.model.net[-1].weight.grad.norm() > 0
    # Replay contains pixels; targets are recomputed after an encoder change.
    with torch.no_grad():
        next(runner.perception.encoder.parameters()).add_(0.1)
    changed, _ = runner.model_loss(batch, update_statistics=False)
    assert not torch.isclose(actual, changed)
    assert metrics["latent"] > 0


def test_action_to_latent_to_later_physics_is_not_detached():
    model = WorldModel(1, 1, 1, z_dim=32, hidden=())
    # A deterministic minimal F: a0 -> z0, then z0 -> x displacement.
    with torch.no_grad():
        model.net[-1].weight.zero_()
        model.net[-1].bias.zero_()
        from instinctlab_supermimic.networks import WM_OUT
        from instinctlab_supermimic.physical import INV_DIM

        model.net[-1].weight[WM_OUT + 1, INV_DIM] = 1
        model.net[-1].weight[0, model.in_dim] = 1
    state = (
        torch.zeros(2, 3),
        torch.eye(3).expand(2, -1, -1),
        torch.zeros(2, 3),
        torch.zeros(2, 3),
        torch.zeros(2, 29),
        torch.zeros(2, 29),
    )
    action = torch.ones(2, 29, requires_grad=True)
    with frozen_parameters(model):
        first, _, z = model.step(state, action, torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 32))
        last, _, _ = model.step(first, torch.zeros_like(action), torch.zeros(2, 1), torch.zeros(2, 1), z)
        gradient = torch.autograd.grad(last[0][:, 0].sum(), action)[0]
    assert gradient[:, 0].min() > 0.9
    assert all(p.grad is None for p in model.parameters())


def test_native_imagination_uses_single_wm_and_updates_actor_only():
    from test_hiking_native_task import imagination_fixture

    task, _, _, seed = imagination_fixture()
    task.physics.model = WorldModel(3, 2 * len(task.body_names) + 2, 30, z_dim=32, hidden=(32, 32), aux_dim=58)
    with torch.no_grad():
        task.physics.model.net[-1].weight.normal_(std=0.003)
    autoencoder, actor = DepthAutoencoder(), DepthLatentPolicy()
    seed = replace(seed, perception_state=None, visual_latent=torch.randn(2, 32))
    with frozen_parameters(task.physics.model, autoencoder):
        imagined = imagine_native(task, actor, autoencoder, seed, horizon=3, add_noise=False)
        assert imagined["beliefs"][:, 1:].requires_grad
        assert not torch.equal(imagined["beliefs"][:, 0], imagined["beliefs"][:, -1])
        assert imagined["value_features"].shape[-1] == imagined["physical_features"].shape[-1] + 768 + 670 + 32
        gradient = torch.autograd.grad(imagined["task_rewards"].sum(), tuple(actor.parameters()))
    assert all(torch.isfinite(g).all() for g in gradient)
    assert sum(g.norm() for g in gradient) > 0
    assert all(p.grad is None for p in task.physics.model.parameters())


def test_replay_visual_window_terminal_mask():
    from test_hiking_native_training_replay import collect

    replay, _, _, _ = collect()
    for index, row in enumerate(replay.rows):
        row["depth"] = torch.full((2, 8, 18, 32), index / 20)
        row["next_depth"] = torch.full((2, 8, 18, 32), (index + 0.5) / 20)
    batch = replay.sample_physics(128, 8, (0,), "cpu", extra_fields=("depth", "next_depth"))
    terminal = batch["next_depth"][:, :, 0, 0, 0] == 3.5 / 20
    assert (terminal & batch["mask"]).any()
    for index in range(7):
        assert not (terminal[:, index] & batch["mask"][:, index] & batch["mask"][:, index + 1]).any()


def test_first_real_latents_initialize_scales_instead_of_unit_variance():
    runner, batch = latent_physics_fixture()
    batch["next_depth"] = (batch["depth"] + 1e-3).clamp(max=1)
    runner.model_loss(batch)
    model = runner.physics.model
    assert model.z_std.max() < 0.1
    assert model.dz_std.max() < 0.1
    assert model.z_std.min() >= 0.00099
    assert torch.isfinite(model.z_std).all()


def test_latent_checkpoint_restores_modules_and_starts_new_segment(tmp_path):
    from types import SimpleNamespace

    from test_hiking_native_training_replay import collect

    from instinctlab.tasks.parkour.supermimic.native_latent import NativeLatentCfg
    from instinctlab.tasks.parkour.supermimic.native_training_replay import NativeTrainingReplay

    runner = object.__new__(NativeLatentRunner)
    runner.cfg = NativeLatentCfg()
    runner.reference_bank = None
    runner.contract = dict(architecture="test_latent", camera="fixed")
    runner.iteration = 1
    runner.env = SimpleNamespace(device="cpu")
    runner.replay, _, _, _ = collect()
    runner.modules = dict(depth=DepthAutoencoder(), actor=DepthLatentPolicy())
    runner.optimizers = {name: torch.optim.Adam(module.parameters()) for name, module in runner.modules.items()}
    runner.update_counts = dict(depth=1, actor=1)
    path = tmp_path / "latent.pt"
    runner.save(path)
    before = {name: {k: v.clone() for k, v in module.state_dict().items()} for name, module in runner.modules.items()}
    with torch.no_grad():
        for module in runner.modules.values():
            next(module.parameters()).add_(1)
    runner.iteration = 0
    runner.replay = NativeTrainingReplay(2, capacity_steps=30)
    runner.load(path)
    assert runner.iteration == 1 and runner.replay.segment == 1
    assert runner.episodes.tolist() == [2, 1]
    for name, module in runner.modules.items():
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)


def test_latent_start_sampling_recomputes_encoding_and_excludes_holdout():
    from types import SimpleNamespace

    from test_hiking_native_task import imagination_fixture

    from instinctlab.tasks.parkour.supermimic.native_latent import NativeLatentCfg

    _, _, _, seed = imagination_fixture()
    runner = object.__new__(NativeLatentRunner)
    runner.cfg = NativeLatentCfg(actor_batch=8)
    runner.env = SimpleNamespace(device="cpu", step_dt=0.02)
    runner.perception = DepthAutoencoder()
    runner.holdout_envs = (1,)
    values = {
        k: v
        for k, v in vars(seed).items()
        if k not in ("perception_state", "visual_latent", "depth_history", "source_env")
    }
    pixels = torch.rand(2, *DEPTH_SHAPE)
    runner.replay = SimpleNamespace(rows=[dict(learning=dict(start=values), depth=pixels)])
    first = runner.sample_start()
    torch.testing.assert_close(first.state, seed.state[0:1].expand(8, -1))
    torch.testing.assert_close(first.visual_latent, runner.perception.encode(pixels[0:1]).expand(8, -1))
    with torch.no_grad():
        runner.perception.encoder[-1].bias.add_(0.3)
    second = runner.sample_start()
    torch.testing.assert_close(second.visual_latent, first.visual_latent + 0.3)


def test_without_amp_removes_reference_dependencies_and_keeps_camera():
    from types import SimpleNamespace as NS

    from instinctlab.tasks.parkour.supermimic.native_latent import NativeLatentCfg, configure_without_amp

    camera, proprio_history = object(), object()
    cfg = NS(
        scene=NS(camera=camera, motion_reference=object()),
        observations=NS(amp_reference=object(), amp_policy=proprio_history),
        terminations=NS(dataset_exhausted=object(), root_height=object()),
    )
    assert not NativeLatentCfg().use_amp
    configure_without_amp(cfg)
    assert cfg.scene.motion_reference is None and cfg.observations.amp_reference is None
    assert cfg.terminations.dataset_exhausted is None
    assert cfg.scene.camera is camera and cfg.observations.amp_policy is proprio_history
