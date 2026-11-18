"""Raycast feedback uses predicted poses and trains only the intended modules."""

import torch
from dataclasses import replace
from types import SimpleNamespace

import pytest
from instinctlab_supermimic.depth import DepthAutoencoder, DepthLatentPolicy
from instinctlab_supermimic.networks import WM_OUT, WorldModel

from instinctlab.tasks.parkour.supermimic.native_imagination import frozen_parameters, imagine_native
from instinctlab.tasks.parkour.supermimic.native_imagined_camera import ImaginedDepthSequence


class PoseCamera:
    def __init__(self):
        self.calls = []

    @torch.no_grad()
    def render(self, state, env_ids):
        assert torch.isfinite(state).all()
        self.calls.append((state.clone(), env_ids.clone()))
        return dict(image=state[:, 2, None, None].expand(-1, 18, 32).clone())


def test_physical_head_has_no_future_latent_and_keeps_auxiliary_layout():
    from instinctlab_supermimic.physical import unflatten_state

    model = WorldModel(0, 2, 7, z_dim=32, hidden=(32,), aux_dim=58, predict_latent=False)
    assert model.net[-1].out_features == WM_OUT + 2 + 58
    state = torch.zeros(2, 71)
    state[:, 3] = 1
    with torch.no_grad():
        model.net[-1].bias[-58:] = torch.arange(58)
    prediction, contacts, next_latent, auxiliary = model.step_with_aux(
        unflatten_state(state), torch.zeros(2, 29), torch.zeros(2, 7), torch.empty(2, 0), torch.zeros(2, 32)
    )
    assert next_latent is None
    assert contacts.shape == (2, 2)
    torch.testing.assert_close(auxiliary, torch.arange(58).float()[None].expand(2, -1))
    torch.testing.assert_close(prediction[0], state[:, :3])


def test_multistep_physics_trains_encoder_without_future_pixels_or_reset_padding():
    from test_hiking_native_physics import fixture

    physics, inputs = fixture()
    state, target, actions, context, contacts, mask = inputs
    physics.model = WorldModel(0, 2, 7, z_dim=32, hidden=(32,), predict_latent=False)
    with torch.no_grad():
        physics.model.net[-1].weight.normal_(std=0.01)
    physics.foot_loss_weight = 0.0
    encoder = DepthAutoencoder()
    camera = PoseCamera()
    images = torch.rand(2, 36, 18, 32)
    original = images.clone()
    sequence = ImaginedDepthSequence(camera, encoder, images, torch.tensor([9, 3]))
    target[0, 1] = float("nan")
    loss, metrics = physics.window_loss(state, target, actions, context, contacts, mask, latent_observation=sequence)
    loss.backward()
    assert "latent" not in metrics
    assert next(encoder.encoder.parameters()).grad.norm() > 0
    assert all(p.grad is None for p in encoder.decoder.parameters())
    assert physics.model.net[-1].weight.grad.norm() > 0
    assert len(camera.calls) == 1
    assert camera.calls[0][1].tolist() == [3]  # first trajectory ended after one control
    assert not torch.equal(camera.calls[0][0], state[1:2, 1])  # predicted, not teacher-forced pose
    torch.testing.assert_close(images, original, rtol=0, atol=0)
    assert not sequence.images.requires_grad


def test_full_history_advances_one_frame_with_duplicate_source_environments():
    from instinctlab_supermimic.physical import unflatten_state

    class IdentityEncoder:
        def encode(self, images):
            return images[:, :, 0, 0]

    images = torch.arange(36).float()[None, :, None, None].expand(2, -1, 18, 32).clone()
    sequence = ImaginedDepthSequence(PoseCamera(), IdentityEncoder(), images, torch.tensor([4, 4]))
    state = torch.zeros(2, 71, requires_grad=True)
    with torch.no_grad():
        state[:, 2:4] = 1
    active = torch.ones(2, dtype=torch.bool)
    first = sequence(unflatten_state(state), active, 0)
    following = sequence(unflatten_state(state), active, 1)
    torch.testing.assert_close(first[0], torch.arange(0, 36, 5).float())
    torch.testing.assert_close(following[0], torch.tensor([1, 6, 11, 16, 21, 26, 31, 1]).float())
    torch.testing.assert_close(following[0], following[1])
    assert not following.requires_grad


def test_runner_physical_loss_does_not_teacher_force_future_depth():
    from test_hiking_native_physics import fixture

    from instinctlab.tasks.parkour.supermimic.native_latent import NativeLatentRunner

    physics, inputs = fixture()
    physics.model = WorldModel(0, 2, 7, z_dim=32, hidden=(32,), predict_latent=False)
    with torch.no_grad():
        physics.model.net[-1].weight.normal_(std=0.01)
    runner = object.__new__(NativeLatentRunner)
    runner.physics, runner.perception, runner.imagined_camera = physics, DepthAutoencoder(), PoseCamera()
    # Isolate physical supervision from the intentionally separate image loss.
    runner.perception.reconstruction_loss = lambda _: torch.zeros(())
    batch = dict(zip(("state", "next_state", "actions", "context", "contacts", "mask"), inputs))
    batch.update(
        depth=torch.rand(2, 2, 8, 18, 32),
        next_depth=torch.rand(2, 2, 8, 18, 32),
        depth_history=torch.rand(2, 2, 36, 18, 32),
        source_env=torch.tensor([[4, 4], [2, 2]]),
    )
    expected, _ = runner.model_loss(batch, update_statistics=False)
    batch["depth"].fill_(0)
    batch["next_depth"].fill_(1)
    actual, metrics = runner.model_loss(batch, update_statistics=False)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "latent" not in metrics
    actual.backward()
    assert next(runner.perception.encoder.parameters()).grad.norm() > 0


def test_actor_updates_through_physics_with_frozen_encoder_and_detached_camera():
    from test_hiking_native_task import imagination_fixture

    task, _, _, seed = imagination_fixture()
    task.physics.model = WorldModel(
        3, 2 * len(task.body_names) + 2, 30, z_dim=32, hidden=(32,), aux_dim=58, predict_latent=False
    )
    with torch.no_grad():
        task.physics.model.net[-1].weight.normal_(std=0.003)
    encoder, actor, camera = DepthAutoencoder(), DepthLatentPolicy(), PoseCamera()
    images = torch.rand(2, 36, 18, 32)
    seed = replace(
        seed,
        perception_state=None,
        visual_latent=encoder.encode(images[:, ::5]).detach(),
        depth_history=images,
        source_env=torch.tensor([7, 7]),
    )
    before = {k: v.clone() for k, v in encoder.state_dict().items()}
    with frozen_parameters(task.physics.model, encoder):
        imagined = imagine_native(task, actor, encoder, seed, horizon=3, add_noise=False, imagined_camera=camera)
        imagined["task_rewards"].sum().backward()
    assert len(camera.calls) == 3
    assert not imagined["beliefs"].requires_grad
    assert imagined["states"].requires_grad
    assert sum(p.grad.norm() for p in actor.parameters() if p.grad is not None) > 0
    assert all(p.grad is None for p in task.physics.model.parameters())
    assert all(p.grad is None for p in encoder.parameters())
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_reward_configuration_replaces_duplicate_gait_terms():
    from instinctlab.tasks.parkour.supermimic.native_rewards import configure_imagined_depth_rewards

    rewards = SimpleNamespace(
        **{
            name: SimpleNamespace(weight=1.0)
            for name in ("feet_air_time", "feet_slide", "feet_flat_ori", "feet_at_plane", "track_lin_vel_xy_exp")
        }
    )
    cfg = SimpleNamespace(rewards=SimpleNamespace(rewards=rewards))
    configure_imagined_depth_rewards(cfg)
    assert rewards.track_lin_vel_xy_exp.weight == 1.0
    assert rewards.feet_slide.weight == rewards.feet_air_time.weight == 0.0
    assert cfg.locomotion_reward_weights["locomotion_support"] == 0.0
    assert cfg.locomotion_reward_weights["locomotion_support_field"] == -3.0


def test_terrain_transfer_copies_modules_and_rejects_camera_changes(tmp_path):
    from instinctlab.tasks.parkour.supermimic.native_latent import NativeLatentCfg, NativeLatentRunner

    runner = object.__new__(NativeLatentRunner)
    runner.cfg = NativeLatentCfg(imagined_camera=True, terrain_locomotion=True)
    runner.iteration = 0
    runner.replay = SimpleNamespace(controls=0, rows=[])
    runner.contract = dict(
        architecture="native_hiking_physics_imagined_depth_v1",
        physical=dict(terrain_sha256="hiking", robot="g1"),
        camera=dict(height=18, width=32),
    )
    runner.modules = dict(world_model=torch.nn.Linear(2, 1), depth=DepthAutoencoder())
    runner.optimizers = {key: torch.optim.Adam(value.parameters()) for key, value in runner.modules.items()}
    source = dict(runner.contract, physical=dict(terrain_sha256="flat", robot="g1"))
    modules = {
        key: {name: value + 0.1 for name, value in module.state_dict().items()}
        for key, module in runner.modules.items()
    }
    payload = dict(format=source["architecture"], contract=source, modules=modules)
    checkpoint = tmp_path / "flat.pt"
    torch.save(payload, checkpoint)
    runner.initialize_from(checkpoint)
    for key, module in runner.modules.items():
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, modules[key][name], rtol=0, atol=0)
    assert runner.iteration == runner.replay.controls == len(runner.replay.rows) == 0
    assert all(not optimizer.state for optimizer in runner.optimizers.values())
    payload["contract"] = dict(source, camera=dict(height=18, width=64))
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="geometry only"):
        runner.initialize_from(checkpoint)
