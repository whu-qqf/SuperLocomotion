"""Privileged dynamics and deployable depth perception have separate learning paths."""

import torch
from dataclasses import replace

from instinctlab_supermimic.depth import DepthAutoencoder, DepthLatentPolicy
from instinctlab_supermimic.networks import Critic, WorldModel
from instinctlab_supermimic.physical import unflatten_state


def test_terrain_features_follow_native_shoes_and_preserve_coordinate_gradients():
    from test_hiking_native_physics import fixture

    learner, inputs = fixture()
    learner.terrain_features = True
    learner.model = WorldModel(68, 2, 7, hidden=(32,), predict_latent=False)
    state = inputs[0][:, 0].clone().requires_grad_()
    features, valid = learner.terrain_features_with_validity(unflatten_state(state))
    assert features.shape == (2, 68) and valid.all()
    assert learner.model.z_dim == 0 and learner.model.latent_out_dim == 0
    gradient = torch.autograd.grad(features.sum(), state)[0]
    assert gradient[:, 13:42].abs().sum() > 0
    assert gradient[:, 2].abs().min() > 0
    epsilon = 1e-3
    plus, minus = state.detach().clone(), state.detach().clone()
    plus[:, 2] += epsilon
    minus[:, 2] -= epsilon
    difference = (
        learner.model_scan(unflatten_state(plus)).sum(-1) - learner.model_scan(unflatten_state(minus)).sum(-1)
    ) / (2 * epsilon)
    torch.testing.assert_close(gradient[:, 2], difference, atol=2e-3, rtol=1e-3)
    learner.terrain.query = lambda xy: (xy[..., 0] * 0, torch.zeros(xy.shape[:-1], dtype=torch.bool))
    assert not learner.terrain_features_with_validity(unflatten_state(state))[1].any()


def test_policy_return_trains_depth_encoder_while_dynamics_and_critic_stay_frozen():
    from test_hiking_imagined_depth_training import PoseCamera
    from test_hiking_native_task import imagination_fixture

    from instinctlab.tasks.parkour.supermimic.native_policy import NativePolicyUpdateCfg, update_native_policy

    task, _, _, seed = imagination_fixture()
    task.physics.model = WorldModel(3, 2 * len(task.body_names) + 2, 30, hidden=(32,), aux_dim=58, predict_latent=False)
    with torch.no_grad():
        task.physics.model.net[-1].weight.normal_(std=0.003)
    encoder, actor = DepthAutoencoder(), DepthLatentPolicy(detach_latent=False)
    images = torch.rand(2, 36, 18, 32)
    seed = replace(
        seed,
        perception_state=None,
        visual_latent=encoder.encode(images[:, ::5]).detach(),
        depth_history=images,
        source_env=torch.tensor([0, 0]),
    )
    from instinctlab.tasks.parkour.supermimic.native_imagination import imagine_native

    with torch.no_grad():
        features = imagine_native(task, actor, encoder, seed, horizon=1, imagined_camera=PoseCamera())["value_features"]
    critic = Critic(features.shape[-1])
    before = {
        name: {k: v.clone() for k, v in module.state_dict().items()}
        for name, module in dict(model=task.physics.model, critic=critic, encoder=encoder).items()
    }
    report = update_native_policy(
        task,
        actor,
        encoder,
        critic,
        None,
        seed,
        torch.optim.Adam(actor.parameters()),
        torch.optim.Adam(critic.parameters()),
        NativePolicyUpdateCfg(horizon=3, critic_epochs=0, critic_minibatches=1),
        imagined_camera=PoseCamera(),
        depth_optimizer=torch.optim.Adam(encoder.parameters()),
    )
    assert report["depth_task_grad_norm"] > 0
    assert any(not torch.equal(v, before["encoder"][k]) for k, v in encoder.state_dict().items())
    assert all(p.grad is None for p in encoder.decoder.parameters())
    for name, module in dict(model=task.physics.model, critic=critic).items():
        assert all(p.grad is None for p in module.parameters())
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, before[name][key], rtol=0, atol=0)
