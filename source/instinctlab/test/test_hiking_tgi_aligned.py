"""TGI objective, mirror conventions, effective batches and deployment normalization."""

import torch
from dataclasses import replace
from types import SimpleNamespace

from instinctlab_supermimic.depth import DepthAutoencoder, DepthLatentPolicy, load_depth_controller
from instinctlab_supermimic.networks import Critic, WorldModel

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.tasks.parkour.supermimic.native_tgi_policy import (
    mirror_joints,
    mirror_proprio,
    update_tgi_depth_policy,
)


def test_mirror_is_an_involution_in_native_joint_order():
    names = list(G1_29DOF_POLICY_JOINT_ORDER_V1)
    joints, proprio = torch.randn(3, 29), torch.randn(3, 768)
    torch.testing.assert_close(mirror_joints(mirror_joints(joints, names), names), joints)
    torch.testing.assert_close(mirror_proprio(mirror_proprio(proprio, names), names), proprio)
    waist_pitch = names.index("waist_pitch_joint")
    torch.testing.assert_close(mirror_joints(joints, names)[:, waist_pitch], joints[:, waist_pitch])
    angular = proprio[:, :24].reshape(3, 8, 3)
    torch.testing.assert_close(
        mirror_proprio(proprio, names)[:, :24].reshape(3, 8, 3), angular * torch.tensor([-1, 1, -1])
    )


def test_pose_reward_std_matches_tgi_by_speed_mode():
    from instinctlab.tasks.parkour.supermimic.native_rewards import _tgi_pose_std

    names = ["left_hip_pitch_joint", "left_knee_joint", "waist_pitch_joint", "left_wrist_roll_joint"]
    walking = _tgi_pose_std(names, "walking", "cpu", torch.float32)
    running = _tgi_pose_std(names, "running", "cpu", torch.float32)
    torch.testing.assert_close(walking, torch.tensor([0.3, 0.35, 0.1, 0.3]))
    torch.testing.assert_close(running, torch.tensor([0.5, 0.6, 0.2, 0.3]))


def test_normalized_controller_export_keeps_exact_outputs(tmp_path):
    actor, perception = DepthLatentPolicy(normalize_inputs=True), DepthAutoencoder()
    proprio, images = torch.randn(4, 768), torch.rand(4, 8, 18, 32)
    with torch.no_grad():
        actor.proprio_norm.update(proprio)
        actor.latent_norm.update(perception.encode(images))
        expected = actor(proprio, perception.encode(images))
    path = tmp_path / "controller.pt"
    torch.save(
        dict(
            format="hiking_depth_latent_controller_v2",
            depth_shape=(8, 18, 32),
            depth_stride=5,
            actor=actor.state_dict(),
            depth=perception.encoder.state_dict(),
        ),
        path,
    )
    controller = load_depth_controller(path)
    with torch.no_grad():
        torch.testing.assert_close(controller.actor(proprio, controller.encoder(images)), expected, rtol=0, atol=0)


def aligned_fixture():
    from test_hiking_imagined_depth_training import PoseCamera
    from test_hiking_native_task import imagination_fixture

    from instinctlab.tasks.parkour.supermimic.native_imagination import imagine_native

    task, _, _, seed = imagination_fixture()
    task.physics.model = WorldModel(3, 2 * len(task.body_names) + 2, 30, hidden=(32,), aux_dim=58, predict_latent=False)
    task.physics.recorder = SimpleNamespace(joint_names=list(G1_29DOF_POLICY_JOINT_ORDER_V1))
    task.env.cfg.terrain_locomotion = True
    task.locomotion_readout = lambda state, context, contacts, velocity: (
        {"test_support": state[0][:, 2] * 0},
        torch.zeros_like(contacts),
    )
    with torch.no_grad():
        task.physics.model.net[-1].weight.normal_(std=0.003)
    encoder, actor = DepthAutoencoder(), DepthLatentPolicy(detach_latent=False, normalize_inputs=True)
    images = torch.rand(2, 36, 18, 32)
    seed = replace(
        seed,
        perception_state=None,
        visual_latent=encoder.encode(images[:, ::5]).detach(),
        depth_history=images,
        source_env=torch.tensor([0, 0]),
    )
    with torch.no_grad():
        features = imagine_native(task, actor, encoder, seed, horizon=1, imagined_camera=PoseCamera())["value_features"]
    return task, actor, encoder, Critic(features.shape[-1]), seed, PoseCamera()


def test_effective_batch_steps_optimizers_once_and_freezes_dynamics():
    from instinctlab.tasks.parkour.supermimic.native_policy import NativePolicyUpdateCfg

    task, actor, encoder, critic, seed, camera = aligned_fixture()
    before = {k: v.clone() for k, v in task.physics.model.state_dict().items()}
    optimizers = {
        name: torch.optim.Adam(module.parameters())
        for name, module in dict(actor=actor, depth=encoder, critic=critic).items()
    }
    counts = dict.fromkeys(optimizers, 0)

    def count(name):
        def hook(*_):
            counts[name] += 1

        return hook

    for name, opt in optimizers.items():
        opt.register_step_post_hook(count(name))
    report = update_tgi_depth_policy(
        task,
        actor,
        encoder,
        critic,
        seed,
        optimizers["actor"],
        optimizers["depth"],
        optimizers["critic"],
        NativePolicyUpdateCfg(horizon=3, critic_epochs=1, critic_minibatches=1, touchdown_cut_threshold=0.3),
        camera,
        microbatch=1,
    )
    assert counts == dict(actor=1, depth=1, critic=1)
    assert report["effective_actor_batch"] == 2
    assert report["actor_grad_norm"] > 0 and report["depth_task_grad_norm"] > 0
    assert report["symmetry_loss"] >= 0 and torch.isfinite(torch.tensor(report["likelihood_ratio_loss"]))
    assert all(p.grad is None for p in task.physics.model.parameters())
    for key, value in task.physics.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_mean_contact_loss_is_invariant_to_number_of_identical_channels():
    from test_hiking_native_physics import fixture

    physics, inputs = fixture()
    physics.foot_loss_weight = 0
    physics.mean_contact_loss = True
    _, first = physics.window_loss(*inputs)
    physics.model = WorldModel(3, 4, 7, hidden=(32, 32))
    inputs[4] = inputs[4].repeat(1, 1, 2)
    _, second = physics.window_loss(*inputs)
    torch.testing.assert_close(first["contact"], second["contact"])


def test_tgi_reward_shared_with_imagination_and_rewards_default_standing():
    from test_hiking_native_task import fixture

    from instinctlab.tasks.parkour.supermimic.native_rewards import tgi_core_reward

    task, state, context = fixture()
    task.env.cfg.tgi_aligned = True
    task.physics.recorder = SimpleNamespace(joint_names=list(G1_29DOF_POLICY_JOINT_ORDER_V1))
    position, rotation, linear, angular, joints, joint_velocity = state
    state = (
        position,
        torch.eye(3).expand_as(rotation).clone(),
        torch.zeros_like(linear),
        torch.zeros_like(angular),
        context["default_joint_pos"][:, task.joint_ids],
        torch.zeros_like(joint_velocity),
    )
    action = torch.zeros(len(position), 29)
    command = torch.zeros(len(position), 3)
    actual, terms = tgi_core_reward(task, state, action, action, command, context)
    imagined, _ = task.soft_reward(
        state,
        action,
        action,
        command,
        context,
        None,
        torch.zeros(len(position), 2 * len(task.physics.contact_body_names) + 2),
        torch.zeros(len(position), 4),
        torch.zeros(len(position)),
        [],
    )
    torch.testing.assert_close(actual, imagined)
    assert actual.min() > 0
    assert "flat_orientation_l2" not in terms and "undesired_contacts" not in terms


def test_gradient_accumulation_matches_unsplit_effective_batch(monkeypatch):
    import copy

    from instinctlab.tasks.parkour.supermimic import native_tgi_policy
    from instinctlab.tasks.parkour.supermimic.native_policy import NativePolicyUpdateCfg

    task, actor, encoder, critic, seed, camera = aligned_fixture()
    seed.state[:, 0] = torch.tensor([-2.0, 2.0])

    def deterministic_rollout(task, actor, encoder, start, horizon, **kwargs):
        images = start.depth_history[:, ::5].float()
        proprio = start.policy_history.flatten(1)
        mean = actor(proprio, encoder.encode(images))
        sample = mean + 0.2
        reward = (sample.sum(-1) + start.state[:, 0])[:, None].expand(-1, horizon)
        features = torch.stack((mean.sum(-1), start.state[:, 0]), -1)[:, None].expand(-1, horizon + 1, -1)
        return dict(
            task_rewards=reward,
            terminations=reward * 0,
            value_features=features,
            terrain_supported=torch.ones_like(reward, dtype=torch.bool),
            touchdown_events=torch.zeros_like(reward, dtype=torch.bool),
            reward_terms={"test": reward},
            action_log_probs=(-((sample.detach() - mean).square()).sum(-1) / 0.08)[:, None].expand(-1, horizon),
            policy_observations=proprio[:, None].expand(-1, horizon, -1),
            depth_inputs=images[:, None].expand(-1, horizon, -1, -1, -1),
        )

    monkeypatch.setattr(native_tgi_policy, "imagine_native", deterministic_rollout)
    critic = Critic(2)
    copies = []
    for microbatch in (1, 2):
        a, e, v = copy.deepcopy(actor), copy.deepcopy(encoder), copy.deepcopy(critic)
        torch.manual_seed(123)
        update_tgi_depth_policy(
            task,
            a,
            e,
            v,
            seed,
            torch.optim.SGD(a.parameters(), lr=0.001),
            torch.optim.SGD(e.parameters(), lr=0.001),
            torch.optim.SGD(v.parameters(), lr=0.001),
            NativePolicyUpdateCfg(horizon=2, critic_epochs=0, critic_minibatches=1),
            camera,
            microbatch=microbatch,
        )
        copies.append((a, e))
    for first, second in zip(copies[0], copies[1]):
        for p, q in zip(first.parameters(), second.parameters()):
            torch.testing.assert_close(p, q, atol=2e-6, rtol=5e-5)
