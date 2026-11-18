"""Tests for the native Hiking Success-Guided Sampling curriculum."""

import torch

from instinctlab.tasks.parkour.mdp.sgs import SuccessGuidedSampler


def test_sgs_records_duplicate_configurations_in_distinct_ring_slots():
    sampler = SuccessGuidedSampler(3, "cpu", window=4)
    sampler.record(torch.tensor([1, 1, 2]), torch.tensor([1.0, 0.0, 1.0]))
    assert sampler.count.tolist() == [0, 2, 1]
    torch.testing.assert_close(sampler.success_rate(), torch.tensor([0.0, 0.5, 1.0]))


def test_sgs_preserves_terrain_type_prior():
    sampler = SuccessGuidedSampler(
        4,
        "cpu",
        group=torch.tensor([0, 0, 1, 1]),
        group_prior=torch.tensor([0.25, 0.75]),
    )
    probabilities = sampler.probs()
    torch.testing.assert_close(probabilities[:2].sum(), torch.tensor(0.25))
    torch.testing.assert_close(probabilities[2:].sum(), torch.tensor(0.75))


def test_sgs_neighbor_success_is_propagated_from_easier_level():
    sampler = SuccessGuidedSampler(
        2,
        "cpu",
        prev=torch.tensor([-1, 0]),
        neighbor=0.5,
    )
    sampler.record(torch.tensor([0]), torch.tensor([1.0]))
    torch.testing.assert_close(sampler.effective_success(), torch.tensor([1.0, 0.5]))


def test_ring_overflow_keeps_latest_outcomes_and_roundtrips(tmp_path):
    sampler = SuccessGuidedSampler(1, "cpu", window=3)
    sampler.record(torch.zeros(7, dtype=torch.long), torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0]))
    torch.testing.assert_close(sampler.success_rate(), torch.tensor([1 / 3]))
    path = tmp_path / "sgs.pt"
    torch.save(sampler.state_dict(), path)
    restored = SuccessGuidedSampler(1, "cpu", window=3)
    restored.load_state_dict(torch.load(path, weights_only=True))
    torch.testing.assert_close(restored.probs(), sampler.probs())
    assert restored.count.item() == 7


def fixture():
    from types import SimpleNamespace as NS

    origins = torch.zeros(2, 2, 3)
    origins[1, :, 0] = 8
    origins[:, 1, 1] = 8
    generator = NS(size=(8, 8), num_cols=2, sub_terrains={"stairs": NS(proportion=0.25), "rough": NS(proportion=0.75)})
    terrain = NS(
        cfg=NS(terrain_generator=generator),
        terrain_origins=origins,
        terrain_levels=torch.zeros(2, dtype=torch.long),
        terrain_types=torch.zeros(2, dtype=torch.long),
        env_origins=torch.zeros(2, 3),
    )
    pose = torch.zeros(2, 7)
    pose[:, 6] = 1

    def write(*, root_pose, env_ids):
        pose[env_ids] = root_pose

    robot = NS(
        data=NS(root_pos_w=NS(torch=pose[:, :3]), root_quat_w=NS(torch=pose[:, 3:])), write_root_pose_to_sim_index=write
    )

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.terrain = terrain
    command = NS(command=torch.tensor([[1.0, 0, 0], [0.0, 0, 0]]))
    return NS(
        scene=scene,
        num_envs=2,
        device="cpu",
        step_dt=0.02,
        max_episode_length=100,
        episode_length_buf=torch.zeros(2, dtype=torch.long),
        command_manager=NS(get_term=lambda _: command, get_command=lambda _: command.command),
        cfg=NS(events=NS(reset_base=NS(params={"pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05)}}))),
    )


def test_full_episode_distance_and_standing_survival_ignore_last_step_marker():
    from instinctlab.tasks.parkour.mdp.sgs import _episode_success, _state, sgs_pre_step

    env = fixture()
    state = _state(env)
    env.episode_length_buf[:] = 50
    for _ in range(50):
        sgs_pre_step(env)
    env.scene["robot"].data.root_pos_w.torch[0, 0] = 1.7
    # 1m accumulated + 1m unexecuted budget: 1.7m exceeds 80%.
    assert _episode_success(env, torch.arange(2), state).tolist() == [True, False]
    env.scene["robot"].data.root_pos_w.torch[0, 0] = 0.1
    assert not _episode_success(env, torch.arange(2), state)[0]
    env.episode_length_buf[1] = 99
    assert _episode_success(env, torch.arange(2), state)[1]


def test_reset_assignment_spawn_and_zero_length_resets_do_not_record_outcomes():
    from instinctlab.tasks.parkour.mdp.sgs import sgs_reset_spawn, sgs_terrain_levels

    env = fixture()
    sgs_terrain_levels(env, slice(None))
    state = env._hiking_sgs
    sgs_reset_spawn(env, slice(None))
    torch.testing.assert_close(state["start_xy"], env.scene["robot"].data.root_pos_w.torch[:, :2])
    sgs_terrain_levels(env, slice(None))
    assert state["sampler"].count.sum() == 0
    env.episode_length_buf[:] = 1
    sgs_terrain_levels(env, slice(None))
    assert state["sampler"].count.sum() == 2


def test_command_refreshes_type_ranges_and_keeps_sampled_travel_sector():
    from types import SimpleNamespace as NS

    from instinctlab.tasks.parkour.mdp.commands.pose_velocity_command import PoseVelocityCommand
    from instinctlab.tasks.parkour.mdp.sgs import _state

    env = fixture()
    state = _state(env)
    state["config_ids"][:] = torch.tensor([0, 1])  # different direction sectors
    state["travel"][:] = torch.tensor([-2.5, -0.8])
    env.scene.terrain.terrain_types[:] = torch.tensor([0, 1])
    ranges = {
        name: {"lin_vel_x": value, "lin_vel_y": (0.0, 0.0), "ang_vel_z": (-1.0, 1.0)}
        for name, value in [("stairs", (0.5, 0.8)), ("rough", (0.1, 0.2))]
    }
    command = NS(
        _env=env,
        num_envs=2,
        device="cpu",
        terrain=env.scene.terrain,
        robot=env.scene["robot"],
        cfg=NS(velocity_ranges=ranges, random_velocity_terrain=["rough"], rel_standing_envs=0.05),
        progress_start_pos_w=torch.zeros(2, 3),
        pos_command_w=torch.zeros(2, 3),
        valid_targets=torch.tensor([[[[2.0, 2.0, 0.0]], [[2.0, 2.0, 0.0]]]]).expand(2, -1, -1, -1),
        lin_vel_x_range=torch.zeros(2, 2),
        lin_vel_y_range=torch.zeros(2, 2),
        ang_vel_z_range=torch.zeros(2, 2),
        max_command_b=torch.zeros(2, 3),
        random_velocity_indices=torch.zeros(2, dtype=torch.bool),
        is_standing_env=torch.zeros(2, dtype=torch.bool),
    )
    for name in ("x", "y"):
        setattr(command, f"random_lin_vel_{name}_range", torch.zeros(2, 2))
        setattr(command, f"random_lin_vel_{name}", torch.zeros(2))
    command.random_ang_vel_z_range = torch.zeros(2, 2)
    command.random_ang_vel_z = torch.zeros(2)
    PoseVelocityCommand._resample_command(command, torch.arange(2))
    torch.testing.assert_close(command.lin_vel_x_range, torch.tensor([[0.5, 0.8], [0.1, 0.2]]))
    assert command.random_velocity_indices.tolist() == [False, True]
    angle = torch.atan2(command.pos_command_w[:, 1], command.pos_command_w[:, 0])
    sector = ((angle + torch.pi) / (torch.pi / 2)).long()
    assert sector.tolist() == [0, 1]
