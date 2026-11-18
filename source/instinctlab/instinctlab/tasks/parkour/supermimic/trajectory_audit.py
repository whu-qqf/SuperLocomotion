"""Offline reference provenance and terminal-safe teacher trajectory review."""

import argparse
import hashlib
import json
import numpy as np
import torch
import yaml
from pathlib import Path


def quantiles(values):
    values = values.flatten().float()
    if not values.numel():
        return {"count": 0}
    q = torch.quantile(values, torch.tensor([0.05, 0.5, 0.95, 0.99]))
    return {
        "count": values.numel(),
        "mean": float(values.mean()),
        **dict(zip(("p05", "p50", "p95", "p99"), q.tolist())),
        "max": float(values.max()),
    }


def episode_segments(done):
    """Return observed segments with explicit left/right censoring."""
    segments = []
    for env in range(done.shape[1]):
        start = 0
        for end in done[:, env].nonzero().flatten().tolist():
            segments.append(
                {"env": env, "start": start, "end": end, "left_censored": start == 0, "right_censored": False}
            )
            start = end + 1
        if start < done.shape[0]:
            segments.append(
                {
                    "env": env,
                    "start": start,
                    "end": done.shape[0] - 1,
                    "left_censored": start == 0,
                    "right_censored": True,
                }
            )
    return segments


def audit_motions(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    source_path = Path(manifest["source"])
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != manifest["source_sha256"]:
        raise ValueError("motion source hash differs from manifest")
    selected = yaml.safe_load((directory / "train.yaml").read_text())["selected_files"]
    entries = {clip["name"]: clip for clip in manifest["clips"]}
    result = []
    with np.load(source_path, allow_pickle=False) as source:
        for filename in selected:
            path = directory / filename
            name = filename.removesuffix("_retargetted.npz")
            entry = entries[name]
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
                raise ValueError(f"motion clip hash differs: {name}")
            start, end = entry["frames"]
            with np.load(path, allow_pickle=False) as clip:
                np.testing.assert_array_equal(clip["joint_pos"], source["joint_pos"][start:end])
                np.testing.assert_array_equal(clip["base_quat_w"], source["base_quat_w"][start:end])
                np.testing.assert_array_equal(clip["joint_names"], source["joint_names"])
                translated = source["base_pos_w"][start:end] + np.array(entry["translation_m"], dtype=np.float32)
                np.testing.assert_allclose(clip["base_pos_w"], translated, atol=1e-6, rtol=0)
                fps = float(clip["framerate"])
                if fps != float(source["framerate"]):
                    raise ValueError("motion frame rate was changed")
                position = clip["base_pos_w"]
                joint_step = np.abs(np.diff(clip["joint_pos"], axis=0))
                frame, joint = np.unravel_index(joint_step.argmax(), joint_step.shape)
                result.append(
                    {
                        "name": name,
                        "source_frames": [start, end],
                        "frames": len(position),
                        "duration_seconds": (len(position) - 1) / fps,
                        "root_net_dz_m": float(position[-1, 2] - position[0, 2]),
                        "root_z_range_m": float(np.ptp(position[:, 2])),
                        "root_xy_path_m": float(np.linalg.norm(np.diff(position[:, :2], axis=0), axis=-1).sum()),
                        "max_joint_step_rad": float(joint_step.max()),
                        "max_joint_step_name": str(clip["joint_names"][joint]),
                        "max_joint_step_source_frame_pair": [start + int(frame), start + int(frame) + 1],
                        "floor_spread_m_from_preparation": entry["floor_p95_minus_p05_m"],
                        "hash_and_source_pose_parity": True,
                    }
                )
    return {
        "source": str(source_path),
        "selected_clips": result,
        "total_unique_seconds": sum(v["duration_seconds"] for v in result),
    }


def audit_replay(replay, schema, body_names, dt=0.02):
    """Use saved pre-reset reached states; contact motion is a slide proxy only."""
    filled, capacity = replay["filled"], replay["capacity_per_env"]
    start = replay["cursor"] if filled == capacity else 0
    order = (torch.arange(filled) + start) % capacity

    def read(name, final=True):
        return replay["buffer/" + name + ("@final" if final else "")][order]

    done = read("done", False).squeeze(-1).bool()
    contact = read("contact_state")
    root = read("root_pos_w")
    quat = read("root_quat_w")
    quat = quat / quat.norm(dim=-1, keepdim=True)
    # Replay states use XYZW; motion files on disk use WXYZ.
    tilt = torch.acos((1 - 2 * (quat[..., 0].square() + quat[..., 1].square())).clamp(-1, 1))
    origin = read("env_origin", False)
    relative_height = root[..., 2] - origin[..., 2].clamp(max=0)
    torso = read("illegal_torso_contact").squeeze(-1) > 0.5
    low = relative_height < 0.5
    tilted = tilt > 1.0
    failure = torso | low | tilted
    feet = [body_names.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link")]
    velocity = read("body_link_lin_vel_w").reshape(filled, -1, len(body_names), 3)[..., feet, :]
    foot_speed = velocity[..., :2].norm(dim=-1)
    total_contact = contact[..., -9:].reshape(filled, -1, 3, 3)[..., :2].amax(-2) > 0.5
    current_contact = contact[..., 4:6] > 0.5
    stable_contact = total_contact & current_contact & (contact[..., 2:4] >= 0.1) & ~done[..., None]
    segments = episode_segments(done)
    for segment in segments:
        env, first, last = segment["env"], segment["start"], segment["end"]
        trajectory = root[first : last + 1, env]
        segment.update(
            observed_seconds=(last - first + 1) * dt,
            root_xy_net_m=float((trajectory[-1, :2] - trajectory[0, :2]).norm()),
            root_z_net_m=float(trajectory[-1, 2] - trajectory[0, 2]),
            physical_failure_at_end=bool(failure[last, env]) if not segment["right_censored"] else None,
        )
    complete = [s for s in segments if not s["left_censored"] and not s["right_censored"]]
    slide_term = read("task_reward_terms")[..., schema["reward_names"].index("feet_slide")]
    native_slide = slide_term / (schema["reward_weights"]["feet_slide"] * dt)
    # Match the native reward's weighting/integration. Link-origin velocity is
    # not necessarily the COM velocity used by contact_slide, nor contact-point slip.
    proxy_sum = (foot_speed * total_contact).sum(-1)
    raw_clock = contact[..., 4:12].reshape(filled, -1, 4, 2).flip(-2).bool()
    within = raw_clock[..., 1:, :] != raw_clock[..., :-1, :]
    between = raw_clock[1:, :, 0] != raw_clock[:-1, :, -1]
    between_mask = (~done[:-1])[..., None].expand_as(between)
    result = {
        "reached_states": int(done.numel()),
        "aggregate_simulated_seconds": done.numel() * dt,
        "done_events": int(done.sum()),
        "observed_segments": len(segments),
        "complete_start_and_end_segments": len(complete),
        "complete_segment_duration_seconds": quantiles(torch.tensor([s["observed_seconds"] for s in complete])),
        "complete_segments_ge_4s": sum(s["observed_seconds"] >= 4 for s in complete),
        "complete_segments_ge_8s": sum(s["observed_seconds"] >= 8 for s in complete),
        "terminal_flags": {
            "torso_contact": int((torso & done).sum()),
            "tilt_gt_1rad": int((tilted & done).sum()),
            "height_lt_0.5m": int((low & done).sum()),
            "physical_union": int((failure & done).sum()),
            "no_physical_flag": int((~failure & done).sum()),
        },
        "root_tilt_rad": quantiles(tilt),
        "root_relative_height_m": quantiles(relative_height),
        "root_horizontal_speed_mps": quantiles(read("root_lin_vel_w")[..., :2].norm(dim=-1)),
        "contact_foot_link_speed_mps": quantiles(foot_speed[total_contact]),
        "contact_foot_link_speed_gt_0.2mps_fraction": float((foot_speed[total_contact] > 0.2).float().mean()),
        "strict_current_and_total_contact_speed_mps": quantiles(foot_speed[total_contact & current_contact]),
        "nonterminal_contact_duration_ge_0.1s_link_speed_mps": quantiles(foot_speed[stable_contact]),
        "nonterminal_contact_duration_ge_0.1s_speed_gt_0.2mps_fraction": (
            float((foot_speed[stable_contact] > 0.2).float().mean()) if stable_contact.any() else None
        ),
        "native_slide_body_speed_sum_mps": quantiles(native_slide),
        "link_proxy_vs_native_slide_rmse_mps": float((proxy_sum - native_slide).square().mean().sqrt()),
        "double_support_fraction": float(current_contact.all(-1).float().mean()),
        "no_support_fraction": float((~current_contact.any(-1)).float().mean()),
        "within_control_step_clock_switches": int(within.sum()),
        "reset_safe_between_control_step_clock_switches": int(between[between_mask].sum()),
        "control_foot_steps_with_multiple_within_step_clock_switches": int((within.sum(-2) >= 2).sum()),
        "root_in_assigned_origin_center_2_5m_box_fraction": float(
            ((root[..., :2] - origin[..., :2]).abs().amax(-1) <= 1.25).float().mean()
        ),
        "replay_lacks": [
            "teacher reference clip/frame per step",
            "per-reason termination flags",
            "contact-point tangential velocity",
            "per-foot ground surface height",
        ],
    }
    return result, segments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-directory", type=Path, required=True)
    parser.add_argument("--motion-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from instinctlab.tasks.shadowing.supermimic.config.g1.supermimic_plane_cfg import TRACKED_BODIES

    collection = json.loads((args.replay_directory / "collection.json").read_text())
    replay = torch.load(args.replay_directory / "replay.pt", map_location="cpu", weights_only=True)
    results, segments = audit_replay(replay, collection["schema"], TRACKED_BODIES)
    result = {
        "motions": audit_motions(args.motion_directory),
        "trajectories": results,
        "collection_termination_counts_may_overlap": collection["failures"],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "segments.json").write_text(json.dumps(segments, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
