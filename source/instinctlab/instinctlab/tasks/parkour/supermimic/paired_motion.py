"""Measured motion/contact targets tied to their recorded collision terrain.

Unlike flat motion matching, a staircase clip cannot be moved to a different
terrain or silently continued with another clip. Queries retain both the clip
and its exact frame. Environment origins translate poses, never velocities.
"""

import hashlib
import json
import numpy as np
import torch
import yaml
from pathlib import Path

from instinctlab_supermimic.paired_reference import PairedReferenceTargets

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1

from .lafan_reference import FEET
from .stair_kinematics import NativeLinkKinematics


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PairedMotionLibrary(PairedReferenceTargets):
    """Immutable 50 Hz native torso poses, measured twists and foot forces.

    The loader verifies the recorded-source audit, motion and mesh hashes.
    Contact sidecar hashes are added to the checkpoint contract because the
    older export audit did not store them. They are also checked against the
    original post-step forces: frame zero has no contact measurement.
    """

    def __init__(self, directory, urdf, device="cpu"):
        directory, urdf = Path(directory).resolve(), Path(urdf).resolve()
        metadata = yaml.safe_load((directory / "metadata.yaml").read_text())
        audit = json.loads((directory / "audit.json").read_text())
        if audit["quaternion_npz"] != "wxyz" or audit["quaternion_runtime"] != "xyzw":
            raise ValueError("Paired motion quaternion contract differs.")
        self.directory = directory
        self.device = torch.device(device)
        self.kinematics = NativeLinkKinematics(urdf, device, tracked_bodies=FEET)
        self.joint_names = list(G1_29DOF_POLICY_JOINT_ORDER_V1)
        self.dt = 0.02
        terrain_records = {record["terrain_id"]: record for record in audit["terrains"]}
        self.terrain_ids = torch.tensor(
            [entry["terrain_id"] for entry in metadata["motion_files"]], device=device, dtype=torch.long
        )
        sources, motion_records, meshes = {}, {}, {}
        for terrain in metadata["terrains"]:
            record = terrain_records[terrain["terrain_id"]]
            if record["mesh_source"] != "recorded_collision_heightfield":
                raise ValueError("Paired tracking requires captured collision geometry, not inferred stairs.")
            mesh = directory / terrain["terrain_file"]
            if file_hash(mesh) != record["mesh_sha256"]:
                raise ValueError(f"Paired terrain hash mismatch: {mesh}")
            meshes[terrain["terrain_file"]] = record["mesh_sha256"]
            source = Path(record["source"])
            if source not in sources:
                collection = json.loads((source / "result.json").read_text())
                if collection["robot_urdf_sha256"] != file_hash(urdf):
                    raise ValueError("Paired teacher and learner robot assets differ.")
                if file_hash(source / "physical_trace.pt") != record["source_trace_sha256"]:
                    raise ValueError("Paired source trace hash mismatch.")
                sources[source] = torch.load(source / "physical_trace.pt", map_location="cpu", weights_only=False)
            field = record["collision_heightfield"]
            if file_hash(source / field["file"]) != field["sha256"]:
                raise ValueError("Recorded collision grid hash mismatch.")
            motion_records.update({m["file"]: (record, m) for m in record["motions"]})
        self.frames, lengths, starts, hashes, offset = [], [], [], {}, 0
        for entry in metadata["motion_files"]:
            path = directory / entry["motion_file"]
            terrain, record = motion_records[entry["motion_file"]]
            if terrain["terrain_id"] != entry["terrain_id"] or file_hash(path) != record["sha256"]:
                raise ValueError("Paired clip terrain assignment or motion hash differs.")
            trace = sources[Path(terrain["source"])]
            with np.load(path, allow_pickle=False) as motion, np.load(
                path.with_suffix(".foot_contacts.npz"), allow_pickle=False
            ) as contacts:
                names = motion["joint_names"].tolist()
                if set(names) != set(self.joint_names) or len(names) != 29 or float(motion["framerate"]) != 50:
                    raise ValueError("Paired motions require 29 named native joints at 50 Hz.")
                order = [names.index(name) for name in self.joint_names]
                n = len(motion["joint_pos"])
                if n < 2 or list(contacts["foot_names"]) != list(FEET):
                    raise ValueError("Paired motion needs an ordered two-foot contact sidecar.")
                # Export names encode the original environment index. Read it
                # explicitly, then verify every pose and measurement, not just
                # trust the filename or a historical audit's success flag.
                env_id = int(path.stem.split("_env")[1].split("_")[0])
                self._check_source(motion, contacts, trace, env_id, terrain["origin_w"])
                target = {
                    key: torch.as_tensor(motion[name].copy(), dtype=torch.float32, device=device)
                    for key, name in dict(
                        root_pos="base_pos_w",
                        root_quat="base_quat_w",
                        root_velocity="base_lin_vel_w",
                        root_angular_velocity="base_ang_vel_w",
                        joint_pos="joint_pos",
                        joint_vel="joint_vel",
                    ).items()
                }
                target["joint_pos"] = target["joint_pos"][:, order]
                target["joint_vel"] = target["joint_vel"][:, order]
                target.update(
                    contact=torch.as_tensor(contacts["contact"].copy(), device=device),
                    contact_valid=torch.as_tensor(contacts["valid"].copy(), device=device),
                    force=torch.as_tensor(contacts["force_w"].copy(), device=device),
                    phase=torch.linspace(0, 1, n, device=device),
                    remaining_time=(n - 1 - torch.arange(n, device=device)) * self.dt,
                )
                if not all(torch.isfinite(value).all() for value in target.values()):
                    raise ValueError("Nonfinite paired reference target.")
                if (target["root_quat"].norm(dim=-1) - 1).abs().max() > 1e-4:
                    raise ValueError("Paired reference quaternions are not normalized.")
                links = self.kinematics.links(
                    dict(
                        root_pos_w=target["root_pos"],
                        root_quat_w=target["root_quat"][:, [1, 2, 3, 0]],
                        root_lin_vel_w=target["root_velocity"],
                        root_ang_vel_w=target["root_angular_velocity"],
                        joint_pos=target["joint_pos"],
                        joint_vel=target["joint_vel"],
                    ),
                    root_com=torch.zeros(3, device=device),
                )
                target["foot_pos"] = links["body_link_pos_w"]
                target.update(next_touchdown_targets(target))
                self.frames.append(target)
                lengths.append(n)
                starts.append(offset)
                offset += n
                hashes[path.name] = record["sha256"]
                hashes[path.with_suffix(".foot_contacts.npz").name] = file_hash(path.with_suffix(".foot_contacts.npz"))
        self.frames = {key: torch.cat([clip[key] for clip in self.frames]) for key in self.frames[0]}
        self.lengths = torch.tensor(lengths, device=device)
        self.starts = torch.tensor(starts, device=device)
        self.contract = dict(
            format="native_paired_motion_contacts_v1",
            metadata_sha256=file_hash(directory / "metadata.yaml"),
            audit_sha256=file_hash(directory / "audit.json"),
            urdf_sha256=file_hash(urdf),
            files=hashes,
            terrain_meshes=meshes,
            dt=self.dt,
            root="torso_link",
            quaternion="wxyz",
            contact_alignment="pose_t_force_after_step_t_minus_1_initial_invalid",
            contact_threshold_N=1.0,
        )

    @staticmethod
    def _check_source(motion, contacts, trace, env_id, origin):
        n = len(motion["joint_pos"])
        if list(motion["joint_names"]) != trace["joint_names"] or abs(trace["dt"] - 0.02) > 1e-8:
            raise ValueError("Paired source joint order or timing differs.")
        checks = {
            "joint_pos": trace["joint_pos"][:n, env_id].numpy(),
            "joint_vel": trace["joint_vel"][:n, env_id].numpy(),
            "base_pos_w": trace["root_pos_w"][:n, env_id].numpy() - np.asarray(origin, dtype=np.float32),
            "base_quat_w": trace["root_quat_w"][:n, env_id].numpy()[:, [3, 0, 1, 2]],
            "base_lin_vel_w": trace["root_lin_vel_w"][:n, env_id].numpy(),
            "base_ang_vel_w": trace["root_ang_vel_w"][:n, env_id].numpy(),
        }
        if any(not np.array_equal(motion[name], expected) for name, expected in checks.items()):
            raise ValueError("Motion coordinates differ from the recorded source trajectory.")
        if contacts["valid"].shape != (n,) or contacts["force_w"].shape != (n, 2, 3):
            raise ValueError("Contact sidecar length differs from paired motion.")
        if contacts["valid"][0] or np.any(contacts["force_w"][0]):
            raise ValueError("Initial pose has no preceding physical contact measurement.")
        if not np.array_equal(contacts["force_w"][1:], trace["post_step_foot_force_w"][: n - 1, env_id].numpy()):
            raise ValueError("Contact force timestamps do not match the reference pose.")
        if not np.array_equal(contacts["valid"][1:], trace["post_step_contact_valid"][: n - 1, env_id].numpy()):
            raise ValueError("Contact validity differs from the source trajectory.")
        if not np.array_equal(contacts["contact"], np.linalg.norm(contacts["force_w"], axis=-1) > 1):
            raise ValueError("Contact labels differ from measured force > 1 N.")


def next_touchdown_targets(target):
    """Next measured air-to-contact event per foot, strictly after this frame.

    Stance frames point past the next lift-off, rather than to their current
    support. Missing future events remain invalid; no labels cross clip ends
    or bridge an interval whose contact measurement is unavailable.
    """
    contact, valid, feet = target["contact"], target["contact_valid"], target["foot_pos"]
    count = len(contact)
    indices = torch.arange(count, device=contact.device)
    positions = torch.zeros_like(feet)
    delays = torch.zeros_like(contact, dtype=torch.long)
    available = torch.zeros_like(contact, dtype=torch.bool)
    for foot in range(2):
        event = contact[1:, foot] & ~contact[:-1, foot] & valid[1:] & valid[:-1]
        events = indices[1:][event]
        if not len(events):
            continue
        at = torch.searchsorted(events, indices, right=True)
        end = events[at.clamp_max(len(events) - 1)]
        # Exclude any unknown contact measurement along the future interval.
        missing = (~valid).long().cumsum(0)
        ok = (at < len(events)) & (missing[end] == missing) & valid
        positions[:, foot] = feet[end, foot]
        delays[:, foot] = end - indices
        available[:, foot] = ok
    return dict(touchdown_pos=positions, touchdown_steps=delays, touchdown_valid=available)


class PairedMotionPlayer:
    """Explicit terrain, clip and phase for real collection or replay imagination.

    Reaching the last frame sets ``ended``. Only an environment reset may
    select another clip; advancing never loops or changes the terrain.
    """

    def __init__(self, library, terrain_ids, origins, state=None):
        self.library = library
        self.terrain_ids = terrain_ids.clone()
        self.origins = origins.clone()
        if origins.shape != (len(terrain_ids), 3):
            raise ValueError("Paired playback needs one origin per terrain assignment.")
        self.state = dict(
            clip_id=torch.full_like(terrain_ids, -1),
            frame=torch.zeros_like(terrain_ids),
        )
        if state is not None:
            if set(state) != set(self.state):
                raise ValueError("Paired playback checkpoint needs explicit clip and frame.")
            self.state = {name: value.clone() for name, value in state.items()}
            if any(value.shape != terrain_ids.shape for value in self.state.values()):
                raise ValueError("Paired playback checkpoint batch differs.")
            self._check_assignment()

    def _check_assignment(self):
        clips = self.state["clip_id"]
        if (clips < 0).any() or (clips >= len(self.library.lengths)).any():
            raise ValueError("Paired playback requires reset or a valid checkpoint clip.")
        if not torch.equal(self.library.terrain_ids[clips], self.terrain_ids):
            raise ValueError("Paired playback clip and terrain assignments differ.")
        if (self.state["frame"] < 0).any() or (self.state["frame"] >= self.library.lengths[clips]).any():
            raise ValueError("Paired playback cursor is outside its clip.")

    @torch.no_grad()
    def reset(self, ids, random_phase=False, minimum_remaining=1):
        """Choose only matching clips. First-frame resets are the evaluation default."""
        if minimum_remaining < 1:
            raise ValueError("Paired reset needs at least one remaining transition.")
        selections, frames = [], []
        for env in ids.tolist():
            candidates = (
                ((self.library.terrain_ids == self.terrain_ids[env]) & (self.library.lengths > minimum_remaining))
                .nonzero()
                .flatten()
            )
            if not len(candidates):
                raise ValueError("No matching terrain clip covers the requested reset horizon.")
            clip = candidates[torch.randint(len(candidates), (), device=candidates.device)]
            frame = (
                torch.randint(int(self.library.lengths[clip]) - minimum_remaining, (), device=candidates.device)
                if random_phase
                else torch.zeros((), device=candidates.device, dtype=torch.long)
            )
            selections.append(clip)
            frames.append(frame)
        if selections:
            self.state["clip_id"][ids] = torch.stack(selections)
            self.state["frame"][ids] = torch.stack(frames)

    def window(self, horizon):
        self._check_assignment()
        if horizon < 1:
            raise ValueError("Paired playback needs a positive horizon.")
        frames = self.state["frame"][:, None] + torch.arange(horizon + 1, device=self.library.device)
        return self.library.query(self.state["clip_id"], frames, self.origins, self.terrain_ids)

    @property
    def ended(self):
        return self.state["frame"] >= self.library.lengths[self.state["clip_id"]] - 1

    def advance(self):
        self._check_assignment()
        self.state["frame"] = torch.minimum(self.state["frame"] + 1, self.library.lengths[self.state["clip_id"]] - 1)
        return self.ended
