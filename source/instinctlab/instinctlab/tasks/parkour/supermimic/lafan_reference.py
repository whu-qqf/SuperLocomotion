"""Verified LAFAN1 locomotion clips in the native torso-root G1 coordinates."""

import argparse
import hashlib
import json
import numpy as np
import pickle
import torch
import xml.etree.ElementTree as ET
from pathlib import Path

import pytorch_kinematics as pk
from instinctlab_supermimic.rotations import quat_to_mat

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1

from .stair_kinematics import matrix_quaternion

FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def convert_lafan_motion(data, chain):
    """Preserve the pelvis world pose through a full SE(3) root conversion.

    Waist joints rotate the native root relative to pelvis: translating only,
    or keeping the original quaternion, gives incorrect torso motion.
    """
    names = list(data["joint_names"])
    order = [names.index(n) for n in G1_29DOF_POLICY_JOINT_ORDER_V1]
    joints = torch.as_tensor(data["dof_pos"], dtype=torch.float32)[:, order]
    if data["quaternion_order"] != "xyzw" or data["fps"] != 50:
        raise ValueError("Expected the hashed 50 Hz XYZW LAFAN1 locomotion library.")
    root = torch.eye(4).repeat(len(joints), 1, 1)
    root[:, :3, :3] = quat_to_mat(torch.as_tensor(data["root_rot"], dtype=torch.float32)[:, [3, 0, 1, 2]])
    root[:, :3, 3] = torch.as_tensor(data["root_pos"], dtype=torch.float32)
    policy_names = list(G1_29DOF_POLICY_JOINT_ORDER_V1)
    fk = chain.forward_kinematics(joints[:, [policy_names.index(n) for n in chain.get_joint_parameter_names()]])
    native_root = root @ torch.linalg.inv(fk["pelvis"].get_matrix())
    torch.testing.assert_close(native_root @ fk["pelvis"].get_matrix(), root, rtol=2e-5, atol=2e-6)
    feet = torch.stack([native_root @ fk[name].get_matrix() for name in FEET], 1)
    points = torch.tensor([[x, y, -0.058] for x in (0.132, -0.054) for y in (-0.018, 0.018)])
    sole = torch.einsum("bfij,pj->bfpi", feet[..., :3, :3], points) + feet[..., None, :3, 3]
    lowest = sole[..., 2].amin((1, 2))
    # Shoe geometry differs from the source robot. This per-clip constant
    # only changes display height; all AMP features and velocities are invariant.
    height_offset = -float(lowest.median())
    native_root[:, 2, 3] += height_offset
    quaternion_xyzw = matrix_quaternion(native_root[:, :3, :3])
    return dict(
        framerate=np.asarray(50),
        joint_names=np.asarray(policy_names),
        joint_pos=joints.numpy(),
        base_pos_w=native_root[:, :3, 3].numpy(),
        base_quat_w=quaternion_xyzw[:, [3, 0, 1, 2]].numpy(),
    ), dict(
        constant_shoe_height_offset_m=height_offset,
        minimum_sole_height_after_offset_m=float(lowest.min()) + height_offset,
        maximum_foot_height_span_m=float((feet[:, :, 2, 3].amax(0) - feet[:, :, 2, 3].amin(0)).max()),
        root_rotation_change_max_rad=float(
            torch.acos(
                ((torch.einsum("bij,bij->b", root[:, :3, :3], native_root[:, :3, :3]) - 1) / 2).clamp(-1, 1)
            ).max()
        ),
    )


def prepare_lafan_reference(library, urdf, output):
    library, urdf, output = Path(library), Path(urdf), Path(output)
    manifest_path = library / "manifest.json"
    source = json.loads(manifest_path.read_text())
    chain = pk.build_chain_from_urdf(urdf.read_text())
    if chain._root.name != "torso_link":
        raise ValueError("Expected the current native torso-root robot.")
    robot = ET.fromstring(urdf.read_text())
    limits = {j.attrib["name"]: j.find("limit") for j in robot.findall("joint") if j.find("limit") is not None}
    names = list(G1_29DOF_POLICY_JOINT_ORDER_V1)
    lower = np.array([float(limits[n].attrib["lower"]) for n in names])
    upper = np.array([float(limits[n].attrib["upper"]) for n in names])
    output.mkdir(parents=True, exist_ok=False)
    clips, files, weights = [], [], []
    for record in source["clips"]:
        if record["split"] != "train":
            continue
        path = library / record["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"LAFAN1 source hash mismatch: {path}")
        with path.open("rb") as stream:
            data = pickle.load(stream)
        motion, metrics = convert_lafan_motion(data, chain)
        q = motion["joint_pos"]
        violation = float(np.maximum(np.maximum(lower - q, q - upper), 0).max())
        if violation > 1e-4 or not all(
            np.isfinite(motion[k]).all() for k in ("joint_pos", "base_pos_w", "base_quat_w")
        ):
            raise ValueError(f"Invalid native G1 reference: {path}, joint violation {violation}")
        name = record["name"] + "_retargetted.npz"
        np.savez_compressed(output / name, **motion)
        files.append(name)
        weights.append(source["split_weights"]["train"][record["name"]])
        clips.append(
            dict(
                file=name,
                sha256=hashlib.sha256((output / name).read_bytes()).hexdigest(),
                source=record["file"],
                source_sha256=record["sha256"],
                category=record["category"],
                native_joint_limit_violation_rad=violation,
                **metrics,
            )
        )
    selection = dict(selected_files=files, motion_weights=weights)
    (output / "train.yaml").write_text(json.dumps(selection, indent=2) + "\n")
    report = dict(
        format="lafan1_native_torso_amp_v1",
        source_library=str(library),
        source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        urdf=str(urdf.resolve()),
        urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
        selection_sha256=hashlib.sha256((output / "train.yaml").read_bytes()).hexdigest(),
        root_conversion="T_world_torso = T_world_pelvis @ inverse(T_torso_pelvis(q)); constant per-clip shoe Z shift",
        source_root="pelvis",
        target_root="torso_link",
        output_quaternion_order="wxyz",
        split="train",
        clips=clips,
        summary=source["summary"]["train"],
        category_weights={
            c: sum(w for w, r in zip(weights, clips) if r["category"] == c) for c in ("stand", "walk", "turn", "run")
        },
        scope="Kinematic conversion, not dynamical tracking validation. Validation/test are not AMP training data.",
    )
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def configure_lafan_reference(cfg, directory):
    directory = Path(directory).resolve()
    report = json.loads((directory / "manifest.json").read_text())
    robot_path = Path(cfg.scene.robot.spawn.asset_path)
    if report["urdf_sha256"] != hashlib.sha256(robot_path.read_bytes()).hexdigest():
        raise ValueError("LAFAN1 conversion used a different robot asset.")
    if hashlib.sha256((directory / "train.yaml").read_bytes()).hexdigest() != report["selection_sha256"]:
        raise ValueError("LAFAN1 motion selection changed after conversion.")
    for clip in report["clips"]:
        if hashlib.sha256((directory / clip["file"]).read_bytes()).hexdigest() != clip["sha256"]:
            raise ValueError(f"Converted LAFAN1 hash mismatch: {clip['file']}")
    for buffer in cfg.scene.motion_reference.motion_buffers.values():
        buffer.path = str(directory)
        buffer.filtered_motion_selection_filepath = str(directory / "train.yaml")
        buffer.subset_selection = None
    cfg.scene.motion_reference.robot_model_path = str(robot_path)
    cfg.lafan_reference_manifest_sha256 = hashlib.sha256((directory / "manifest.json").read_bytes()).hexdigest()
    # Short independent clips must not create artificial discontinuities in
    # ten-frame discriminator windows when the reference silently resamples.
    from isaaclab.managers import ObservationTermCfg

    cfg.terminations.dataset_exhausted.params["obs_term_to_reset"] = [
        f"amp_reference:{name}"
        for name, term in vars(cfg.observations.amp_reference).items()
        if isinstance(term, ObservationTermCfg) and term.history_length > 0
    ]

    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--urdf", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = prepare_lafan_reference(args.library, args.urdf, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "clips"}, indent=2))


if __name__ == "__main__":
    main()
