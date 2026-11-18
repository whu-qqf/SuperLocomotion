"""LAFAN pelvis-to-torso conversion preserves geometry and split ownership."""

import hashlib
import json
import numpy as np
import pickle
import torch
from pathlib import Path

import pytest
import pytorch_kinematics as pk
from instinctlab_supermimic.rotations import quat_to_mat

from instinctlab.assets.unitree_g1 import G1_29DOF_POLICY_JOINT_ORDER_V1
from instinctlab.tasks.parkour.supermimic.lafan_reference import convert_lafan_motion, prepare_lafan_reference

ROOT = Path(__file__).resolve().parents[1] / "instinctlab"
URDF = ROOT / "tasks/parkour/urdf/g1_29dof_torsoBase_popsicle_with_shoe.urdf"


def motion():
    names = list(reversed(G1_29DOF_POLICY_JOINT_ORDER_V1))
    q = np.zeros((20, 29), dtype=np.float32)
    q[:, names.index("waist_yaw_joint")] = np.linspace(-0.2, 0.3, 20)
    q[:, names.index("waist_pitch_joint")] = 0.15
    return dict(
        fps=50,
        quaternion_order="xyzw",
        root_pos=np.tile([1.0, 2.0, 0.8], (20, 1)),
        root_rot=np.tile([0.0, 0.0, 0.0, 1.0], (20, 1)),
        dof_pos=q,
        joint_names=names,
    )


def test_root_conversion_preserves_pelvis_and_original_ankles_with_nonzero_waist():
    data = motion()
    chain = pk.build_chain_from_urdf(URDF.read_text())
    converted, report = convert_lafan_motion(data, chain)
    base = torch.eye(4).repeat(20, 1, 1)
    base[:, :3, :3] = quat_to_mat(torch.from_numpy(converted["base_quat_w"]))
    base[:, :3, 3] = torch.from_numpy(converted["base_pos_w"])
    base[:, 2, 3] -= report["constant_shoe_height_offset_m"]
    names = list(converted["joint_names"])
    joints = torch.from_numpy(converted["joint_pos"])
    native = chain.forward_kinematics(joints[:, [names.index(n) for n in chain.get_joint_parameter_names()]])
    expected = torch.eye(4).repeat(20, 1, 1)
    expected[:, :3, 3] = torch.tensor(data["root_pos"], dtype=torch.float32)
    torch.testing.assert_close(base @ native["pelvis"].get_matrix(), expected, atol=5e-7, rtol=1e-5)
    assert report["root_rotation_change_max_rad"] > 0.2
    original = pk.build_chain_from_urdf((ROOT / "assets/resources/unitree_g1/g1_29dof.urdf").read_text())
    original_fk = original.forward_kinematics(joints[:, [names.index(n) for n in original.get_joint_parameter_names()]])
    for side in ("left", "right"):
        foot = side + "_ankle_roll_link"
        torch.testing.assert_close(
            base @ native[foot].get_matrix(), expected @ original_fk[foot].get_matrix(), atol=1e-6, rtol=1e-5
        )
    assert names == list(G1_29DOF_POLICY_JOINT_ORDER_V1)


def test_preparation_excludes_holdout_and_verifies_hashes(tmp_path):
    library = tmp_path / "library"
    library.mkdir()
    path = library / "clip.pkl"
    path.write_bytes(pickle.dumps(motion()))
    source = dict(
        clips=[
            dict(
                split="train",
                file=path.name,
                name="walk",
                category="walk",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            ),
            dict(split="test", file="never_load_this_test_clip.pkl"),
        ],
        split_weights=dict(train=dict(walk=1.0)),
        summary=dict(train=dict(clips=1)),
    )
    (library / "manifest.json").write_text(json.dumps(source))
    report = prepare_lafan_reference(library, URDF, tmp_path / "converted")
    assert len(report["clips"]) == 1 and report["split"] == "train"
    selection = json.loads((tmp_path / "converted/train.yaml").read_text())
    assert selection == dict(selected_files=["walk_retargetted.npz"], motion_weights=[1.0])
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare_lafan_reference(library, URDF, tmp_path / "rejected")
