"""Small native Hiking deployment artifact: P and actor only, no physical F."""

import math
import torch
from dataclasses import asdict

from .events import PerceptionRuntime
from .policy import make_hiking_policy
from .rssm import PerceptionCfg, PerceptionModel

LAYOUT = "native_hiking_policy96_history8_depth18x32"


def save_hiking_controller(path, perception, actor, control_dt, joint_names):
    if control_dt <= 0 or len(joint_names) != 29 or len(set(joint_names)) != 29:
        raise ValueError("Native controller needs a positive control period and 29 distinct policy joints.")
    cfg = perception.cfg
    if (cfg.action_dim, cfg.action_steps, cfg.proprio_dim, cfg.image_height, cfg.image_width) != (29, 5, 64, 18, 32):
        raise ValueError("Perception model differs from the native Hiking observation/action layout.")
    torch.save(
        dict(
            format="native_hiking_controller_v1",
            perception_cfg=asdict(perception.cfg),
            observation_layout=LAYOUT,
            control_dt=control_dt,
            joint_names=list(joint_names),
            perception=perception.state_dict(),
            policy=actor.state_dict(),
        ),
        path,
    )


def load_hiking_controller(path, num_envs, device, control_dt, joint_names):
    """Return the actor and its explicit action/camera clock for native play.

    The caller supplies the actual submitted action to runtime.after_action()
    and resets only rows that the simulator reset. No simulator truth enters P.
    """
    packet = torch.load(path, map_location="cpu", weights_only=True)
    if packet["format"] != "native_hiking_controller_v1" or packet["observation_layout"] != LAYOUT:
        raise ValueError("Unsupported native controller format or observation layout.")
    if packet["joint_names"] != list(joint_names):
        raise ValueError("Native controller joint ordering differs from the environment.")
    if not math.isclose(packet["control_dt"], control_dt, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Native controller control period differs from the environment.")
    perception = PerceptionModel(PerceptionCfg(**packet["perception_cfg"])).to(device)
    perception.load_state_dict(packet["perception"])
    actor = make_hiking_policy(perception.cfg.feature_dim).to(device)
    actor.load_state_dict(packet["policy"])
    perception.eval().requires_grad_(False)
    actor.eval().requires_grad_(False)
    return actor, PerceptionRuntime(perception, num_envs, control_dt)
