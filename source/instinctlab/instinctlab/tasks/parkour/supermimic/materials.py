"""Realized Newton collision-shape material parameters for dynamics replay."""

import torch

import warp as wp


def shape_labels(robot, count):
    """Describe the view's shape axis using link names and each link's collision order."""
    labels = [None] * count
    backend_ids = robot.map_body_ids_to_backend(list(range(len(robot.body_names))))
    for name, body_id in zip(robot.body_names, backend_ids):
        for collision_id, shape_id in enumerate(robot._root_view.body_shapes[body_id]):
            if labels[shape_id] is not None:
                raise ValueError("collision shape belongs to multiple robot links")
            labels[shape_id] = f"{name}:{collision_id}"
    if any(label is None for label in labels):
        raise ValueError("material view contains collision shapes without a robot link")
    return labels


class NewtonShapeMaterials:
    """Bind live values, including startup randomization; never resample or average."""

    def __init__(self, env, robot):
        model = env.sim.backend_manager.get_model()
        self.friction = wp.to_torch(robot._root_view.get_attribute("shape_material_mu", model)[:, 0])
        self.restitution = wp.to_torch(robot._root_view.get_attribute("shape_material_restitution", model)[:, 0])
        if self.friction.shape != self.restitution.shape or self.friction.shape[0] != env.num_envs:
            raise ValueError("Newton material views have inconsistent shapes")
        self.count = self.friction.shape[1]
        self.labels = shape_labels(robot, self.count)
        self.contract = {"encoding": "newton_shape_mu_restitution_v1", "shape_order": self.labels}

    def values(self):
        if not torch.isfinite(self.friction).all() or not torch.isfinite(self.restitution).all():
            raise FloatingPointError("non-finite realized Newton material parameters")
        return self.friction, self.restitution
