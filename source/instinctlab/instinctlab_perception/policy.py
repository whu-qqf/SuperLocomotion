"""Policy-owned projection; perception features have an explicit stop-gradient."""

import torch
from torch import nn


class PerceptivePolicy(nn.Module):
    def __init__(self, actor, feature_dim=1536, visual_dim=128):
        super().__init__()
        self.actor = actor
        self.projection = nn.Sequential(nn.Linear(feature_dim, 256), nn.ELU(), nn.Linear(256, visual_dim), nn.ELU())

    def forward(self, proprio_history, perception_feature):
        visual = self.projection(perception_feature.detach())
        return self.actor(torch.cat((proprio_history, visual), -1))


def make_hiking_policy(feature_dim=1536):
    """The native Hiking 768-proprio + 128-perception, four-expert actor."""
    from instinct_rl.modules.moe import MoeLayer

    return PerceptivePolicy(MoeLayer(896, 4, 29, "elu", [256, 128, 64]), feature_dim)
