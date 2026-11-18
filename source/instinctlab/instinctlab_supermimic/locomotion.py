"""Terrain locomotion rewards and event-gated returns, without motion references."""

import torch

LOCOMOTION_WEIGHTS = dict(
    locomotion_single_support=0.25,
    locomotion_flight=-0.5,
    locomotion_clearance=-2.0,
    locomotion_slip=-1.0,
    locomotion_support=-3.0,
    locomotion_stand_contact=-1.0,
    locomotion_support_field=-3.0,
)


def locomotion_terms(
    contact,
    foot_velocity,
    sole_points,
    ground,
    valid,
    velocity,
    dt,
    support_field=None,
    weights=None,
    tgi_recipe=False,
    clearance_ground=None,
    clearance_valid=None,
):
    """Shared measured/predicted readouts; contacts are hard labels or probabilities.

    Sole points and ground are corresponding shoe-underside geometric proxies,
    not collision penetration distances. Invalid terrain is unsupported, never
    zero-height ground. All returned terms are weighted and dt-scaled.
    """
    moving = (velocity[..., :2].norm(dim=-1) + velocity[..., 2].abs() > 0.2).to(contact.dtype)
    active = (
        (velocity[..., :2].norm(dim=-1) + velocity[..., 2].abs() > 0.05).to(contact.dtype) if tgi_recipe else moving
    )
    left, right = contact.unbind(-1)
    single = left * (1 - right) + right * (1 - left)
    airborne = (1 - left) * (1 - right)
    # Relative sole-to-ground gaps preserve support on a matching slope. A
    # foot translated vertically keeps its support score; contact gates it.
    gap = sole_points[..., 2] - ground
    relative_gap = gap - gap.amin(-1, keepdim=True)
    unsupported = torch.sigmoid((relative_gap - 0.03) / 0.01)
    unsupported = torch.where(valid, unsupported, torch.ones_like(unsupported)).mean(-1)
    if tgi_recipe:
        # TGI defines clearance from the foot-center ray. Native Hiking keeps
        # four corner proxies for support, so callers provide a separate
        # center terrain query; the 15-point TGI layout uses index 7 directly.
        if clearance_ground is not None:
            center_gap = sole_points.mean(-2)[..., 2] - clearance_ground
            center_valid = torch.ones_like(center_gap, dtype=torch.bool) if clearance_valid is None else clearance_valid
            clearance = torch.where(center_valid, center_gap, torch.zeros_like(center_gap))
        else:
            if sole_points.shape[-2] <= 7:
                raise ValueError("TGI clearance requires center terrain data or the 15-point shoe layout.")
            center = 7
            clearance = torch.where(
                valid[..., center],
                sole_points[..., center, 2] - ground[..., center],
                torch.zeros_like(ground[..., center]),
            )
    else:
        clearance = torch.where(valid, sole_points[..., 2] - ground, torch.zeros_like(ground)).amin(-1)
    speed = foot_velocity[..., :2].norm(dim=-1)
    terms = dict(
        locomotion_single_support=moving * single,
        locomotion_flight=moving * airborne,
        locomotion_clearance=active
        * ((1 if tgi_recipe else 1 - contact) * ((0.1 if tgi_recipe else 0.08) - clearance).clamp_min(0) * speed).sum(
            -1
        ),
        locomotion_slip=active * (contact * speed.square()).sum(-1),
        locomotion_support=(contact * unsupported).sum(-1),
        locomotion_stand_contact=((1 - active)[..., None] * (1 - contact)).sum(-1),
    )
    if support_field is not None:
        deficit = (1 - support_field).mean(-1)
        terms["locomotion_support_field"] = (contact * deficit).sum(-1)
    weights = LOCOMOTION_WEIGHTS if weights is None else weights
    return {name: value * weights[name] * dt for name, value in terms.items()}, unsupported


def touchdown_events(previous_contact, contact, unsupported, threshold):
    """Detached decision; uncertain new contacts cut future model dependence."""
    with torch.no_grad():
        return (contact * (1 - previous_contact) * unsupported).amax(-1) > threshold


def event_gated_return(reward, terminated, next_value, events, gamma):
    """B x H SHAC return, bootstrapping at events and the horizon.

    The event decision is detached. Critic weights are frozen by the caller;
    value gradients with respect to event-state features remain available.
    """
    if reward.shape != terminated.shape or reward.shape != next_value.shape or reward.shape != events.shape:
        raise ValueError("Event returns require matching B x H tensors.")
    following = next_value[:, -1]
    for step in range(reward.shape[1] - 1, -1, -1):
        continuation = torch.where(events[:, step].detach(), next_value[:, step], following)
        following = reward[:, step] + gamma * (1 - terminated[:, step]) * continuation
    return following
