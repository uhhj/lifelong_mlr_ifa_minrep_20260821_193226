"""Candidate CCBP-v3 auxiliary loss: C routing + directional retention.

SYNTHETICALLY TESTED PROTOTYPE ONLY. No real checkpoint or robot evaluation.
Not a complete training adapter. Do not overwrite archived experiments.

Input contract
--------------
Use the same valid old-task windows, frozen teacher cache, group scales, and
frozen per-window weights as the full GOAL41 CCBP-v2 experiment. Student auxiliary
forward is eval-mode WITH gradients; teacher outputs are detached. Preserve the
native BC dropout/RNG stream. Do not estimate weights from the current batch.

L_aux = mean(c_selected + rho * w * directional_anchor_selected
             + beta * original_C_margin_route(logits, c)).
There is NO division by (1 + rho*w), no J-based routing, and no cost-gap softplus.
Deployment stays argmax-logit Gaussian sampling, with NO expert or teacher input.

The anchor is squared distance in standardized (mean, standard deviation)
coordinates to the segment from teacher behavior to the demonstration point.
The demonstration endpoint has zero standard deviation as a metric reference;
it is NOT a Gaussian to instantiate, sample from, or install in the policy.
The segment is NOT a set of proven physically feasible or safe robot actions.
"""
from __future__ import annotations

import math
from typing import Mapping, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

GaussianParams = Tuple[Tensor, Tensor, Tensor]
GROUPS = ((0, 3), (3, 6), (6, 7))


def expert_cost(mu: Tensor, sd: Tensor, target: Tensor, scales: Tensor) -> Tensor:
    """Exactly the existing C-arm grouped expected squared error, [B,K]."""
    target, scales = target.detach(), scales.detach()
    return sum(
        ((mu[:, :, lo:hi] - target[:, None, lo:hi]).square()
         + sd[:, :, lo:hi].square()).mean(-1) / scales[g].square()
        for g, (lo, hi) in enumerate(GROUPS)
    )


def action_normalizer(scales: Tensor) -> Tensor:
    """a/(sqrt(group_dimension)*group_scale) makes Euclidean costs match C."""
    scales = scales.detach()
    return torch.cat([
        scales.new_ones(hi - lo) / (math.sqrt(hi - lo) * scales[g])
        for g, (lo, hi) in enumerate(GROUPS)
    ])


def segment_distance_squared(
    point: Tensor, start: Tensor, end: Tensor
) -> tuple[Tensor, Tensor]:
    """Squared distance to a fixed closed segment, broadcasting leading axes.

    Projection coefficients are detached: the envelope theorem gives the correct
    point gradient 2*(point-projection). Endpoints are always fixed targets.
    Exactly coincident endpoints are handled without a behavioral tolerance.
    Returns (distance_squared, clipped_projection_fraction).
    """
    start, end = start.detach(), end.detach()
    direction = end - start
    denom = direction.square().sum(-1, keepdim=True)
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    fraction = ((point - start) * direction).sum(-1, keepdim=True) / safe_denom
    fraction = torch.where(denom > 0, fraction, torch.zeros_like(fraction))
    fraction = fraction.clamp(0.0, 1.0).detach()
    projection = start + fraction * direction
    return (point - projection).square().sum(-1), fraction.squeeze(-1)


def directional_anchor(
    mu: Tensor, sd: Tensor, teacher_mu: Tensor, teacher_sd: Tensor,
    target: Tensor, scales: Tensor,
) -> tuple[Tensor, Tensor]:
    """[B,K] anchor; teacher_mu/sd are the teacher EXECUTED component [B,7]."""
    normalizer = action_normalizer(scales)
    student = torch.cat((mu * normalizer, sd * normalizer), dim=-1)
    teacher = torch.cat((teacher_mu.detach() * normalizer,
                         teacher_sd.detach() * normalizer), dim=-1)[:, None, :]
    endpoint = torch.cat((target.detach() * normalizer,
                          torch.zeros_like(target)), dim=-1)[:, None, :]
    return segment_distance_squared(student, teacher, endpoint)


def c_margin_route(logits: Tensor, costs: Tensor, margin: float = 0.1) -> Tensor:
    """Original C-arm singleton argmin hinge, including its first-index tie rule.

    This is a soft training penalty, NOT a hard runtime selection constraint.
    Do not call this loss using ground-truth actions at deployment.
    """
    best = costs.detach().argmin(-1)
    good = F.one_hot(best, num_classes=logits.shape[-1]).bool()
    high = logits.masked_fill(~good, -1e9).max(-1).values
    bad = logits.masked_fill(good, -1e9).max(-1).values
    return F.relu(margin + bad - high)


def _validate_params(q: GaussianParams, name: str) -> None:
    mu, sd, logits = q
    if mu.ndim != 3 or mu.shape[-1] != 7 or mu.shape[1] < 2:
        raise ValueError(f'{name} means must have shape [B,K>=2,7].')
    if sd.shape != mu.shape or logits.shape != mu.shape[:2]:
        raise ValueError(f'{name} parameter shapes do not agree.')
    if mu.dtype not in (torch.float32, torch.float64):
        raise ValueError('This prototype is validated only for FP32/FP64.')
    if any(t.device != mu.device or t.dtype != mu.dtype for t in (sd, logits)):
        raise ValueError(f'{name} parameters must share device and dtype.')
    if not all(bool(torch.isfinite(t).all()) for t in q):
        raise ValueError(f'{name} parameters must be finite.')
    if not bool((sd > 0).all()):
        raise ValueError(f'{name} standard deviations must be positive.')


def auxiliary_v3(
    student: GaussianParams,
    target: Tensor,
    teacher: GaussianParams,
    scales: Tensor,
    weights: Tensor,
    *,
    anchor_strength: float = 1.0,
    routing_weight: float = 0.1,
    routing_margin: float = 0.1,
) -> tuple[Tensor, dict[str, float | int], Mapping[str, Tensor]]:
    """Return loss, detached statistics, and three separate loss tensors.

    Initial experimental total (not a tuned or validated setting):
        aligned_bc + 0.1*ifa + 0.1*loss

    weights [B] are frozen train-replay agreement weights in [0,1], NOT safety
    probabilities. The first comparison should retain v2's calibration protocol.
    Changing teacher origin to task-acquisition snapshots is a SEPARATE ablation.

    anchor_strength=0 exactly reduces to the archived C loss (same tie rule).
    """
    for value in (anchor_strength, routing_weight, routing_margin):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Coefficients must be finite and nonnegative.')
    _validate_params(student, 'student')
    _validate_params(teacher, 'teacher')
    mu, sd, logits = student
    tm, ts, tl = (t.detach() for t in teacher)
    b = mu.shape[0]
    if tm.shape[0] != b:
        raise ValueError('Teacher and student batch sizes must agree.')
    if target.shape != (b, 7) or scales.shape != (3,) or weights.shape != (b,):
        raise ValueError('Expected target [B,7], scales [3], weights [B].')
    values = (tm, ts, tl, target, scales, weights)
    if any(t.device != mu.device or t.dtype != mu.dtype for t in values):
        raise ValueError('All inputs must share student device and dtype.')
    if not all(bool(torch.isfinite(t).all()) for t in values):
        raise ValueError('All inputs must be finite.')
    if not bool((scales > 0).all()) or not bool(((weights >= 0) & (weights <= 1)).all()):
        raise ValueError('Scales must be positive; weights must lie in [0,1].')
    target, scales, weights = target.detach(), scales.detach(), weights.detach()
    if b == 0:
        zero = (mu.sum() + sd.sum() + logits.sum()) * 0.0
        return zero, {'count': 0}, {'expert': zero, 'anchor': zero, 'route': zero}
    ix = torch.arange(b, device=mu.device)
    teacher_winner = tl.argmax(-1)
    winner = logits.detach().argmax(-1)
    c = expert_cost(mu, sd, target, scales)
    anchor_all, fraction = directional_anchor(
        mu, sd, tm[ix, teacher_winner], ts[ix, teacher_winner], target, scales
    )
    selected_c = c[ix, winner]
    selected_anchor = anchor_all[ix, winner]
    route = c_margin_route(logits, c, routing_margin)
    weighted_anchor = anchor_strength * weights * selected_anchor
    # Summation order matches C when anchor_strength=0.
    per_sample = (selected_c + weighted_anchor) + routing_weight * route
    loss = per_sample.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Auxiliary loss is nonfinite.')
    with torch.no_grad():
        stats = {
            'count': b,
            'expert_selected_sum': float(selected_c.sum()),
            'directional_anchor_sum': float(selected_anchor.sum()),
            'weighted_anchor_sum': float(weighted_anchor.sum()),
            'route_sum': float(route.sum()),
            'expert_regret_sum': float((selected_c - c.min(-1).values).sum()),
            'projection_fraction_sum': float(fraction[ix, winner].sum()),
            'weight_sum': float(weights.sum()),
            'anchor_bound_max_error': float((anchor_all - c).max()),
        }
    terms = {'expert': selected_c.mean(), 'anchor': weighted_anchor.mean(),
             'route': route.mean()}
    return loss, stats, terms
