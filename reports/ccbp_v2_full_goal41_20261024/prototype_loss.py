"""CCBP-v2 proposal: continuous correction, quality-weighted anchor, cost-sensitive routing.

This is a locally tested loss prototype, NOT a validated robot-learning method.
It does not change the policy architecture, rollout decoder, buffer, or base BC/IFA.
Use in a NEW experiment directory; do not overwrite the frozen CCBP-v1 experiment.

Training contract:
* Call on the same eligible old-task / ALIGN-valid windows as the original auxiliary.
* Teacher parameters come from the frozen stage-start teacher in eval mode.
* Estimate teacher_cost_median ONCE per protected task from its valid training replay.
* Compute the student auxiliary forward in eval mode WITH gradients; preserve and
  restore module modes/RNG as in the original train.py. Teacher targets are detached.
* Suggested initial screening total loss retains the original outer coefficients:
      loss = aligned_bc + 0.1 * ifa + 0.1 * auxiliary
  Coefficients are an unoptimized initial protocol, not evidence of an optimum.
* Deployment remains argmax(logits), followed by Normal sampling. Expert actions and
  teacher_cost_median are NOT required at deployment.
"""
from __future__ import annotations

import math
from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

GaussianParams = Tuple[Tensor, Tensor, Tensor]
_GROUPS = ((0, 3), (3, 6), (6, 7))


def _validate(params: GaussianParams, y: Tensor, scales: Tensor) -> None:
    mu, sd, logits = params
    if mu.ndim != 3 or mu.shape[-1] != 7 or mu.shape[1] < 2:
        raise ValueError("Means must have shape [N, K>=2, 7].")
    if sd.shape != mu.shape or logits.shape != mu.shape[:2]:
        raise ValueError("Standard deviations / logits do not match means.")
    if y.shape != (mu.shape[0], 7) or scales.shape != (3,):
        raise ValueError("Targets must be [N,7] and group scales [3].")
    if not all(torch.isfinite(t).all().item() for t in (*params, y, scales)):
        raise ValueError("All inputs must be finite.")
    if not (sd > 0).all().item() or not (scales > 0).all().item():
        raise ValueError("Standard deviations and group scales must be positive.")
    if not all(t.device == mu.device for t in (sd, logits, y, scales)):
        raise ValueError("Student parameters, targets and scales must share a device.")


def expert_cost(mu: Tensor, sd: Tensor, y: Tensor, scales: Tensor) -> Tensor:
    """Original normalized expected squared action error; returns [N,K]."""
    return sum(
        ((mu[:, :, lo:hi] - y[:, None, lo:hi]).square()
         + sd[:, :, lo:hi].square()).mean(-1) / scales[g].square()
        for g, (lo, hi) in enumerate(_GROUPS)
    )


def behavior_distance(
    mu: Tensor, sd: Tensor, teacher_mu: Tensor, teacher_sd: Tensor, scales: Tensor
) -> Tensor:
    """Grouped diagonal-Gaussian W2-squared to teacher EXECUTED component [N,K]."""
    return sum(
        ((mu[:, :, lo:hi] - teacher_mu[:, None, lo:hi]).square()
         + (sd[:, :, lo:hi] - teacher_sd[:, None, lo:hi]).square()).mean(-1)
        / scales[g].square()
        for g, (lo, hi) in enumerate(_GROUPS)
    )


@torch.no_grad()
def estimate_teacher_cost_median(
    teacher: GaussianParams, y: Tensor, scales: Tensor
) -> float:
    """Estimate once using ALL valid TRAINING replay for one protected task.

    Only this numerical floor prevents a zero denominator. It is NOT a behavior
    tolerance or safety threshold. Freeze the returned value for the whole stage.
    """
    _validate(teacher, y, scales)
    if len(y) == 0:
        raise ValueError("Cannot calibrate from an empty training replay.")
    mu, sd, logits = teacher
    winner = logits.argmax(-1)
    selected = expert_cost(mu, sd, y, scales).gather(1, winner[:, None]).squeeze(1)
    return float(torch.quantile(selected.double(), 0.5).clamp_min(1e-8))


def regret_route(logits: Tensor, score: Tensor) -> Tensor:
    """Cost-gap-weighted softplus ranking, one value per window.

    Detached costs prevent logits supervision from changing a component's score
    just to make it easier to rank. No fixed margin or good/bad set is used.
    This is a SOFT training loss, not an enforced constraint or a success guarantee.
    """
    if logits.shape != score.shape or logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("Logits and scores must share shape [N,K>=2].")
    frozen = score.detach()
    best_score = frozen.min(-1, keepdim=True).values
    gap = (frozen - best_score).clamp_min(0)
    # Resolve exact cost ties by the highest logit, not by a component index.
    # This mask marks only exact minimizers, NOT a tolerance/acceptable set.
    minimum_mask = frozen == best_score
    best_logit = logits.masked_fill(~minimum_mask, -torch.inf).max(-1, keepdim=True).values
    return (gap * F.softplus(logits - best_logit)).sum(-1) / (logits.shape[1] - 1)


def auxiliary_v2(
    student: GaussianParams,
    y: Tensor,
    teacher: GaussianParams,
    scales: Tensor,
    *,
    teacher_cost_median: float,
    anchor_strength: float = 1.0,
    routing_weight: float = 0.1,
    anchor_mode: Literal["quality", "uniform", "none"] = "quality",
) -> tuple[Tensor, dict[str, float | int]]:
    """Return auxiliary scalar and diagnostic aggregates.

    J_m = (c_m + rho*w_i*d_m)/(1 + rho*w_i).
    w_i = tau/(tau+c_teacher) for quality mode, 1 for uniform, 0 for none.
    L_i = J_argmax(logits) + beta * regret_route(logits, J).

    At rho=1, the expert term has coefficient >=1/2, so there is no cost-budget
    dead zone. A quality weight is agreement with demonstration, NOT a calibrated
    estimate of teacher closed-loop correctness. Even a low-weight anchor may
    oppose some useful deviations. Evaluate against the no-anchor control.
    """
    if anchor_mode not in ("quality", "uniform", "none"):
        raise ValueError("Unsupported anchor mode.")
    if not math.isfinite(teacher_cost_median) or teacher_cost_median <= 0:
        raise ValueError("teacher_cost_median must be a frozen positive scalar.")
    if any(not math.isfinite(v) or v < 0 for v in (anchor_strength, routing_weight)):
        raise ValueError("Loss coefficients must be finite and nonnegative.")
    _validate(student, y, scales)
    _validate(teacher, y, scales)
    mu, sd, logits = student
    tm, ts, tl = (v.detach() for v in teacher)
    if len(y) == 0:
        return (mu.sum() + sd.sum() + logits.sum()) * 0, {"count": 0}
    ix = torch.arange(len(y), device=mu.device)
    tmode = tl.argmax(-1)
    tmu, tsd = tm[ix, tmode], ts[ix, tmode]
    teacher_c = expert_cost(tm, ts, y, scales)[ix, tmode].detach()
    if anchor_mode == "quality":
        weight = (teacher_cost_median / (teacher_cost_median + teacher_c)).detach()
    elif anchor_mode == "uniform":
        weight = torch.ones_like(teacher_c)
    else:
        weight = torch.zeros_like(teacher_c)
    relative_anchor = anchor_strength * weight
    cc = expert_cost(mu, sd, y, scales)
    dd = behavior_distance(mu, sd, tmu, tsd, scales)
    score = (cc + relative_anchor[:, None] * dd) / (1 + relative_anchor[:, None])
    winner = logits.detach().argmax(-1)
    selected_score = score[ix, winner]
    routing = regret_route(logits, score)
    loss = (selected_score + routing_weight * routing).mean()
    if not torch.isfinite(loss).item():
        raise FloatingPointError("CCBP-v2 auxiliary became nonfinite.")
    with torch.no_grad():
        selected_cost = cc[ix, winner]
        regret = selected_score - score.min(-1).values
        stats = {
            "count": len(y),
            "expert_selected_sum": float(selected_cost.sum()),
            "behavior_distance_selected_sum": float(dd[ix, winner].sum()),
            "score_selected_sum": float(selected_score.sum()),
            "routing_loss_sum": float(routing.sum()),
            "selection_regret_sum": float(regret.sum()),
            "selection_regret_max": float(regret.max()),
            "teacher_quality_weight_sum": float(weight.sum()),
            "effective_anchor_weight_sum": float((relative_anchor / (1 + relative_anchor)).sum()),
        }
    return loss, stats
