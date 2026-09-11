"""Fixed-strength aggregate MLR-gradient correction; no LIBERO, simulator or policy imports.

The caller supplies the original mixed-batch loss and a reference from its old
samples. Projection is global across all trainable parameters, before clipping.
This constrains a raw gradient, not the subsequent AdamW parameter displacement.
"""
import math

import torch


def validate_kappa(value):
    if isinstance(value, bool):
        raise ValueError("kappa must be a finite number in [0, 1], not bool")
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("kappa must be finite and in [0, 1]")
    return value


def mixed_losses(per_sample, task_ids, current_task):
    """Preserve the original batch mean, including its new/old proportions."""
    if per_sample.ndim != 1 or per_sample.numel() == 0:
        raise ValueError("Expected one scalar BC loss per sample, shape [B].")
    ids = torch.as_tensor(task_ids, device=per_sample.device, dtype=torch.long)
    if ids.shape != per_sample.shape:
        raise ValueError("Task IDs and per-sample losses must have the same shape.")
    if bool(((ids < 0) | (ids > int(current_task))).any()):
        raise ValueError("Unknown or future task in training batch.")
    old = ids < int(current_task)
    new = ids == int(current_task)
    n_old = int(old.sum())
    n_new = int(new.sum())
    return (per_sample.mean(), per_sample[new].mean() if n_new else None,
            per_sample[old].mean() if n_old else None, n_new, n_old)


def _inner(left, right):
    # Model gradients remain in their original dtype. Double-precision reduction
    # limits scalar accumulation error without flattening/copying the full model.
    return (left.detach() * right.detach()).sum(dtype=torch.float64)


@torch.no_grad()
def project_parameter_grads_(parameters, reference, enabled=True,
                             reference_norm_floor=1e-12, kappa=0.5):
    """In-place soft correction of the already accumulated mixed gradient.

    kappa=0 leaves gradients (including None) untouched; kappa=1 is the old
    hard projection. Intermediate kappa leaves residual negative conflict.
    Do not apply repeatedly: soft correction is not idempotent.

    None gradients remain None unless the reference actually uses that parameter.
    A zero reference is a logged skip, never a division by (norm2 + epsilon).
    """
    kappa = validate_kappa(kappa)
    parameters = tuple(parameters)
    reference = tuple(reference)
    if not parameters or len(parameters) != len(reference):
        raise ValueError("Parameters and reference gradients must match.")
    if not math.isfinite(reference_norm_floor) or reference_norm_floor < 0:
        raise ValueError("reference_norm_floor must be finite and nonnegative.")
    device = parameters[0].device
    dot = torch.zeros((), dtype=torch.float64, device=device)
    norm2 = dot.clone()
    ref2 = dot.clone()
    for p, r in zip(parameters, reference):
        g = p.grad
        if p.device != device or p.dtype not in (torch.float32, torch.float64):
            raise ValueError("Use single-device FP32/FP64 parameters; AMP is unsupported.")
        if r is not None and (r.shape != p.shape or r.device != device or r.dtype != p.dtype):
            raise ValueError("Reference gradient shape/device/dtype mismatch.")
        if g is not None:
            norm2 += _inner(g, g)
        if r is not None:
            ref2 += _inner(r, r)
            if g is not None:
                dot += _inner(g, r)
    dot_value, norm_value, ref_value = torch.stack((dot, norm2, ref2)).tolist()
    if not all(math.isfinite(x) for x in (dot_value, norm_value, ref_value)):
        raise FloatingPointError("Nonfinite mixed/reference gradient; update aborted.")
    tiny = ref_value <= reference_norm_floor ** 2
    conflict = dot_value < 0.0
    fired = bool(enabled and kappa > 0 and conflict and not tiny)
    coefficient = dot_value / ref_value if fired else 0.0
    if fired:
        coefficient *= kappa
    change2 = dot.new_zeros(())
    after_dot = dot.new_zeros(())
    for p, r in zip(parameters, reference):
        if r is None:
            continue
        if fired:
            old_grad = p.grad
            if old_grad is None:
                p.grad = r.detach().mul(-coefficient)
                change2 += _inner(p.grad, p.grad)
            else:
                # One parameter-sized temporary, not an additional model vector.
                correction = r.detach().mul(coefficient)
                old_grad.sub_(correction)
                change2 += _inner(correction, correction)
        if p.grad is not None:
            after_dot += _inner(p.grad, r)
    change_value, after_value = torch.stack((change2, after_dot)).tolist()
    if not all(math.isfinite(x) for x in (change_value, after_value)):
        raise FloatingPointError("Nonfinite projected gradient; update aborted.")
    norm = math.sqrt(norm_value)
    ref_norm = math.sqrt(ref_value)
    cosine = dot_value / (norm * ref_norm) if norm > 0 and not tiny else None
    return {
        "kappa": kappa, "has_reference": True, "tiny_reference": tiny,
        "conflict": conflict, "projected": fired,
        "dot_before": dot_value, "dot_after": after_value,
        "mixed_grad_norm": norm, "reference_grad_norm": ref_norm,
        "cosine": cosine,
        "correction_ratio": math.sqrt(change_value) / norm if norm > 0 else 0.0,
    }


def backward_mixed_with_reference(loss, old_loss, parameters, loss_scale=1.0,
                                  enabled=True, reference_norm_floor=1e-12, kappa=0.5):
    """One forward, up to two backward traversals, no extra replay sampling."""
    kappa = validate_kappa(kappa)
    parameters = tuple(parameters)
    if not math.isfinite(float(loss_scale)) or loss_scale <= 0:
        raise ValueError("loss_scale must be finite and positive.")
    # Do not reconstruct the weighted loss from separate forwards. This preserves
    # the mixed loss, its RNG realization (dropout) and its original reduction.
    if old_loss is None:
        (loss_scale * loss).backward()
        metrics = project_parameter_grads_(
            parameters, (None,) * len(parameters), enabled=False,
            reference_norm_floor=reference_norm_floor, kappa=kappa)
        metrics.update(has_reference=False, tiny_reference=False)
        return metrics
    reference = torch.autograd.grad(
        loss_scale * old_loss, parameters, retain_graph=True, allow_unused=True)
    (loss_scale * loss).backward()
    return project_parameter_grads_(
        parameters, reference, enabled=enabled,
        reference_norm_floor=reference_norm_floor, kappa=kappa)
