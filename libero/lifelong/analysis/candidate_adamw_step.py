import math
import statistics

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from libero.lifelong.analysis.online_first_order_margin import (
    equal_task_gradient,
    label_probe_losses,
    mean_backend_gradient_device,
)


def state_step_value(state):
    value = state.get("step", 0)
    if torch.is_tensor(value):
        return int(value.detach().item())
    return int(value)


@torch.no_grad()
def candidate_adamw_deltas(optimizer, named_parameters):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("candidate diagnosis requires torch.optim.AdamW")
    named_parameters = list(named_parameters)
    name_by_id = {id(parameter): name for name, parameter in named_parameters}
    output = {}
    for group_index, group in enumerate(optimizer.param_groups):
        beta1, beta2 = group["betas"]
        lr = float(group["lr"])
        eps = float(group["eps"])
        weight_decay = float(group["weight_decay"])
        amsgrad = bool(group.get("amsgrad", False))
        maximize = bool(group.get("maximize", False))
        for parameter in group["params"]:
            name = name_by_id.get(id(parameter))
            if name is None or parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            if gradient.is_sparse:
                raise RuntimeError("AdamW candidate does not support sparse grad")
            effective_gradient = -gradient if maximize else gradient
            state = optimizer.state.get(parameter, {})
            step_new = state_step_value(state) + 1
            exp_avg_old = state.get("exp_avg")
            exp_avg_sq_old = state.get("exp_avg_sq")
            if exp_avg_old is None:
                exp_avg_old = torch.zeros_like(parameter)
            if exp_avg_sq_old is None:
                exp_avg_sq_old = torch.zeros_like(parameter)
            exp_avg_new = exp_avg_old.detach().clone()
            exp_avg_new.mul_(beta1).add_(
                effective_gradient, alpha=1.0 - beta1
            )
            exp_avg_sq_new = exp_avg_sq_old.detach().clone()
            exp_avg_sq_new.mul_(beta2).addcmul_(
                effective_gradient,
                effective_gradient,
                value=1.0 - beta2,
            )
            if amsgrad:
                max_old = state.get("max_exp_avg_sq")
                if max_old is None:
                    max_old = torch.zeros_like(parameter)
                variance = max_old.detach().clone()
                torch.maximum(variance, exp_avg_sq_new, out=variance)
            else:
                variance = exp_avg_sq_new
            bias_correction1 = 1.0 - beta1 ** step_new
            bias_correction2 = 1.0 - beta2 ** step_new
            denominator = variance.sqrt()
            denominator.div_(math.sqrt(bias_correction2)).add_(eps)
            step_size = lr / bias_correction1

            # Mirror Torch 1.13 AdamW's two in-place parameter operations on a
            # private clone so float32 rounding matches the realized update.
            predicted_parameter = parameter.detach().clone()
            predicted_parameter.mul_(1.0 - lr * weight_decay)
            predicted_parameter.addcdiv_(
                exp_avg_new, denominator, value=-step_size
            )
            output[name] = {
                "delta": predicted_parameter.sub(parameter.detach()).detach(),
                "group_index": group_index,
                "lr": lr,
                "step_new": step_new,
            }
    return output


def displacement_metrics(candidate, actual, eps=1e-12):
    names = sorted(set(candidate) | set(actual))
    if not names:
        raise ValueError("empty displacement dictionaries")
    candidate_norm2 = 0.0
    actual_norm2 = 0.0
    difference_norm2 = 0.0
    dot = 0.0
    max_abs = 0.0
    for name in names:
        candidate_tensor = candidate.get(name)
        actual_tensor = actual.get(name)
        if candidate_tensor is None:
            candidate_tensor = torch.zeros_like(actual_tensor)
        if actual_tensor is None:
            actual_tensor = torch.zeros_like(candidate_tensor)
        candidate64 = candidate_tensor.detach().double()
        actual64 = actual_tensor.detach().double()
        difference = candidate64 - actual64
        candidate_norm2 += float(torch.sum(candidate64 * candidate64))
        actual_norm2 += float(torch.sum(actual64 * actual64))
        difference_norm2 += float(torch.sum(difference * difference))
        dot += float(torch.sum(candidate64 * actual64))
        max_abs = max(max_abs, float(difference.abs().max()))
    candidate_norm = math.sqrt(candidate_norm2)
    actual_norm = math.sqrt(actual_norm2)
    norm_product = candidate_norm * actual_norm
    if norm_product > eps:
        delta_cosine = dot / norm_product
    else:
        delta_cosine = 1.0 if candidate_norm <= eps and actual_norm <= eps else 0.0
    return {
        "candidate_actual_delta_cosine": delta_cosine,
        "relative_l2_error": math.sqrt(difference_norm2) / (actual_norm + eps),
        "max_abs_delta_error": max_abs,
        "candidate_delta_norm": candidate_norm,
        "actual_delta_norm": actual_norm,
    }


def old_gradient_risk(old_gradient_by_name, displacement_by_name, eps=1e-12):
    if not old_gradient_by_name:
        raise ValueError("empty old-task gradient")
    dot = 0.0
    gradient_norm2 = 0.0
    delta_norm2 = 0.0
    for name, gradient in old_gradient_by_name.items():
        displacement = displacement_by_name.get(name)
        if displacement is None:
            displacement = torch.zeros_like(gradient)
        gradient64 = gradient.detach().double()
        displacement64 = displacement.detach().double()
        dot += float(torch.sum(gradient64 * displacement64))
        gradient_norm2 += float(torch.sum(gradient64 * gradient64))
        delta_norm2 += float(torch.sum(displacement64 * displacement64))
    gradient_norm = math.sqrt(gradient_norm2)
    delta_norm = math.sqrt(delta_norm2)
    return {
        "score": dot,
        "cosine": dot / (gradient_norm * delta_norm + eps),
        "old_grad_norm": gradient_norm,
        "delta_norm": delta_norm,
    }


def observe_with_candidate_step(
    algo,
    data,
    current_task_id,
    epoch,
    aggregate_old_gradient,
    old_gradient_names,
):
    optimizer = algo.optimizer
    module = algo.policy.module if isinstance(algo.policy, DDP) else algo.policy
    named_trainable = [
        (name, parameter)
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    if len(old_gradient_names) != len(aggregate_old_gradient):
        raise ValueError("old-gradient name/value lengths differ")
    old_gradient_by_name = dict(zip(old_gradient_names, aggregate_old_gradient))
    captured = {}
    original_step = optimizer.step

    def wrapped_step(*args, **kwargs):
        before = {
            name: parameter.detach().clone()
            for name, parameter in named_trainable
            if parameter.grad is not None
        }
        candidate_records = candidate_adamw_deltas(optimizer, named_trainable)
        candidate = {
            name: record["delta"] for name, record in candidate_records.items()
        }
        candidate_risk = old_gradient_risk(old_gradient_by_name, candidate)
        result = original_step(*args, **kwargs)
        actual = {
            name: parameter.detach() - before[name]
            for name, parameter in named_trainable
            if name in before
        }
        actual_risk = old_gradient_risk(old_gradient_by_name, actual)
        parity = displacement_metrics(candidate, actual)
        captured.update(
            {
                **parity,
                "candidate_step_score": candidate_risk["score"],
                "candidate_step_cosine": candidate_risk["cosine"],
                "actual_step_score": actual_risk["score"],
                "actual_step_cosine": actual_risk["cosine"],
                "risk_cosine_abs_error": abs(
                    candidate_risk["cosine"] - actual_risk["cosine"]
                ),
                "risk_score_abs_error": abs(
                    candidate_risk["score"] - actual_risk["score"]
                ),
                "risk_sign_agreement": int(
                    (candidate_risk["cosine"] > 0.0)
                    == (actual_risk["cosine"] > 0.0)
                ),
            }
        )
        return result

    optimizer.step = wrapped_step
    try:
        training_loss = algo.observe(
            data, current_task_id=current_task_id, epoch=epoch, ifa_info=None
        )
    finally:
        optimizer.step = original_step
    if not captured:
        raise RuntimeError("optimizer.step interceptor did not fire")
    return training_loss, captured


def candidate_instrumented_observe(
    algo,
    data,
    current_task_id,
    epoch,
    label_probes,
    score_probes,
    label_batch_size=32,
    score_batch_size=32,
):
    label_losses = label_probe_losses(
        algo.policy, label_probes, batch_size=label_batch_size
    )
    task_gradient_results = {
        int(task_id): mean_backend_gradient_device(
            algo.policy, paths, batch_size=score_batch_size
        )
        for task_id, paths in sorted(score_probes.items())
    }
    names = next(iter(task_gradient_results.values()))["names"]
    if any(result["names"] != names for result in task_gradient_results.values()):
        raise ValueError("backend name order differs across old-task probes")
    aggregate_gradient = equal_task_gradient(
        {
            task_id: result["gradient"]
            for task_id, result in task_gradient_results.items()
        }
    )
    training_loss, step = observe_with_candidate_step(
        algo,
        data,
        current_task_id,
        epoch,
        aggregate_gradient,
        names,
    )
    return {
        "training_loss": training_loss,
        "aggregate_old_loss": statistics.mean(label_losses.values()),
        **step,
    }
