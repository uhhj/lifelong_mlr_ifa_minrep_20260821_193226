import copy
import csv
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from libero.lifelong.analysis.candidate_adamw_step import (
    candidate_adamw_deltas,
    old_gradient_risk,
)
from libero.lifelong.analysis.first_order_retention_validity import (
    evaluate_mean_loss,
)
from libero.lifelong.analysis.online_first_order_margin import (
    equal_task_gradient,
    label_probe_losses,
    mean_backend_gradient_device,
)


CURRENT_PROBE_SEED = 20260901


def load_h100_cosine_thresholds(path, seeds=(42, 43)):
    requested = {int(seed) for seed in seeds}
    selected = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not (
                row["scope"] == "full"
                and row["horizon"] == "H100"
                and row["task"] == "endpoint_harmful"
                and row["feature"] == "actual_step_cosine"
                and int(row["valid_fold"]) == 1
                and int(row["heldout_seed"]) in requested
            ):
                continue
            seed = int(row["heldout_seed"])
            if seed in selected:
                raise ValueError(f"duplicate valid H100 cosine threshold for seed {seed}")
            selected[seed] = float(row["threshold"])
    missing = requested - set(selected)
    if missing:
        raise ValueError(f"missing valid H100 cosine thresholds for seeds {sorted(missing)}")
    return {seed: selected[seed] for seed in sorted(selected)}


def numeric_path_key(path):
    path = Path(path)
    try:
        return int(path.stem)
    except ValueError:
        return path.name


def select_current_probe(current_paths, seed, stage, size=256):
    ordered = sorted((Path(path) for path in current_paths), key=numeric_path_key)
    if len(ordered) < int(size):
        raise ValueError(f"current task has fewer than {int(size)} feature records")
    selected = random.Random(
        CURRENT_PROBE_SEED + int(seed) * 100 + int(stage)
    ).sample(ordered, int(size))
    return sorted(selected, key=numeric_path_key)


def capture_branch_state(algo, scheduler):
    return {
        "policy": copy.deepcopy(algo.policy.state_dict()),
        "optimizer": copy.deepcopy(algo.optimizer.state_dict()),
        "scheduler": (
            None if scheduler is None else copy.deepcopy(scheduler.state_dict())
        ),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_cpu_rng": torch.get_rng_state().clone(),
        "torch_cuda_rng": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else None
        ),
    }


def restore_branch_state(algo, scheduler, state):
    algo.policy.load_state_dict(copy.deepcopy(state["policy"]))
    algo.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    if scheduler is None:
        if state["scheduler"] is not None:
            raise ValueError("captured scheduler state but restore scheduler is None")
    else:
        if state["scheduler"] is None:
            raise ValueError("missing captured scheduler state")
        scheduler.load_state_dict(copy.deepcopy(state["scheduler"]))
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_cpu_rng"])
    if state["torch_cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_rng"])


def guarded_optimizer_step(
    optimizer,
    named_parameters,
    old_gradient_by_name,
    threshold,
    scale,
    enabled,
    step_fn=None,
    step_args=(),
    step_kwargs=None,
):
    if not (0.0 < float(scale) <= 1.0):
        raise ValueError("guard scale must be in (0, 1]")
    named_parameters = list(named_parameters)
    parameter_by_name = dict(named_parameters)
    before = {
        name: parameter.detach().clone()
        for name, parameter in named_parameters
        if parameter.grad is not None
    }
    candidate_records = candidate_adamw_deltas(optimizer, named_parameters)
    candidate = {
        name: record["delta"] for name, record in candidate_records.items()
    }
    if not candidate:
        raise RuntimeError("AdamW candidate displacement is empty")
    risk = old_gradient_risk(old_gradient_by_name, candidate)
    would_fire = bool(risk["cosine"] > float(threshold))
    guard_fire = bool(enabled and would_fire)
    applied_scale = float(scale) if guard_fire else 1.0
    if step_fn is None:
        step_fn = optimizer.step
    result = step_fn(*(step_args or ()), **(step_kwargs or {}))
    if applied_scale != 1.0:
        with torch.no_grad():
            for name, delta in candidate.items():
                parameter_by_name[name].copy_(before[name] + applied_scale * delta)
    return result, {
        "candidate_step_score": risk["score"],
        "candidate_step_cosine": risk["cosine"],
        "candidate_delta_norm": risk["delta_norm"],
        "old_grad_norm": risk["old_grad_norm"],
        "would_fire": int(would_fire),
        "guard_fire": int(guard_fire),
        "applied_scale": applied_scale,
        "predicted_applied_score": applied_scale * risk["score"],
        "predicted_applied_delta_norm": applied_scale * risk["delta_norm"],
    }


def observe_with_directional_guard(
    algo,
    data,
    current_task_id,
    epoch,
    aggregate_old_gradient,
    old_gradient_names,
    threshold,
    scale,
    enabled,
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
        result, metrics = guarded_optimizer_step(
            optimizer,
            named_trainable,
            old_gradient_by_name,
            threshold,
            scale,
            enabled,
            step_fn=original_step,
            step_args=args,
            step_kwargs=kwargs,
        )
        captured.update(metrics)
        return result

    optimizer.step = wrapped_step
    try:
        training_loss = algo.observe(
            data,
            current_task_id=current_task_id,
            epoch=epoch,
            ifa_info=None,
        )
    finally:
        optimizer.step = original_step
    if not captured:
        raise RuntimeError("optimizer.step interceptor did not fire")
    return training_loss, captured


def directional_guard_step(
    algo,
    data,
    stage,
    epoch,
    score_probes,
    threshold,
    scale,
    enabled,
    score_batch_size=32,
):
    task_gradient_results = {
        int(task_id): mean_backend_gradient_device(
            algo.policy, paths, batch_size=score_batch_size
        )
        for task_id, paths in sorted(score_probes.items())
    }
    if not task_gradient_results:
        raise ValueError("empty old-task score probes")
    names = next(iter(task_gradient_results.values()))["names"]
    if any(result["names"] != names for result in task_gradient_results.values()):
        raise ValueError("backend name order differs across old-task probes")
    aggregate_gradient = equal_task_gradient(
        {
            task_id: result["gradient"]
            for task_id, result in task_gradient_results.items()
        }
    )
    return observe_with_directional_guard(
        algo,
        data,
        current_task_id=stage,
        epoch=epoch,
        aggregate_old_gradient=aggregate_gradient,
        old_gradient_names=names,
        threshold=threshold,
        scale=scale,
        enabled=enabled,
    )


def evaluate_guard_probes(policy, old_label_probes, current_probe, batch_size=32):
    was_training = policy.training
    try:
        old_losses = label_probe_losses(
            policy, old_label_probes, batch_size=batch_size
        )
        current_loss = evaluate_mean_loss(
            policy, current_probe, batch_size=batch_size
        )
    finally:
        policy.train(was_training)
    return {
        "old_task_losses": old_losses,
        "old_mean_loss": statistics.mean(old_losses.values()),
        "current_loss": current_loss,
    }


def paired_drift_metrics(start_old, start_current, control, guard, eps=1e-12):
    control_old_drift = float(control["old"]) - float(start_old)
    guard_old_drift = float(guard["old"]) - float(start_old)
    old_benefit = control_old_drift - guard_old_drift
    control_current_drift = float(control["current"]) - float(start_current)
    guard_current_drift = float(guard["current"]) - float(start_current)
    current_harm = guard_current_drift - control_current_drift
    control_improvement = float(start_current) - float(control["current"])
    preservation = ""
    if control_improvement > 1e-8:
        preservation = (
            float(start_current) - float(guard["current"])
        ) / control_improvement
    return {
        "control_old_drift": control_old_drift,
        "guard_old_drift": guard_old_drift,
        "old_benefit": old_benefit,
        "relative_old_benefit": old_benefit / (abs(float(start_old)) + eps),
        "control_current_drift": control_current_drift,
        "guard_current_drift": guard_current_drift,
        "current_harm": current_harm,
        "relative_current_harm": current_harm / (abs(float(start_current)) + eps),
        "plasticity_preservation": preservation,
    }
