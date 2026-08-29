import math
import statistics

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from libero.lifelong.algos.mlr_temporal_robot_igfa import temporal_head_corrections
from libero.lifelong.analysis.first_order_retention_validity import (
    _load_feature_paths,
    _loss_vector,
    backend_parameter_groups,
    evaluate_mean_loss,
    tuple_dot,
    tuple_norm,
)


def raw_risk_from_dot(dot):
    return -float(dot)


def margin_gate(risks, threshold):
    risks = [float(value) for value in risks]
    return bool(risks and max(risks) > float(threshold))


def equal_task_gradient(task_gradients):
    task_ids = sorted(task_gradients)
    if not task_ids:
        raise ValueError("cannot aggregate empty task gradients")
    length = len(task_gradients[task_ids[0]])
    if any(len(task_gradients[task_id]) != length for task_id in task_ids):
        raise ValueError("task gradient lengths differ")
    return tuple(
        sum(task_gradients[task_id][index] for task_id in task_ids) / len(task_ids)
        for index in range(length)
    )


def actual_step_metrics(gradient, before, after, eps=1e-12):
    if not (len(gradient) == len(before) == len(after)):
        raise ValueError("gradient/snapshot length mismatch")
    displacement = tuple(end - start for start, end in zip(before, after))
    score = float(tuple_dot(gradient, displacement))
    gradient_norm = tuple_norm(gradient)
    delta_norm = tuple_norm(displacement)
    cosine = score / (gradient_norm * delta_norm + eps)
    return {
        "score": score,
        "cosine": cosine,
        "delta_norm": delta_norm,
        "old_grad_norm": gradient_norm,
    }


def mean_backend_gradient_device(policy, feature_paths, batch_size=32):
    was_training = policy.training
    policy.eval()
    device = next(policy.parameters()).device
    head, transformer = backend_parameter_groups(policy)
    named = head + transformer
    parameters = [parameter for _, parameter in named]
    accumulators = [torch.zeros_like(parameter) for parameter in parameters]
    loss_sum = 0.0
    count_sum = 0
    paths = list(feature_paths)
    try:
        for start in range(0, len(paths), int(batch_size)):
            chunk = paths[start : start + int(batch_size)]
            features, actions = _load_feature_paths(chunk, device)
            losses = _loss_vector(policy, features, actions)
            gradients = torch.autograd.grad(
                losses.mean(),
                parameters,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            count = len(chunk)
            loss_sum += float(losses.detach().double().sum().cpu())
            count_sum += count
            for accumulator, gradient in zip(accumulators, gradients):
                if gradient is not None:
                    accumulator.add_(gradient.detach(), alpha=count)
            del features, actions, losses, gradients
    finally:
        policy.train(was_training)
    if count_sum != len(paths) or not count_sum:
        raise RuntimeError("score-gradient probe count mismatch")
    return {
        "mean_loss": loss_sum / count_sum,
        "names": [name for name, _ in named],
        "gradient": tuple(value / count_sum for value in accumulators),
    }


def label_probe_losses(policy, probes, batch_size=32):
    was_training = policy.training
    try:
        return {
            int(task_id): evaluate_mean_loss(policy, paths, batch_size=batch_size)
            for task_id, paths in sorted(probes.items())
        }
    finally:
        policy.train(was_training)


@torch.no_grad()
def capture_backend_device(policy):
    head, transformer = backend_parameter_groups(policy)
    named = head + transformer
    return {
        "names": [name for name, _ in named],
        "values": tuple(parameter.detach().clone() for _, parameter in named),
    }


def shadow_temporal_v1_scores(algo, data, current_task_id):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    was_training = algo.policy.training
    try:
        mapped = algo.map_tensor_to_device(data[0])
        module = algo.policy.module if isinstance(algo.policy, DDP) else algo.policy
        feature = mapped["feature"].squeeze(1)
        actions = mapped["actions"].squeeze(1)
        distribution, _ = module(feature)
        loss_vector = module.policy_head.loss_fn(
            distribution, actions, reduction="none"
        )
        task_ids = torch.tensor(
            [int(value) for value in data[1]],
            dtype=torch.long,
            device=loss_vector.device,
        )
        progress_bins = mapped["progress_bin"].to(loss_vector.device).long().view(-1)
        head_parameters = [
            parameter for parameter in module.policy_head.parameters()
            if parameter.requires_grad
        ]
        _, results = temporal_head_corrections(
            loss_vector,
            task_ids,
            progress_bins,
            current_task_id,
            head_parameters,
            gate_strength=float(getattr(algo.cfg, "robot_igfa_gate_strength", 1.0)),
            n_bins=int(algo.n_temporal_bins),
        )
        raw_risks = [raw_risk_from_dot(result["metrics"]["dot"]) for result in results]
        cosine_risks = [
            -float(result["metrics"]["cosine"].detach()) for result in results
        ]
        return {
            "native_candidate_bins": len(results),
            "native_raw_max": "" if not raw_risks else max(raw_risks),
            "native_raw_mean": "" if not raw_risks else statistics.mean(raw_risks),
            "native_cos_max": "" if not cosine_risks else max(cosine_risks),
            "native_cos_mean": "" if not cosine_risks else statistics.mean(cosine_risks),
            "native_v1_would_gate": "" if not raw_risks else int(max(raw_risks) > 0.0),
        }
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        algo.policy.train(was_training)


def instrumented_observe(
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
    task_gradients = {
        task_id: result["gradient"]
        for task_id, result in task_gradient_results.items()
    }
    aggregate_gradient = equal_task_gradient(task_gradients)
    aggregate_old_loss = statistics.mean(label_losses.values())
    native = shadow_temporal_v1_scores(algo, data, current_task_id)
    before = capture_backend_device(algo.policy)
    if before["names"] != names:
        raise ValueError("snapshot and old-gradient parameter order differ")
    training_loss = algo.observe(
        data, current_task_id=current_task_id, epoch=epoch, ifa_info=None
    )
    after = capture_backend_device(algo.policy)
    if before["names"] != after["names"]:
        raise ValueError("backend parameter order changed across optimizer step")
    aggregate_step = actual_step_metrics(
        aggregate_gradient, before["values"], after["values"]
    )
    task_rows = {}
    for task_id, gradient in task_gradients.items():
        step = actual_step_metrics(gradient, before["values"], after["values"])
        task_rows[task_id] = {
            "old_label_loss": label_losses[task_id],
            "old_score_probe_loss": task_gradient_results[task_id]["mean_loss"],
            "actual_step_task_score": step["score"],
            "actual_step_task_cosine": step["cosine"],
            "actual_step_task_relative": step["score"]
            / max(abs(label_losses[task_id]), 1e-12),
        }
    return {
        "training_loss": training_loss,
        "aggregate_old_loss": aggregate_old_loss,
        **native,
        "actual_step_score": aggregate_step["score"],
        "actual_step_cosine": aggregate_step["cosine"],
        "actual_step_relative": aggregate_step["score"]
        / max(abs(aggregate_old_loss), 1e-12),
        "actual_step_delta_norm": aggregate_step["delta_norm"],
        "actual_step_old_grad_norm": aggregate_step["old_grad_norm"],
        "task_rows": task_rows,
    }


def build_future_cases(anchor_rows, horizon):
    horizon = int(horizon)
    lookup = {
        (int(row["seed"]), int(row["stage"]), int(row["global_update"])): row
        for row in anchor_rows
    }
    output = []
    for row in anchor_rows:
        future_update = int(row["global_update"]) + horizon
        future = lookup.get((int(row["seed"]), int(row["stage"]), future_update))
        if future is None:
            continue
        loss_start = float(row["aggregate_old_loss"])
        actual_delta = float(future["aggregate_old_loss"]) - loss_start
        case = {
                "seed": row["seed"],
                "stage": row["stage"],
                "epoch": row["epoch"],
                "global_update": row["global_update"],
                "future_global_update": future_update,
                "actual_delta": actual_delta,
                "relative_actual_delta": actual_delta / max(abs(loss_start), 1e-12),
                "endpoint_harmful": int(actual_delta > 0.0),
        }
        for field in (
            "native_candidate_bins", "native_raw_max", "native_raw_mean",
            "native_cos_max", "native_cos_mean", "native_v1_would_gate",
            "actual_step_score", "actual_step_cosine", "actual_step_relative",
            "candidate_step_score", "candidate_step_cosine",
            "candidate_actual_delta_cosine", "relative_l2_error",
            "risk_cosine_abs_error",
        ):
            if field in row:
                case[field] = row[field]
        output.append(case)
    return output


def incremental_margin_counts(rows, thresholds):
    counts = {
        "currently_ungated": 0,
        "currently_ungated_harmful": 0,
        "newly_flagged": 0,
        "newly_flagged_harmful": 0,
        "incremental_false_alarms": 0,
        "harmful_still_missed_after_margin": 0,
    }
    for row in rows:
        score = float(row["native_raw_max"])
        harmful = float(row["actual_delta"]) > 0.0
        threshold = float(thresholds[int(row["seed"])])
        current = score > 0.0
        margin = score > threshold
        newly = margin and not current
        counts["currently_ungated"] += int(not current)
        counts["currently_ungated_harmful"] += int((not current) and harmful)
        counts["newly_flagged"] += int(newly)
        counts["newly_flagged_harmful"] += int(newly and harmful)
        counts["incremental_false_alarms"] += int(newly and not harmful)
        counts["harmful_still_missed_after_margin"] += int((not margin) and harmful)
    denominator = counts["currently_ungated_harmful"]
    counts["incremental_harmful_recall"] = (
        counts["newly_flagged_harmful"] / denominator if denominator else float("nan")
    )
    return counts
