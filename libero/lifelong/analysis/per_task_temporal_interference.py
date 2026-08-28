import math
import random
from pathlib import Path

import torch

from libero.lifelong.algos.mlr import canonical_task_record_id
from libero.lifelong.algos.mlr_static_robot_igfa import signed_replay_projection


def _numeric_path_key(path):
    path = Path(path)
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def reconstruct_stage_replay_paths(feature_root, seed, old_task_ids, rate=0.5):
    """Reconstruct the source paths selected by update_buffer_equal_quota_10."""
    feature_root = Path(feature_root)
    selected = {}
    for task_id in old_task_ids:
        paths = sorted(
            feature_root.joinpath(str(task_id)).glob("*.pt"),
            key=_numeric_path_key,
        )
        if not paths:
            raise FileNotFoundError(
                f"no feature records for replay task {task_id}: {feature_root}"
            )
        sample_count = max(1, math.ceil(len(paths) * float(rate)))
        rng = random.Random(int(seed))
        selected[int(task_id)] = (
            list(paths)
            if len(paths) <= sample_count
            else rng.sample(paths, sample_count)
        )
    return selected


def load_temporal_bin_map(path, expected_seed, expected_bins=5):
    import json

    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload["seed"]) != int(expected_seed):
        raise ValueError(
            f"sidecar seed {payload['seed']} does not match {expected_seed}"
        )
    if int(payload["n_bins"]) != int(expected_bins):
        raise ValueError(
            f"sidecar bins {payload['n_bins']} does not match {expected_bins}"
        )
    return {str(key): int(value) for key, value in payload["records"].items()}


def make_record(path, temporal_bin_map, expected_task=None):
    task_id, record_id = canonical_task_record_id(path)
    if expected_task is not None and task_id != int(expected_task):
        raise ValueError(f"expected task {expected_task}, got {task_id}: {path}")
    key = f"{task_id}:{record_id}"
    if key not in temporal_bin_map:
        raise KeyError(f"missing temporal sidecar key {key}: {path}")
    return {
        "path": str(path),
        "task_id": task_id,
        "record_id": record_id,
        "progress_bin": int(temporal_bin_map[key]),
    }


def build_stage_records(feature_root, seed, current_task, temporal_bin_map, rate=0.5):
    feature_root = Path(feature_root)
    current_paths = sorted(
        feature_root.joinpath(str(current_task)).glob("*.pt"),
        key=_numeric_path_key,
    )
    if not current_paths:
        raise FileNotFoundError(f"no current-task features: task {current_task}")
    replay_paths = reconstruct_stage_replay_paths(
        feature_root,
        seed,
        range(6, int(current_task)),
        rate=rate,
    )
    records = [
        make_record(path, temporal_bin_map, expected_task=current_task)
        for path in current_paths
    ]
    for task_id in sorted(replay_paths):
        records.extend(
            make_record(path, temporal_bin_map, expected_task=task_id)
            for path in replay_paths[task_id]
        )
    return records, replay_paths


def deterministic_diagnostic_batches(
    records,
    batch_size,
    batch_count,
    sample_seed,
):
    """Yield full deterministic batches, reshuffling for another cycle as needed."""
    records = list(records)
    if len(records) < int(batch_size):
        raise ValueError("diagnostic dataset is smaller than one full batch")
    rng = random.Random(int(sample_seed))
    batches = []
    while len(batches) < int(batch_count):
        permutation = list(records)
        rng.shuffle(permutation)
        for start in range(0, len(permutation) - int(batch_size) + 1, int(batch_size)):
            batches.append(permutation[start : start + int(batch_size)])
            if len(batches) == int(batch_count):
                break
    return batches


def _squeeze_saved_tensor(tensor, target_ndim):
    while tensor.ndim > target_ndim and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    return tensor


def load_record_batch(records, device):
    features = []
    actions = []
    task_ids = []
    progress_bins = []
    for record in records:
        saved = torch.load(record["path"], map_location="cpu")
        feature = _squeeze_saved_tensor(saved["feature"], 3)
        action = _squeeze_saved_tensor(saved["actions"], 2)
        if tuple(feature.shape) != (8, 4, 512):
            raise ValueError(
                f"unexpected feature shape {tuple(feature.shape)}: {record['path']}"
            )
        if tuple(action.shape) != (8, 7):
            raise ValueError(
                f"unexpected action shape {tuple(action.shape)}: {record['path']}"
            )
        features.append(feature)
        actions.append(action)
        task_ids.append(int(record["task_id"]))
        progress_bins.append(int(record["progress_bin"]))
    return (
        torch.stack(features).to(device),
        torch.stack(actions).to(device),
        torch.tensor(task_ids, dtype=torch.long, device=device),
        torch.tensor(progress_bins, dtype=torch.long, device=device),
    )


def weighted_task_gradients(task_gradients, task_counts):
    task_ids = sorted(task_gradients)
    total = sum(int(task_counts[task_id]) for task_id in task_ids)
    if total <= 0:
        raise ValueError("empty replay gradient aggregation")
    return tuple(
        sum(
            task_gradients[task_id][parameter_index]
            * (int(task_counts[task_id]) / total)
            for task_id in task_ids
        )
        for parameter_index in range(len(task_gradients[task_ids[0]]))
    )


def gradient_parity(first, second, eps=1e-12):
    differences = [a - b for a, b in zip(first, second)]
    max_abs = max(float(item.detach().abs().max()) for item in differences)
    difference_norm2 = sum(torch.sum(item.detach().double() ** 2) for item in differences)
    reference_norm2 = sum(torch.sum(item.detach().double() ** 2) for item in second)
    relative_l2 = float(torch.sqrt(difference_norm2) / (torch.sqrt(reference_norm2) + eps))
    return max_abs, relative_l2


def gradient_relation(first, second, eps=1e-12):
    dot = sum(torch.sum(a.detach().double() * b.detach().double()) for a, b in zip(first, second))
    first_norm2 = sum(torch.sum(a.detach().double() ** 2) for a in first)
    second_norm2 = sum(torch.sum(b.detach().double() ** 2) for b in second)
    first_norm = torch.sqrt(first_norm2)
    second_norm = torch.sqrt(second_norm2)
    denominator = first_norm * second_norm
    cosine = float(dot / denominator) if float(denominator) > eps else float("nan")
    return {
        "dot": float(dot),
        "cosine": cosine,
        "first_norm": float(first_norm),
        "second_norm": float(second_norm),
    }


def classify_task_relations(individual_pre_dots, aggregate_dot, individual_post_dots):
    pre = [float(value) for value in individual_pre_dots]
    post = [float(value) for value in individual_post_dots]
    multi_old = len(pre) >= 2
    has_negative = any(value < 0.0 for value in pre)
    has_nonnegative = any(value >= 0.0 for value in pre)
    return {
        "multi_old": multi_old,
        "mixed_old_sign": multi_old and has_negative and has_nonnegative,
        "masked_conflict": multi_old and float(aggregate_dot) >= 0.0 and has_negative,
        "residual_conflict": multi_old and any(value < 0.0 for value in post),
    }


def diagnose_batch(policy, batch_records, current_task, n_bins=5):
    device = next(policy.parameters()).device
    features, actions, task_ids, progress_bins = load_record_batch(
        batch_records, device
    )
    temporal = policy.temporal_encode(features)
    distribution = policy.policy_head(temporal[:, -1, :], train=True)
    loss_vector = policy.policy_head.loss_fn(distribution, actions, reduction="none")
    if not torch.isfinite(loss_vector).all():
        raise FloatingPointError("non-finite diagnostic loss")
    head_params = [
        parameter for parameter in policy.policy_head.parameters()
        if parameter.requires_grad
    ]
    results = []
    max_parity_abs = 0.0
    max_parity_relative_l2 = 0.0
    for bin_id in range(int(n_bins)):
        current_mask = (task_ids == int(current_task)) & (progress_bins == bin_id)
        replay_mask = (task_ids != int(current_task)) & (progress_bins == bin_id)
        if not bool(current_mask.any()) or not bool(replay_mask.any()):
            continue
        present_old_tasks = sorted(
            int(task_id)
            for task_id in torch.unique(task_ids[replay_mask]).detach().cpu().tolist()
        )
        loss_new = loss_vector[current_mask].mean()
        g_new = torch.autograd.grad(loss_new, head_params, retain_graph=True)
        old_gradients = {}
        old_counts = {}
        for old_task in present_old_tasks:
            old_mask = replay_mask & (task_ids == int(old_task))
            old_counts[old_task] = int(old_mask.sum().item())
            old_gradients[old_task] = torch.autograd.grad(
                loss_vector[old_mask].mean(), head_params, retain_graph=True
            )
        weighted_aggregate = weighted_task_gradients(old_gradients, old_counts)
        direct = torch.autograd.grad(
            loss_vector[replay_mask].mean(), head_params, retain_graph=True
        )
        parity_abs, parity_relative_l2 = gradient_parity(weighted_aggregate, direct)
        max_parity_abs = max(max_parity_abs, parity_abs)
        max_parity_relative_l2 = max(max_parity_relative_l2, parity_relative_l2)
        if parity_relative_l2 > 1e-6:
            raise AssertionError(
                f"aggregate gradient parity failed: max_abs={parity_abs} "
                f"relative_l2={parity_relative_l2}"
            )
        # Temporal V1 obtains its aggregate anchor from the direct replay-bin
        # mean. Use that exact float32 gradient for the simulated gate; the
        # independently reconstructed weighted gradient is the parity audit.
        aggregate = direct
        gated_new, gate_metrics = signed_replay_projection(
            g_new, aggregate, gate_strength=1.0
        )
        aggregate_relation = gradient_relation(g_new, aggregate)
        pre_relations = {
            task_id: gradient_relation(g_new, old_gradients[task_id])
            for task_id in present_old_tasks
        }
        post_relations = {
            task_id: gradient_relation(gated_new, old_gradients[task_id])
            for task_id in present_old_tasks
        }
        classification = classify_task_relations(
            [pre_relations[task_id]["dot"] for task_id in present_old_tasks],
            aggregate_relation["dot"],
            [post_relations[task_id]["dot"] for task_id in present_old_tasks],
        )
        results.append(
            {
                "bin": bin_id,
                "n_new_bin": int(current_mask.sum().item()),
                "n_replay_bin_total": int(replay_mask.sum().item()),
                "present_old_tasks": present_old_tasks,
                "old_counts": old_counts,
                "aggregate_relation": aggregate_relation,
                "aggregate_gate_fired": bool(gate_metrics["gate_fired"]),
                "pre_relations": pre_relations,
                "post_relations": post_relations,
                **classification,
            }
        )
        del g_new, old_gradients, weighted_aggregate, aggregate, direct
        del gated_new, gate_metrics
    del features, actions, temporal, distribution, loss_vector
    return results, {
        "max_parity_abs": max_parity_abs,
        "max_parity_relative_l2": max_parity_relative_l2,
    }
