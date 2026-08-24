import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def normalize_actions(actions):
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions)
    while actions.ndim > 2 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[-1] != 7:
        raise ValueError(f"expected actions shaped [T, 7], got {actions.shape}")
    return np.ascontiguousarray(actions, dtype=np.float32)


def action_key(actions):
    return normalize_actions(actions).tobytes()


def assign_progress_bin(progress, n_bins=5):
    progress = float(progress)
    if not 0.0 <= progress <= 1.0:
        raise ValueError(f"progress must be in [0, 1], got {progress}")
    return min(int(progress * n_bins), n_bins - 1)


def build_raw_temporal_index(sequence_dataset):
    index_to_demo = sequence_dataset._index_to_demo_id
    demo_starts = sequence_dataset._demo_id_to_start_indices
    demo_lengths = sequence_dataset._demo_id_to_demo_length
    raw_index = defaultdict(list)
    for global_index in range(len(sequence_dataset)):
        demo_id = index_to_demo[global_index]
        rank = global_index - demo_starts[demo_id]
        count = int(demo_lengths[demo_id])
        progress = rank / max(count - 1, 1)
        raw_index[action_key(sequence_dataset[global_index]["actions"])].append(
            {
                "global_index": global_index,
                "demo_id": demo_id,
                "rank_in_demo": rank,
                "demo_sequence_count": count,
                "progress": progress,
            }
        )
    return dict(raw_index)


def _numeric_path_key(path):
    try:
        return (0, int(path.stem))
    except ValueError:
        return (1, path.name)


def annotate_feature_records(feature_dir, raw_index, task_id, n_bins=5):
    paths = sorted(Path(feature_dir, str(task_id)).glob("*.pt"), key=_numeric_path_key)
    records = []
    matched = ambiguous = unmatched = 0
    for path in paths:
        saved = torch.load(path, map_location="cpu")
        actions = normalize_actions(saved["actions"])
        feature = saved["feature"]
        while feature.ndim > 3 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        if tuple(feature.shape) != (8, 4, 512):
            raise ValueError(f"unexpected feature shape {tuple(feature.shape)} in {path}")
        candidates = raw_index.get(action_key(actions), [])
        if not candidates:
            unmatched += 1
            continue
        if len(candidates) > 1:
            ambiguous += 1
            continue
        metadata = candidates[0]
        records.append(
            {
                "path": str(path),
                "task_id": int(task_id),
                "demo_id": metadata["demo_id"],
                "global_index": metadata["global_index"],
                "rank_in_demo": metadata["rank_in_demo"],
                "demo_sequence_count": metadata["demo_sequence_count"],
                "progress": metadata["progress"],
                "bin_id": assign_progress_bin(metadata["progress"], n_bins),
            }
        )
        matched += 1
    total = len(paths)
    return records, {
        "feature_record_count": total,
        "matched_count": matched,
        "ambiguous_count": ambiguous,
        "unmatched_count": unmatched,
        "coverage": matched / total if total else 0.0,
    }


def select_records_by_bin(records, samples_per_bin, sample_seed, n_bins=5):
    by_bin = defaultdict(list)
    for record in records:
        by_bin[record["bin_id"]].append(record)
    selected = {}
    for bin_id in range(n_bins):
        candidates = sorted(by_bin[bin_id], key=lambda item: item["path"])
        if len(candidates) < samples_per_bin:
            raise ValueError(
                f"bin {bin_id} has {len(candidates)} records, needs {samples_per_bin}"
            )
        rng = random.Random(sample_seed + bin_id)
        selected[bin_id] = rng.sample(candidates, samples_per_bin)
    return selected


def enumerate_natural_pairs(start_task=6, end_task=10):
    return [
        (old_task, new_task)
        for new_task in range(start_task + 1, end_task)
        for old_task in range(start_task, new_task)
    ]


def resolve_stage_checkpoint(run_dir, stage_id, preferred_epoch=95):
    checkpoint = Path(run_dir) / f"mlrifa_{stage_id}_model_ep{preferred_epoch}.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing exact checkpoint: {checkpoint}")
    return checkpoint


def diagnostic_bc_loss(policy, feature, actions):
    temporal = policy.temporal_encode(feature)
    last_feature = temporal[:, -1, :]
    distribution = policy.policy_head(last_feature, train=True)
    return policy.policy_head.loss_fn(distribution, actions, reduction="mean")


def trainable_parameter_scope(policy):
    named = [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad]
    groups = {
        "temporal_transformer": [i for i, (name, _) in enumerate(named) if name.startswith("temporal_transformer.")],
        "policy_head": [i for i, (name, _) in enumerate(named) if name.startswith("policy_head.")],
    }
    covered = set(groups["temporal_transformer"]) | set(groups["policy_head"])
    if covered != set(range(len(named))):
        names = [name for i, (name, _) in enumerate(named) if i not in covered]
        raise ValueError(f"unexpected trainable parameters: {names}")
    return named, groups


def _load_record_batch(records, device):
    features = []
    actions = []
    for record in records:
        saved = torch.load(record["path"], map_location="cpu")
        feature = saved["feature"]
        action = saved["actions"]
        while feature.ndim > 3 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        while action.ndim > 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        features.append(feature)
        actions.append(action)
    return torch.stack(features).to(device), torch.stack(actions).to(device)


def mean_gradient(policy, records, batch_size, device):
    named, group_indices = trainable_parameter_scope(policy)
    parameters = [parameter for _, parameter in named]
    accumulators = [torch.zeros_like(parameter, device="cpu", dtype=torch.float32) for parameter in parameters]
    unused = set()
    total = 0
    for start in range(0, len(records), batch_size):
        batch_records = records[start : start + batch_size]
        feature, actions = _load_record_batch(batch_records, device)
        loss = diagnostic_bc_loss(policy, feature, actions)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite diagnostic loss: {loss}")
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        weight = len(batch_records)
        for index, gradient in enumerate(gradients):
            if gradient is None:
                unused.add(named[index][0])
                continue
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"non-finite gradient: {named[index][0]}")
            accumulators[index].add_(gradient.detach().float().cpu(), alpha=weight)
        total += weight
        del feature, actions, loss, gradients
    if total != len(records):
        raise RuntimeError("gradient sample accounting mismatch")
    for accumulator in accumulators:
        accumulator.div_(total)
    vectors = {}
    for group, indices in group_indices.items():
        vectors[group] = torch.cat([accumulators[index].reshape(-1) for index in indices])
    return vectors, sorted(unused)


def aggregate_bin_gradients(bin_gradients, bin_counts):
    total = sum(bin_counts.values())
    result = {}
    for group in ("temporal_transformer", "policy_head"):
        result[group] = sum(
            bin_gradients[bin_id][group] * bin_counts[bin_id]
            for bin_id in sorted(bin_gradients)
        ) / total
    return result


def gradient_cosine(first, second):
    first = first.reshape(-1).double()
    second = second.reshape(-1).double()
    first_norm = torch.linalg.vector_norm(first)
    second_norm = torch.linalg.vector_norm(second)
    if first_norm.item() == 0.0 or second_norm.item() == 0.0:
        raise ValueError("zero gradient norm")
    cosine = torch.dot(first, second) / (first_norm * second_norm)
    return float(cosine), float(first_norm), float(second_norm)


def grouped_gradient_stats(old_gradients, new_gradients):
    stats = {}
    dot_all = old_norm_sq = new_norm_sq = 0.0
    for group in ("temporal_transformer", "policy_head"):
        old = old_gradients[group].double()
        new = new_gradients[group].double()
        cosine, old_norm, new_norm = gradient_cosine(old, new)
        stats[group] = {"cosine": cosine, "old_norm": old_norm, "new_norm": new_norm}
        dot_all += float(torch.dot(old, new))
        old_norm_sq += old_norm**2
        new_norm_sq += new_norm**2
    old_norm = math.sqrt(old_norm_sq)
    new_norm = math.sqrt(new_norm_sq)
    if old_norm == 0.0 or new_norm == 0.0:
        raise ValueError("zero all-trainable gradient norm")
    stats["all_trainable"] = {
        "cosine": dot_all / (old_norm * new_norm),
        "old_norm": old_norm,
        "new_norm": new_norm,
    }
    return stats


def action_descriptor(records):
    actions = []
    for record in records:
        saved = torch.load(record["path"], map_location="cpu")
        actions.append(normalize_actions(saved["actions"]))
    actions = np.concatenate(actions, axis=0)
    return np.concatenate(
        [
            actions[:, 0:3].mean(axis=0),
            np.abs(actions[:, 0:3]).mean(axis=0),
            actions[:, 3:6].mean(axis=0),
            np.abs(actions[:, 3:6]).mean(axis=0),
            actions[:, 6].mean(keepdims=True),
            np.abs(actions[:, 6]).mean(keepdims=True),
        ]
    ).astype(np.float64)


def action_relation(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    cosine = float(np.dot(first, second) / denominator) if denominator else float("nan")
    return cosine, float(np.linalg.norm(first - second))
