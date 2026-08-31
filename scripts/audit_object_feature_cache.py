#!/usr/bin/env python3
import csv
import json
import os
import random
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "paper_reproduction_v2" / "feature_audit"
SEEDS = (42, 43, 44)
MODALITIES = ("language", "agent_view", "eye_in_hand", "state")


def action_key(payload):
    actions = payload["actions"].detach().cpu().contiguous()
    return (str(actions.dtype), tuple(actions.shape), actions.numpy().tobytes())


def cache_index(root):
    result = {}
    for task in range(6, 10):
        records = {}
        for path in sorted((root / str(task)).glob("*.pt")):
            payload = torch.load(path, map_location="cpu")
            records[action_key(payload)] = path
        result[task] = records
    return result


def compare(left, right, sample_size=64):
    rng = random.Random(20260831)
    candidates = []
    for task in sorted(set(left) & set(right)):
        common = sorted(set(left[task]) & set(right[task]), key=repr)
        candidates.extend((task, key) for key in common)
    rng.shuffle(candidates)
    candidates = candidates[:sample_size]
    exact = 0
    max_abs = 0.0
    squared_diff = 0.0
    squared_left = 0.0
    cosines = {name: [] for name in MODALITIES}
    shapes_equal = True
    dtypes_equal = True
    finite = True
    for task, key in candidates:
        a_raw = torch.load(left[task][key], map_location="cpu")["feature"]
        b_raw = torch.load(right[task][key], map_location="cpu")["feature"]
        dtypes_equal &= a_raw.dtype == b_raw.dtype
        a = a_raw.float()
        b = b_raw.float()
        shapes_equal &= a.shape == b.shape
        finite &= bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        exact += int(torch.equal(a, b))
        max_abs = max(max_abs, float((a - b).abs().max()))
        squared_diff += float(torch.sum((a - b) ** 2))
        squared_left += float(torch.sum(a ** 2))
        for index, name in enumerate(MODALITIES):
            left_mod = a[..., index, :].reshape(-1)
            right_mod = b[..., index, :].reshape(-1)
            cosine = torch.nn.functional.cosine_similarity(left_mod, right_mod, dim=0)
            cosines[name].append(float(cosine))
    result = {
        "aligned_samples": len(candidates),
        "actions_exact": True,
        "exact_feature_matches": exact,
        "shape_equal": shapes_equal,
        "dtype_equal": dtypes_equal,
        "finite": finite,
        "max_abs": max_abs,
        "relative_l2": (squared_diff / max(squared_left, 1e-30)) ** 0.5,
    }
    result.update({f"{name}_cosine": sum(values) / len(values) for name, values in cosines.items()})
    return result


def task_summary(seed, task, records, file_count, sample_size=64):
    paths = sorted(records.values())[:sample_size]
    shapes, dtypes, finite = set(), set(), True
    norms = {name: [] for name in MODALITIES}
    for path in paths:
        feature_raw = torch.load(path, map_location="cpu")["feature"]
        feature = feature_raw.float()
        shapes.add(str(tuple(feature.shape)))
        dtypes.add(str(feature_raw.dtype))
        finite &= bool(torch.isfinite(feature).all())
        for index, name in enumerate(MODALITIES):
            values = feature[..., index, :].reshape(-1, feature.shape[-1])
            norms[name].append(float(torch.linalg.vector_norm(values, dim=-1).mean()))
    row = {"seed": seed, "task": task, "records": file_count,
           "unique_action_records": len(records), "audited_records": len(paths),
           "feature_shapes": ";".join(sorted(shapes)), "dtypes": ";".join(sorted(dtypes)),
           "finite": finite}
    row.update({f"{name}_mean_norm": sum(values) / len(values) for name, values in norms.items()})
    return row


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    roots = {seed: ROOT / "features" / f"object_seed{seed}" for seed in SEEDS}
    indices = {seed: cache_index(path) for seed, path in roots.items()}
    file_counts = {
        seed: {task: len(list((path / str(task)).glob("*.pt"))) for task in range(6, 10)}
        for seed, path in roots.items()
    }
    payload = {
        "source_behavior": "save_feature uses seeded shuffle and enabled stochastic image augmentation",
        "counts": {str(seed): {str(task): file_counts[seed][task] for task in range(6, 10)} for seed in SEEDS},
        "unique_action_records": {str(seed): {str(task): len(indices[seed][task]) for task in range(6, 10)} for seed in SEEDS},
        "cross_seed": {},
    }
    task_rows = [task_summary(seed, task, indices[seed][task], file_counts[seed][task]) for seed in SEEDS for task in range(6, 10)]
    pair_rows = []
    for left, right in ((42, 43), (42, 44), (43, 44)):
        aggregate = compare(indices[left], indices[right])
        payload["cross_seed"][f"{left}_vs_{right}"] = aggregate
        for task in range(6, 10):
            row = {"seed_a": left, "seed_b": right, "task": task}
            row.update(compare({task: indices[left][task]}, {task: indices[right][task]}))
            pair_rows.append(row)

    repeat = os.environ.get("FEATURE_REPEAT_ROOT")
    if repeat:
        repeat_index = cache_index(Path(repeat))
        payload["official_cache_vs_repeat_42"] = compare(indices[42], repeat_index)
        repeat_b = os.environ.get("FEATURE_REPEAT_ROOT_B")
        if repeat_b:
            repeat_b_index = cache_index(Path(repeat_b))
            payload["same_seed_repeat_42"] = compare(repeat_index, repeat_b_index)
        else:
            payload["same_seed_repeat_42"] = {"status": "pending", "reason": "FEATURE_REPEAT_ROOT_B not provided"}
    else:
        payload["same_seed_repeat_42"] = {"status": "pending", "reason": "FEATURE_REPEAT_ROOT not provided"}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT_DIR / "feature_seed_pair_summary.csv", pair_rows)
    write_csv(REPORT_DIR / "feature_task_summary.csv", task_rows)
    with (REPORT_DIR / "feature_cache_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
