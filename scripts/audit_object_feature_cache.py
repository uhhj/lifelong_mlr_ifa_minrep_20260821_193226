#!/usr/bin/env python3
import json
import os
import random
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "paper_reproduction_v2" / "feature_audit"
SEEDS = (42, 43, 44)


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
    for task in range(6, 10):
        common = sorted(set(left[task]) & set(right[task]), key=repr)
        candidates.extend((task, key) for key in common)
    rng.shuffle(candidates)
    candidates = candidates[:sample_size]
    exact = 0
    max_abs = 0.0
    for task, key in candidates:
        a = torch.load(left[task][key], map_location="cpu")["feature"]
        b = torch.load(right[task][key], map_location="cpu")["feature"]
        exact += int(torch.equal(a, b))
        max_abs = max(max_abs, float((a - b).abs().max()))
    return {"aligned_samples": len(candidates), "exact_feature_matches": exact, "max_abs": max_abs}


def main():
    roots = {seed: ROOT / "features" / f"object_seed{seed}" for seed in SEEDS}
    indices = {seed: cache_index(path) for seed, path in roots.items()}
    payload = {
        "source_behavior": "save_feature uses seeded shuffle and enabled stochastic image augmentation",
        "counts": {str(seed): {str(task): len(indices[seed][task]) for task in range(6, 10)} for seed in SEEDS},
        "cross_seed": {},
    }
    for left, right in ((42, 43), (42, 44), (43, 44)):
        payload["cross_seed"][f"{left}_vs_{right}"] = compare(indices[left], indices[right])

    repeat = os.environ.get("FEATURE_REPEAT_ROOT")
    if repeat:
        repeat_index = cache_index(Path(repeat))
        payload["same_seed_repeat_42"] = compare(indices[42], repeat_index)
    else:
        payload["same_seed_repeat_42"] = {"status": "pending", "reason": "FEATURE_REPEAT_ROOT not provided"}

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with (REPORT_DIR / "feature_cache_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
