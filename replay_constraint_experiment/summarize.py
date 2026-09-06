#!/usr/bin/env python3
"""Strict fixed-ep99 metrics, epoch diagnostics and paired-seed differences."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

LINE = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.eE+-]+)")


def read_fixed_matrix(path):
    records = {}
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        m = LINE.search(text)
        if not m:
            continue
        task, epoch, stage = map(int, m.group(1, 2, 3))
        if not (6 <= task <= stage <= 9):
            raise ValueError("Unexpected evaluation task/stage: " + text)
        if epoch != 99:
            raise ValueError("Only fixed ep99 results belong to this experiment.")
        key = (stage, task)
        if key in records:
            raise ValueError("Duplicate evaluation record: %r" % (key,))
        value = float(m.group(4))
        if not 0 <= value <= 1:
            raise ValueError("Invalid success rate.")
        if abs(value * 20 - round(value * 20)) > 1e-5:
            raise ValueError("Success rate is inconsistent with 20 binary trials.")
        records[key] = value * 100
    expected = {(s, t) for s in range(6, 10) for t in range(6, s + 1)}
    if set(records) != expected:
        raise ValueError("Incomplete fixed-ep99 matrix; missing %r" % sorted(expected - set(records)))
    return [[records.get((s, t)) for t in range(6, 10)] for s in range(6, 10)]


def score_matrix(r):
    # Preserve this project's published FWT/NBT/AUC definitions. FWT here is the
    # stage diagonal, not an independently measured zero-shot forward transfer.
    m = len(r)
    fwt = statistics.mean(r[i][i] for i in range(m))
    per_task_nbt = [statistics.mean(r[i][i] - r[q][i] for q in range(i + 1, m)) for i in range(m - 1)]
    auc = statistics.mean(statistics.mean(r[q][i] for q in range(i, m)) for i in range(m))
    max_forgetting = [max(r[q][i] for q in range(i, m)) - r[-1][i] for i in range(m - 1)]
    return {"fwt": fwt, "nbt": statistics.mean(per_task_nbt), "auc": auc,
            "final_mean_success": statistics.mean(r[-1]),
            "worst_old_final_success": min(r[-1][:-1]),
            "worst_peak_to_final_forgetting": max(max_forgetting), "r": r}


def diagnostic_summary(path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for r in rows:
        key = (int(r["stage"]), int(r["epoch"]))
        if key in seen:
            raise ValueError("Duplicate diagnostic epoch: %r" % (key,))
        seen.add(key)
    def total(name):
        return sum(float(r[name]) for r in rows)
    batches = total("batches")
    valid = total("valid_reference_batches")
    projected = total("projected_batches")
    return {"epoch_rows": len(rows), "expected_epoch_rows": 400,
            "complete_epoch_grid": seen == {(s, e) for s in range(6, 10) for e in range(100)},
            "batches": batches, "valid_reference_batches": valid,
            "projected_batches": projected,
            "projection_rate_all": projected / batches if batches else None,
            "projection_rate_valid_reference": projected / valid if valid else None,
            "mean_correction_ratio": sum(float(r["mean_correction_ratio"]) * int(r["batches"]) for r in rows) / batches if batches else None,
            "observe_seconds": total("observe_seconds"),
            "peak_allocated_bytes": max((int(r["stage_peak_allocated_bytes"]) for r in rows), default=0)}


def aggregate(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True, help=".../aggregate_v1/formal")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", action="append", default=[], metavar="ARM:SEED:LOG",
                   help="Existing matched fixed-ep99 baseline, e.g. mlr:42:/path/primary_ep99.log")
    args = p.parse_args()
    result = {"protocol": "LIBERO_OBJECT tasks6-9, fixed ep99, 20 trials; original metric definitions",
              "runs": {}, "aggregate": {}, "paired_constraint_minus_mlr": {}}
    for arm_dir in sorted(args.run_root.glob("*")):
        if not arm_dir.is_dir():
            continue
        for seed_dir in sorted(arm_dir.glob("seed*")):
            log = seed_dir / "primary_ep99.log"
            if not log.exists():
                continue
            seed = int(seed_dir.name[4:])
            metrics = score_matrix(read_fixed_matrix(log))
            metrics["diagnostics"] = diagnostic_summary(seed_dir / "run_001/replay_constraint_epochs.csv")
            metrics["source_log"] = str(log)
            execution = seed_dir / "train_execution.json"
            if execution.is_file():
                metrics["training_wall_seconds"] = json.loads(execution.read_text(encoding="utf-8"))["wall_seconds"]
            result["runs"].setdefault(arm_dir.name, {})[str(seed)] = metrics
    for ref in args.reference:
        arm, seed, path = ref.split(":", 2)
        target = result["runs"].setdefault(arm, {})
        if seed in target:
            raise ValueError("Duplicate arm/seed: " + ref)
        target[seed] = dict(score_matrix(read_fixed_matrix(path)), source_log=path,
                            provenance="existing baseline; verify matching protocol before comparison")
    names = ("fwt", "nbt", "auc", "final_mean_success", "worst_old_final_success", "worst_peak_to_final_forgetting")
    for arm, seeds in result["runs"].items():
        result["aggregate"][arm] = {n: aggregate([s[n] for s in seeds.values()]) for n in names}
    control = result["runs"].get("mlr", {})
    candidate = result["runs"].get("constraint", {})
    common = sorted(set(control) & set(candidate))
    if common:
        result["paired_constraint_minus_mlr"] = {
            "seeds": common,
            "metrics": {n: dict(aggregate([candidate[s][n] - control[s][n] for s in common]),
                                per_seed={s: candidate[s][n] - control[s][n] for s in common}) for n in names}}
    if not result["runs"]:
        raise ValueError("No completed evaluation matrices found.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
