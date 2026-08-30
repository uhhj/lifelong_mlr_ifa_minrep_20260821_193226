#!/usr/bin/env python3
import argparse
import csv
import re
from pathlib import Path


RUN_SUFFIX = {
    "temporal_v1": "mlr_temporal_igfa",
    "zero_guard": "mlr_zero_directional_guard",
}
LINE = re.compile(
    r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)"
)
FIELDS = [
    "method",
    "seed",
    "stage",
    "epoch",
    "task",
    "is_old_task",
    "is_acquisition_stage",
    "success_rate",
    "success_pct",
    "trials",
]


def parse_log(path):
    records = {}
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        match = LINE.search(text)
        if not match:
            continue
        task, epoch, stage = map(int, match.group(1, 2, 3))
        key = (stage, epoch, task)
        value = float(match.group(4))
        if key in records and records[key] != value:
            raise ValueError(f"inconsistent duplicate rollout record {key} in {path}")
        records[key] = value
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/bc_rollout_retention_alignment/rollout_checkpoint_grid.csv"
        ),
    )
    parser.add_argument(
        "--methods", nargs="+", choices=sorted(RUN_SUFFIX), default=list(RUN_SUFFIX)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", nargs="+", type=int, default=[70, 75, 80, 85, 90, 95])
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--old-only", action="store_true")
    parser.add_argument(
        "--old-log-template",
        default="runs/object_seed{seed}_{suffix}/alignment_old_grid_seed{seed}.log",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    rows = []
    seen = set()
    for method in args.methods:
        suffix = RUN_SUFFIX[method]
        for seed in args.seeds:
            old_path = root / args.old_log_template.format(
                seed=seed, suffix=suffix, method=method
            )
            old_records = parse_log(old_path)
            formal_records = {}
            if not args.old_only:
                formal_records = parse_log(
                    root / "runs" / f"object_seed{seed}_{suffix}" / f"eval_seed{seed}.log"
                )
            selected = []
            for (stage, epoch, task), value in old_records.items():
                if task < stage and epoch in args.epochs:
                    selected.append((stage, epoch, task, value))
            if not args.old_only:
                for (stage, epoch, task), value in formal_records.items():
                    if task == stage and task in (6, 7, 8) and epoch in args.epochs:
                        selected.append((stage, epoch, task, value))
            for stage, epoch, task, value in sorted(selected):
                key = (method, seed, stage, epoch, task)
                if key in seen:
                    raise ValueError(f"duplicate merged rollout key: {key}")
                if not 0.0 <= value <= 1.0:
                    raise ValueError(f"success outside [0,1] for {key}: {value}")
                seen.add(key)
                rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "stage": stage,
                        "epoch": epoch,
                        "task": task,
                        "is_old_task": int(task < stage),
                        "is_acquisition_stage": int(task == stage),
                        "success_rate": value,
                        "success_pct": value * 100.0,
                        "trials": args.trials,
                    }
                )
    expected = len(args.methods) * len(args.seeds) * (
        len(args.epochs) if args.old_only else 54
    )
    if len(rows) != expected:
        raise ValueError(f"expected {expected} rollout rows, found {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[rollout-grid] rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
