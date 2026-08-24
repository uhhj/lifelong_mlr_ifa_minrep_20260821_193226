#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from pathlib import Path


LINE = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
TASKS = tuple(range(6, 10))
EPOCHS = (70, 75, 80, 85, 90, 95)
METRICS = ("fwt", "nbt", "auc")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_evaluation(path):
    records = {}
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        match = LINE.search(text)
        if match:
            task, epoch, stage = map(int, match.group(1, 2, 3))
            records[(stage, epoch, task)] = float(match.group(4)) * 100.0
    if len(records) != 30:
        raise ValueError(f"expected 30 success records in {path}, found {len(records)}")
    selected = {
        stage: max(EPOCHS, key=lambda epoch: (records[(stage, epoch, stage)], epoch))
        for stage in TASKS
    }
    matrix = []
    for stage in TASKS:
        matrix.append([
            None if task > stage else records[(stage, selected[stage], task)]
            for task in TASKS
        ])
    return selected, matrix


def compute_metrics(matrix):
    diagonal = [matrix[index][index] for index in range(4)]
    forgetting = []
    auc_tasks = []
    for task_index in range(4):
        trajectory = [matrix[stage][task_index] for stage in range(task_index, 4)]
        auc_tasks.append(statistics.mean(trajectory))
        if task_index < 3:
            forgetting.append(diagonal[task_index] - statistics.mean(trajectory[1:]))
    return {
        "fwt": statistics.mean(diagonal),
        "nbt": statistics.mean(forgetting),
        "auc": statistics.mean(auc_tasks),
    }


def aggregate_gate_stats(path, seed):
    rows = read_csv(path)
    output = []
    for stage in TASKS:
        selected = [row for row in rows if int(row["stage"]) == stage]
        if len(selected) != 100:
            raise ValueError(f"seed {seed} stage {stage} has {len(selected)} gate-stat epochs")
        total = sum(int(row["total_batches"]) for row in selected)
        paired = sum(int(row["paired_batches"]) for row in selected)
        conflicts = sum(int(row["conflict_batches"]) for row in selected)

        def weighted(column, denominator, weight_column):
            if not denominator:
                return None
            return sum(
                float(row[column]) * int(row[weight_column])
                for row in selected if row[column] != ""
            ) / denominator

        output.append({
            "seed": seed,
            "stage": stage,
            "total_batches": total,
            "paired_batches": paired,
            "paired_batch_fraction": paired / total,
            "conflict_batches": conflicts,
            "gate_fire_rate": None if not paired else conflicts / paired,
            "mean_cos": weighted("mean_cos", paired, "paired_batches"),
            "mean_conflict_cos": weighted("mean_conflict_cos", conflicts, "conflict_batches"),
            "mean_new_head_grad_norm": weighted("mean_new_head_grad_norm", paired, "paired_batches"),
            "mean_replay_head_grad_norm": weighted("mean_replay_head_grad_norm", paired, "paired_batches"),
            "mean_projection_cost": weighted("mean_projection_cost", paired, "paired_batches"),
            "mean_current_fraction": weighted("mean_current_fraction", total, "total_batches"),
            "mean_replay_fraction": weighted("mean_replay_fraction", total, "total_batches"),
            "zero_anchor_count": sum(int(row["zero_anchor_count"]) for row in selected),
        })
    return output


def rounded(value):
    return None if value is None else round(value, 6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--baseline-json", type=Path,
        default=Path("reports/baseline_reconciliation/object_3seed_metrics.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/static_robot_igfa")
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))

    per_seed = {}
    per_seed_rows = []
    per_task_fwt = []
    per_task_forgetting = []
    per_stage_success = []
    gate_rows = []
    for seed in args.seeds:
        run = args.runs_root / f"object_seed{seed}_mlr_static_igfa"
        selected, matrix = parse_evaluation(run / f"eval_seed{seed}.log")
        score = compute_metrics(matrix)
        baseline_score = baseline["per_seed"]["mlr"][str(seed)]
        deltas = {metric: score[metric] - baseline_score[metric] for metric in METRICS}
        per_seed[str(seed)] = {
            **{metric: rounded(score[metric]) for metric in METRICS},
            "delta_vs_mlr": {metric: rounded(deltas[metric]) for metric in METRICS},
            "selected_epochs": {str(stage): epoch for stage, epoch in selected.items()},
            "r": [[rounded(value) for value in row] for row in matrix],
        }
        per_seed_rows.append({
            "seed": seed,
            **{metric: rounded(score[metric]) for metric in METRICS},
            **{f"delta_{metric}_vs_mlr": rounded(deltas[metric]) for metric in METRICS},
        })
        for task_index, task in enumerate(TASKS):
            diagonal = matrix[task_index][task_index]
            per_task_fwt.append({"seed": seed, "task": task, "fwt": rounded(diagonal)})
            later = [matrix[stage][task_index] for stage in range(task_index + 1, 4)]
            per_task_forgetting.append({
                "seed": seed,
                "task": task,
                "diagonal_success": rounded(diagonal),
                "mean_later_success": None if not later else rounded(statistics.mean(later)),
                "forgetting": None if not later else rounded(diagonal - statistics.mean(later)),
            })
        for stage_index, stage in enumerate(TASKS):
            for task_index, task in enumerate(TASKS):
                value = matrix[stage_index][task_index]
                if value is not None:
                    per_stage_success.append({
                        "seed": seed, "stage": stage, "task": task,
                        "selected_epoch": selected[stage], "success": rounded(value),
                    })
        gate_rows.extend(aggregate_gate_stats(
            run / "run_001" / "static_igfa_gate_stats.csv", seed
        ))

    summary = {}
    delta_summary = {}
    for metric in METRICS:
        values = [per_seed[str(seed)][metric] for seed in args.seeds]
        deltas = [per_seed[str(seed)]["delta_vs_mlr"][metric] for seed in args.seeds]
        summary[metric] = {
            "mean": rounded(statistics.mean(values)),
            "sample_std": rounded(statistics.stdev(values)),
        }
        delta_summary[metric] = {
            "mean": rounded(statistics.mean(deltas)),
            "sample_std": rounded(statistics.stdev(deltas)),
            "positive_seed_count": sum(value > 0 for value in deltas),
        }

    paired_total = sum(row["paired_batches"] for row in gate_rows)
    batch_total = sum(row["total_batches"] for row in gate_rows)
    conflict_total = sum(row["conflict_batches"] for row in gate_rows)
    mechanism = {
        "total_batches": batch_total,
        "paired_batches": paired_total,
        "paired_batch_fraction": rounded(paired_total / batch_total),
        "conflict_batches": conflict_total,
        "gate_fire_rate": rounded(conflict_total / paired_total),
        "mean_conflict_cos": rounded(sum(
            row["mean_conflict_cos"] * row["conflict_batches"]
            for row in gate_rows if row["mean_conflict_cos"] is not None
        ) / conflict_total),
        "mean_projection_cost": rounded(sum(
            row["mean_projection_cost"] * row["paired_batches"]
            for row in gate_rows if row["mean_projection_cost"] is not None
        ) / paired_total),
        "zero_anchor_count": sum(row["zero_anchor_count"] for row in gate_rows),
    }
    payload = {
        "benchmark": "LIBERO_OBJECT",
        "method": "mlr_static_robot_igfa",
        "seeds": args.seeds,
        "standard_deviation": "sample",
        "per_seed": per_seed,
        "three_seed_summary": summary,
        "delta_vs_frozen_mlr": delta_summary,
        "gate_mechanism_overall": mechanism,
        "frozen_baseline_summary": baseline["three_seed_summary"],
    }
    (args.output_dir / "static_robot_igfa_3seed_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "per_seed_metrics.csv", per_seed_rows[0].keys(), per_seed_rows)
    write_csv(args.output_dir / "per_task_fwt.csv", per_task_fwt[0].keys(), per_task_fwt)
    write_csv(
        args.output_dir / "per_task_forgetting.csv",
        per_task_forgetting[0].keys(), per_task_forgetting,
    )
    write_csv(
        args.output_dir / "per_stage_success.csv",
        per_stage_success[0].keys(), per_stage_success,
    )
    rounded_gate_rows = [
        {key: rounded(value) if isinstance(value, float) else value for key, value in row.items()}
        for row in gate_rows
    ]
    write_csv(
        args.output_dir / "gate_mechanism_summary.csv",
        rounded_gate_rows[0].keys(), rounded_gate_rows,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
