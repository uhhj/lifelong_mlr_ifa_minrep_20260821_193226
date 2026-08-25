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
    matrix = [
        [None if task > stage else records[(stage, selected[stage], task)] for task in TASKS]
        for stage in TASKS
    ]
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


def rounded(value):
    return None if value is None else round(value, 6)


def weighted(rows, column, denominator, weight_column):
    if not denominator:
        return None
    return sum(
        float(row[column]) * int(row[weight_column])
        for row in rows if row[column] != ""
    ) / denominator


def aggregate_temporal_gate(run, seed):
    bins = read_csv(run / "run_001" / "temporal_igfa_gate_stats.csv")
    overall = read_csv(run / "run_001" / "temporal_igfa_gate_overall.csv")
    if len(bins) != 2000 or len(overall) != 400:
        raise ValueError(
            f"seed {seed} expected 2000 bin rows and 400 overall rows, "
            f"found {len(bins)} and {len(overall)}"
        )
    stage_rows = []
    bin_rows = []
    for stage in TASKS:
        stage_bins = [row for row in bins if int(row["stage"]) == stage]
        stage_overall = [row for row in overall if int(row["stage"]) == stage]
        total_batches = sum(int(row["total_batches"]) for row in stage_overall)
        anchor_batches = sum(int(row["batches_with_any_same_bin_anchor"]) for row in stage_overall)
        candidates = sum(int(row["candidate_bin_pairs"]) for row in stage_bins)
        conflicts = sum(int(row["conflict_bin_pairs"]) for row in stage_bins)
        stage_rows.append({
            "seed": seed,
            "stage": stage,
            "total_batches": total_batches,
            "batches_with_any_same_bin_anchor": anchor_batches,
            "same_bin_candidate_coverage": anchor_batches / total_batches,
            "candidate_bin_pairs": candidates,
            "conflict_bin_pairs": conflicts,
            "gate_fire_rate": None if not candidates else conflicts / candidates,
            "mean_conflict_cos": weighted(stage_bins, "mean_conflict_cos", conflicts, "conflict_bin_pairs"),
            "mean_projection_cost": weighted(stage_bins, "mean_projection_cost", candidates, "candidate_bin_pairs"),
            "mean_candidate_bins_per_batch": sum(float(row["mean_candidate_bins_per_batch"]) * int(row["total_batches"]) for row in stage_overall) / total_batches,
            "mean_gated_bins_per_batch": sum(float(row["mean_gated_bins_per_batch"]) * int(row["total_batches"]) for row in stage_overall) / total_batches,
            "resolved_current_fraction": sum(float(row["resolved_current_fraction"]) * int(row["total_batches"]) for row in stage_overall) / total_batches,
            "resolved_replay_fraction": None if stage == 6 else sum(float(row["resolved_replay_fraction"]) * int(row["total_batches"]) for row in stage_overall) / total_batches,
            "zero_anchor_count": sum(int(row["zero_anchor_count"]) for row in stage_bins),
        })
        for bin_id in range(5):
            selected = [row for row in stage_bins if int(row["bin"]) == bin_id]
            candidates_bin = sum(int(row["candidate_bin_pairs"]) for row in selected)
            conflicts_bin = sum(int(row["conflict_bin_pairs"]) for row in selected)
            bin_rows.append({
                "seed": seed,
                "stage": stage,
                "bin": bin_id,
                "candidate_bin_pairs": candidates_bin,
                "conflict_bin_pairs": conflicts_bin,
                "gate_fire_rate": None if not candidates_bin else conflicts_bin / candidates_bin,
                "mean_cos": weighted(selected, "mean_cos", candidates_bin, "candidate_bin_pairs"),
                "mean_conflict_cos": weighted(selected, "mean_conflict_cos", conflicts_bin, "conflict_bin_pairs"),
                "mean_projection_cost": weighted(selected, "mean_projection_cost", candidates_bin, "candidate_bin_pairs"),
                "mean_new_bin_count": weighted(selected, "mean_new_bin_count", candidates_bin, "candidate_bin_pairs"),
                "mean_replay_bin_count": weighted(selected, "mean_replay_bin_count", candidates_bin, "candidate_bin_pairs"),
                "mean_new_head_grad_norm": weighted(selected, "mean_new_head_grad_norm", candidates_bin, "candidate_bin_pairs"),
                "mean_replay_head_grad_norm": weighted(selected, "mean_replay_head_grad_norm", candidates_bin, "candidate_bin_pairs"),
                "zero_anchor_count": sum(int(row["zero_anchor_count"]) for row in selected),
            })
    return stage_rows, bin_rows


def summarize(values):
    return {
        "mean": rounded(statistics.mean(values)),
        "sample_std": rounded(statistics.stdev(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--baseline-json", type=Path, default=Path("reports/baseline_reconciliation/object_3seed_metrics.json"))
    parser.add_argument("--static-json", type=Path, default=Path("reports/static_robot_igfa/static_robot_igfa_3seed_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/temporal_robot_igfa_v1"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    static = json.loads(args.static_json.read_text(encoding="utf-8"))

    per_seed = {}
    metric_rows = []
    per_stage = []
    per_task_fwt = []
    per_task_forgetting = []
    gate_stages = []
    gate_bins = []
    for seed in args.seeds:
        run = args.runs_root / f"object_seed{seed}_mlr_temporal_igfa"
        selected, matrix = parse_evaluation(run / f"eval_seed{seed}.log")
        score = compute_metrics(matrix)
        mlr = baseline["per_seed"]["mlr"][str(seed)]
        static_seed = static["per_seed"][str(seed)]
        delta_mlr = {metric: score[metric] - mlr[metric] for metric in METRICS}
        delta_static = {metric: score[metric] - static_seed[metric] for metric in METRICS}
        per_seed[str(seed)] = {
            **{metric: rounded(score[metric]) for metric in METRICS},
            "delta_vs_mlr": {metric: rounded(delta_mlr[metric]) for metric in METRICS},
            "delta_vs_static": {metric: rounded(delta_static[metric]) for metric in METRICS},
            "selected_epochs": {str(stage): epoch for stage, epoch in selected.items()},
            "r": [[rounded(value) for value in row] for row in matrix],
        }
        metric_rows.append({
            "seed": seed,
            **{metric: rounded(score[metric]) for metric in METRICS},
            **{f"delta_{metric}_vs_static": rounded(delta_static[metric]) for metric in METRICS},
            **{f"delta_{metric}_vs_mlr": rounded(delta_mlr[metric]) for metric in METRICS},
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
                if matrix[stage_index][task_index] is not None:
                    per_stage.append({
                        "seed": seed,
                        "stage": stage,
                        "task": task,
                        "selected_epoch": selected[stage],
                        "success": rounded(matrix[stage_index][task_index]),
                    })
        stage_gate, bin_gate = aggregate_temporal_gate(run, seed)
        gate_stages.extend(stage_gate)
        gate_bins.extend(bin_gate)

    summary = {
        metric: summarize([per_seed[str(seed)][metric] for seed in args.seeds])
        for metric in METRICS
    }
    deltas = {}
    for reference in ("static", "mlr"):
        deltas[reference] = {}
        for metric in METRICS:
            values = [per_seed[str(seed)][f"delta_vs_{reference}"][metric] for seed in args.seeds]
            deltas[reference][metric] = {
                **summarize(values),
                "positive_seed_count": sum(value > 0 for value in values),
            }

    total_batches = sum(row["total_batches"] for row in gate_stages)
    anchor_batches = sum(row["batches_with_any_same_bin_anchor"] for row in gate_stages)
    candidate_pairs = sum(row["candidate_bin_pairs"] for row in gate_stages)
    conflict_pairs = sum(row["conflict_bin_pairs"] for row in gate_stages)
    mechanism = {
        "total_batches": total_batches,
        "batches_with_any_same_bin_anchor": anchor_batches,
        "same_bin_candidate_coverage": rounded(anchor_batches / total_batches),
        "candidate_bin_pairs": candidate_pairs,
        "conflict_bin_pairs": conflict_pairs,
        "gate_fire_rate": rounded(conflict_pairs / candidate_pairs),
        "mean_conflict_cos": rounded(sum((row["mean_conflict_cos"] or 0) * row["conflict_bin_pairs"] for row in gate_stages) / conflict_pairs),
        "mean_projection_cost": rounded(sum((row["mean_projection_cost"] or 0) * row["candidate_bin_pairs"] for row in gate_stages) / candidate_pairs),
        "zero_anchor_count": sum(row["zero_anchor_count"] for row in gate_stages),
    }
    delta_static_auc = deltas["static"]["auc"]["mean"]
    delta_static_fwt = deltas["static"]["fwt"]["mean"]
    delta_static_nbt = deltas["static"]["nbt"]["mean"]
    if (
        delta_static_auc >= 1.0
        and deltas["static"]["auc"]["positive_seed_count"] >= 2
        and delta_static_fwt >= -1.0
        and delta_static_nbt <= 1.0
    ):
        verdict = "TEMPORAL_ROBOT_IGFA_SUPPORTED"
    elif delta_static_auc <= -1.0 or (
        delta_static_auc <= 0.0 and delta_static_fwt < -2.0
    ):
        verdict = "TEMPORAL_ROBOT_IGFA_NOT_SUPPORTED"
    else:
        verdict = "TEMPORAL_ROBOT_IGFA_INCONCLUSIVE"
    practical = (
        "SUPPORTED"
        if deltas["mlr"]["auc"]["mean"] >= 1.0
        and deltas["mlr"]["auc"]["positive_seed_count"] >= 2
        and deltas["mlr"]["fwt"]["mean"] >= -1.0
        else "NOT ESTABLISHED"
    )
    payload = {
        "benchmark": "LIBERO_OBJECT",
        "method": "mlr_temporal_robot_igfa_v1",
        "seeds": args.seeds,
        "standard_deviation": "sample",
        "verdict": verdict,
        "practical_benefit_vs_mlr": practical,
        "per_seed": per_seed,
        "three_seed_summary": summary,
        "delta_vs_static": deltas["static"],
        "delta_vs_mlr": deltas["mlr"],
        "temporal_mechanism_overall": mechanism,
        "frozen_references": {
            "mlr_and_mlr_ifa": baseline["three_seed_summary"],
            "static_robot_igfa": static["three_seed_summary"],
        },
    }
    (args.output_dir / "temporal_robot_igfa_3seed_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "per_seed_metrics.csv", metric_rows[0].keys(), metric_rows)
    write_csv(args.output_dir / "per_stage_success.csv", per_stage[0].keys(), per_stage)
    write_csv(args.output_dir / "per_task_fwt.csv", per_task_fwt[0].keys(), per_task_fwt)
    write_csv(args.output_dir / "per_task_forgetting.csv", per_task_forgetting[0].keys(), per_task_forgetting)
    rounded_stage_gate = [
        {key: rounded(value) if isinstance(value, float) else value for key, value in row.items()}
        for row in gate_stages
    ]
    rounded_bin_gate = [
        {key: rounded(value) if isinstance(value, float) else value for key, value in row.items()}
        for row in gate_bins
    ]
    write_csv(args.output_dir / "temporal_gate_summary.csv", rounded_stage_gate[0].keys(), rounded_stage_gate)
    write_csv(args.output_dir / "temporal_gate_by_bin.csv", rounded_bin_gate[0].keys(), rounded_bin_gate)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
