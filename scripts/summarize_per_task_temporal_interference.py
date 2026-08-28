#!/usr/bin/env python3
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


STAGE_FIELDS = [
    "seed",
    "new_task",
    "checkpoint_stage",
    "bin",
    "candidate_batches",
    "multi_old_candidate_batches",
    "aggregate_safe_multi_old_batches",
    "aggregate_gate_fire_rate",
    "mixed_old_sign_rate",
    "masked_conflict_rate",
    "residual_conflict_after_aggregate_gate_rate",
    "mean_aggregate_cos",
    "mean_worst_old_cos_pre",
    "mean_worst_old_cos_post",
]
OLD_TASK_FIELDS = [
    "seed",
    "new_task",
    "checkpoint_stage",
    "bin",
    "old_task",
    "individual_candidate_count",
    "pre_conflict_rate",
    "post_conflict_rate",
    "mean_pre_cos",
    "mean_post_cos",
]
CASE_FIELDS = [
    "seed",
    "new_task",
    "analysis_phase",
    "checkpoint_stage",
    "checkpoint_epoch",
    "bin",
    "individual_candidate_count",
    "pre_conflict_rate",
    "post_conflict_rate",
    "mean_pre_cos",
    "mean_post_cos",
    "masked_conflict_participation_rate",
]


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def display(value):
    if value is None:
        return ""
    if isinstance(value, (int, float)) and not math.isfinite(value):
        return ""
    return value


def write_rows(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: display(row.get(field)) for field in fields})


def load_raw(raw_dir):
    rows = []
    for path in sorted(raw_dir.glob("*.csv")):
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"no raw diagnosis CSVs in {raw_dir}")
    return rows


def candidate_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["analysis_phase"],
            int(row["seed"]),
            int(row["checkpoint_stage"]),
            int(row["checkpoint_epoch"]),
            int(row["new_task"]),
            int(row["batch_index"]),
            int(row["bin"]),
        )
        groups[key].append(row)
    return groups


def summarize_stage_bins(primary_rows):
    grouped = candidate_groups(primary_rows)
    stage_bins = defaultdict(list)
    for key, rows in grouped.items():
        _, seed, checkpoint_stage, _, new_task, _, bin_id = key
        stage_bins[(seed, new_task, checkpoint_stage, bin_id)].append(rows)
    output = []
    for key in sorted(stage_bins):
        seed, new_task, checkpoint_stage, bin_id = key
        candidates = stage_bins[key]
        multi = [rows for rows in candidates if int(rows[0]["multi_old_candidate"])]
        safe_multi = [
            rows for rows in multi if float(rows[0]["dot_aggregate_pre"]) >= 0.0
        ]
        output.append(
            {
                "seed": seed,
                "new_task": new_task,
                "checkpoint_stage": checkpoint_stage,
                "bin": bin_id,
                "candidate_batches": len(candidates),
                "multi_old_candidate_batches": len(multi),
                "aggregate_safe_multi_old_batches": len(safe_multi),
                "aggregate_gate_fire_rate": mean(
                    int(rows[0]["aggregate_gate_fired"]) for rows in candidates
                ),
                "mixed_old_sign_rate": ratio(
                    sum(int(rows[0]["mixed_old_sign"]) for rows in multi), len(multi)
                ),
                "masked_conflict_rate": ratio(
                    sum(int(rows[0]["masked_conflict"]) for rows in safe_multi),
                    len(safe_multi),
                ),
                "residual_conflict_after_aggregate_gate_rate": ratio(
                    sum(int(rows[0]["residual_conflict"]) for rows in multi),
                    len(multi),
                ),
                "mean_aggregate_cos": mean(
                    float(rows[0]["cos_aggregate_pre"]) for rows in candidates
                ),
                "mean_worst_old_cos_pre": mean(
                    min(float(row["cos_individual_pre"]) for row in rows)
                    for rows in candidates
                ),
                "mean_worst_old_cos_post": mean(
                    min(float(row["cos_individual_post"]) for row in rows)
                    for rows in candidates
                ),
            }
        )
    return output


def summarize_old_tasks(primary_rows):
    groups = defaultdict(list)
    for row in primary_rows:
        key = (
            int(row["seed"]),
            int(row["new_task"]),
            int(row["checkpoint_stage"]),
            int(row["bin"]),
            int(row["old_task"]),
        )
        groups[key].append(row)
    output = []
    for key in sorted(groups):
        seed, new_task, checkpoint_stage, bin_id, old_task = key
        rows = groups[key]
        output.append(
            {
                "seed": seed,
                "new_task": new_task,
                "checkpoint_stage": checkpoint_stage,
                "bin": bin_id,
                "old_task": old_task,
                "individual_candidate_count": len(rows),
                "pre_conflict_rate": mean(
                    int(row["individual_conflict_pre"]) for row in rows
                ),
                "post_conflict_rate": mean(
                    int(row["individual_conflict_post"]) for row in rows
                ),
                "mean_pre_cos": mean(float(row["cos_individual_pre"]) for row in rows),
                "mean_post_cos": mean(float(row["cos_individual_post"]) for row in rows),
            }
        )
    return output


def summarize_seed44_case(rows):
    groups = defaultdict(list)
    for row in rows:
        if int(row["seed"]) != 44 or int(row["old_task"]) != 7:
            continue
        if int(row["new_task"]) not in (8, 9):
            continue
        key = (
            int(row["new_task"]),
            row["analysis_phase"],
            int(row["checkpoint_stage"]),
            int(row["checkpoint_epoch"]),
            int(row["bin"]),
        )
        groups[key].append(row)
    output = []
    for key in sorted(groups):
        new_task, phase, checkpoint_stage, checkpoint_epoch, bin_id = key
        items = groups[key]
        output.append(
            {
                "seed": 44,
                "new_task": new_task,
                "analysis_phase": phase,
                "checkpoint_stage": checkpoint_stage,
                "checkpoint_epoch": checkpoint_epoch,
                "bin": bin_id,
                "individual_candidate_count": len(items),
                "pre_conflict_rate": mean(
                    int(row["individual_conflict_pre"]) for row in items
                ),
                "post_conflict_rate": mean(
                    int(row["individual_conflict_post"]) for row in items
                ),
                "mean_pre_cos": mean(float(row["cos_individual_pre"]) for row in items),
                "mean_post_cos": mean(float(row["cos_individual_post"]) for row in items),
                "masked_conflict_participation_rate": mean(
                    int(row["masked_conflict"])
                    and int(row["individual_conflict_pre"])
                    for row in items
                ),
            }
        )
    return output


def overall_metrics(primary_rows):
    candidates = []
    for key, rows in candidate_groups(primary_rows).items():
        if int(rows[0]["new_task"]) not in (8, 9):
            continue
        if not int(rows[0]["multi_old_candidate"]):
            continue
        candidates.append(rows)
    safe = [rows for rows in candidates if float(rows[0]["dot_aggregate_pre"]) >= 0.0]
    mixed_rate = ratio(
        sum(int(rows[0]["mixed_old_sign"]) for rows in candidates), len(candidates)
    )
    masked_rate = ratio(
        sum(int(rows[0]["masked_conflict"]) for rows in safe), len(safe)
    )
    residual_rate = ratio(
        sum(int(rows[0]["residual_conflict"]) for rows in candidates),
        len(candidates),
    )
    fire_rate = ratio(
        sum(int(rows[0]["aggregate_gate_fired"]) for rows in candidates),
        len(candidates),
    )
    per_seed = {}
    for seed in (42, 43, 44):
        seed_candidates = [rows for rows in candidates if int(rows[0]["seed"]) == seed]
        per_seed[str(seed)] = {
            "multi_old_candidate_count": len(seed_candidates),
            "residual_conflict_rate": ratio(
                sum(int(rows[0]["residual_conflict"]) for rows in seed_candidates),
                len(seed_candidates),
            ),
        }
    replicated = sum(
        value["residual_conflict_rate"] is not None
        and value["residual_conflict_rate"] >= 0.15
        for value in per_seed.values()
    )
    if mixed_rate >= 0.20 and residual_rate >= 0.20 and replicated >= 2:
        verdict = "WITHIN_BIN_TASK_CANCELLATION_SUPPORTED"
    elif mixed_rate < 0.05 and residual_rate < 0.05 and replicated < 2:
        verdict = "WITHIN_BIN_TASK_CANCELLATION_NOT_SUPPORTED"
    else:
        verdict = "WITHIN_BIN_TASK_CANCELLATION_INCONCLUSIVE"
    return {
        "verdict": verdict,
        "scope": "primary entry-proxy stage8/9 multi-old same-bin candidates",
        "multi_old_candidate_count": len(candidates),
        "aggregate_safe_multi_old_candidate_count": len(safe),
        "mixed_old_sign_rate": mixed_rate,
        "masked_conflict_rate": masked_rate,
        "residual_conflict_after_aggregate_gate_rate": residual_rate,
        "aggregate_gate_fire_rate": fire_rate,
        "residual_replication_seed_count_at_least_15pct": replicated,
        "per_seed": per_seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("reports/per_task_temporal_interference_diagnosis/raw"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/per_task_temporal_interference_diagnosis"),
    )
    args = parser.parse_args()
    rows = load_raw(args.raw_dir)
    primary = [row for row in rows if row["analysis_phase"] == "entry"]
    stage_rows = summarize_stage_bins(primary)
    old_task_rows = summarize_old_tasks(primary)
    case_rows = summarize_seed44_case(rows)
    metrics = overall_metrics(primary)
    write_rows(args.output_dir / "per_stage_bin_summary.csv", STAGE_FIELDS, stage_rows)
    write_rows(args.output_dir / "per_old_task_summary.csv", OLD_TASK_FIELDS, old_task_rows)
    write_rows(args.output_dir / "seed44_task7_case.csv", CASE_FIELDS, case_rows)
    with (args.output_dir / "diagnosis_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
