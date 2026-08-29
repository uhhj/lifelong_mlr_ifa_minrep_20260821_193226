#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
import statistics
import subprocess
from pathlib import Path


LINE = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
TASKS = tuple(range(6, 10))
EPOCHS = (70, 75, 80, 85, 90, 95)
METRICS = ("fwt", "nbt", "auc")
STARTING_SHA = "e2ed8e38e21c0f904516046c2b9be6bbd1efb249"


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


def rounded(value):
    return None if value is None else round(float(value), 6)


def parse_evaluation(path):
    records = {}
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        match = LINE.search(text)
        if match:
            task, epoch, stage = map(int, match.group(1, 2, 3))
            records[(stage, epoch, task)] = float(match.group(4)) * 100.0
    if len(records) != 30:
        raise ValueError(
            f"expected exactly 30 success records in {path}, found {len(records)}"
        )
    selected = {
        stage: max(
            EPOCHS,
            key=lambda epoch: (records[(stage, epoch, stage)], epoch),
        )
        for stage in TASKS
    }
    matrix = [
        [
            None
            if task > stage
            else records[(stage, selected[stage], task)]
            for task in TASKS
        ]
        for stage in TASKS
    ]
    return selected, matrix


def compute_metrics(matrix):
    diagonal = [matrix[index][index] for index in range(4)]
    forgetting = []
    auc_tasks = []
    for task_index in range(4):
        trajectory = [
            matrix[stage][task_index]
            for stage in range(task_index, 4)
        ]
        auc_tasks.append(statistics.mean(trajectory))
        if task_index < 3:
            forgetting.append(
                diagonal[task_index] - statistics.mean(trajectory[1:])
            )
    return {
        "fwt": statistics.mean(diagonal),
        "nbt": statistics.mean(forgetting),
        "auc": statistics.mean(auc_tasks),
    }


def summary(values):
    return {
        "mean": rounded(statistics.mean(values)),
        "sample_std": rounded(statistics.stdev(values)),
    }


def weighted(rows, value_field, weight_field):
    usable = [
        row
        for row in rows
        if row[value_field] != "" and int(row[weight_field]) > 0
    ]
    denominator = sum(int(row[weight_field]) for row in usable)
    if not denominator:
        return None
    return sum(
        float(row[value_field]) * int(row[weight_field])
        for row in usable
    ) / denominator


def aggregate_guard(run, seed):
    rows = read_csv(run / "run_001" / "directional_guard_train_stats.csv")
    if len(rows) != 400:
        raise ValueError(
            f"seed {seed} expected 400 directional rows, found {len(rows)}"
        )
    stage_rows = []
    for stage in TASKS:
        selected = [row for row in rows if int(row["stage"]) == stage]
        if len(selected) != 100:
            raise ValueError(
                f"seed {seed} stage {stage} expected 100 rows, found {len(selected)}"
            )
        batches = sum(int(row["total_batches"]) for row in selected)
        eligible = sum(int(row["eligible_batches"]) for row in selected)
        fires = sum(int(row["fire_count"]) for row in selected)
        result = {
            "seed": seed,
            "stage": stage,
            "total_batches": batches,
            "eligible_batches": eligible,
            "fire_count": fires,
            "fire_rate": None if not eligible else fires / eligible,
            "mean_candidate_cosine": weighted(
                selected, "mean_candidate_cosine", "eligible_batches"
            ),
            "mean_fired_cosine": weighted(
                selected, "mean_fired_cosine", "fire_count"
            ),
            "mean_candidate_score": weighted(
                selected, "mean_candidate_score", "eligible_batches"
            ),
            "mean_old_grad_norm": weighted(
                selected, "mean_old_grad_norm", "eligible_batches"
            ),
            "mean_candidate_delta_norm": weighted(
                selected,
                "mean_candidate_delta_norm",
                "eligible_batches",
            ),
            "score_probe_size": 64,
            "score_probe_batch_size": 32,
            "threshold": 0.0,
            "guard_scale": 0.5,
        }
        if stage == 6 and (eligible or fires):
            raise ValueError("stage6 guard must be inactive")
        if stage > 6 and not eligible:
            raise ValueError(f"seed {seed} stage {stage} has no eligible guard batches")
        for key, value in result.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(
                    f"non-finite guard summary seed={seed} stage={stage} field={key}"
                )
        stage_rows.append(result)
    return stage_rows


def git_value(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--temporal-v1-json",
        type=Path,
        default=Path(
            "reports/temporal_robot_igfa_v1/temporal_robot_igfa_3seed_metrics.json"
        ),
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=Path(
            "reports/baseline_reconciliation/object_3seed_metrics.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/zero_threshold_directional_guard_formal"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporal = json.loads(args.temporal_v1_json.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))

    per_seed = {}
    metric_rows = []
    stage_success_rows = []
    forgetting_rows = []
    selected_epoch_rows = []
    guard_stage_rows = []
    runtime_rows = []
    for seed in args.seeds:
        run = args.runs_root / f"object_seed{seed}_mlr_zero_directional_guard"
        selected, matrix = parse_evaluation(run / f"eval_seed{seed}.log")
        score = compute_metrics(matrix)
        v1 = temporal["per_seed"][str(seed)]
        mlr = baseline["per_seed"]["mlr"][str(seed)]
        delta_v1 = {metric: score[metric] - v1[metric] for metric in METRICS}
        delta_mlr = {metric: score[metric] - mlr[metric] for metric in METRICS}
        per_seed[str(seed)] = {
            **{metric: rounded(score[metric]) for metric in METRICS},
            "delta_vs_temporal_v1": {
                metric: rounded(delta_v1[metric]) for metric in METRICS
            },
            "delta_vs_mlr": {
                metric: rounded(delta_mlr[metric]) for metric in METRICS
            },
            "selected_epochs": {
                str(stage): epoch for stage, epoch in selected.items()
            },
            "r": [
                [rounded(value) for value in row]
                for row in matrix
            ],
        }
        metric_rows.append(
            {
                "seed": seed,
                **{metric: rounded(score[metric]) for metric in METRICS},
                **{
                    f"delta_{metric}_vs_temporal_v1": rounded(
                        delta_v1[metric]
                    )
                    for metric in METRICS
                },
                **{
                    f"delta_{metric}_vs_mlr": rounded(delta_mlr[metric])
                    for metric in METRICS
                },
            }
        )
        for stage_index, stage in enumerate(TASKS):
            selected_epoch_rows.append(
                {"seed": seed, "stage": stage, "selected_epoch": selected[stage]}
            )
            for task_index, task in enumerate(TASKS):
                value = matrix[stage_index][task_index]
                if value is not None:
                    stage_success_rows.append(
                        {
                            "seed": seed,
                            "stage": stage,
                            "task": task,
                            "selected_epoch": selected[stage],
                            "success": rounded(value),
                        }
                    )
        for task_index, task in enumerate(TASKS[:3]):
            diagonal = matrix[task_index][task_index]
            later = [
                matrix[stage][task_index]
                for stage in range(task_index + 1, 4)
            ]
            forgetting_rows.append(
                {
                    "seed": seed,
                    "task": task,
                    "diagonal_success": rounded(diagonal),
                    "mean_later_success": rounded(statistics.mean(later)),
                    "forgetting": rounded(
                        diagonal - statistics.mean(later)
                    ),
                }
            )
        guard_stage_rows.extend(aggregate_guard(run, seed))
        seconds = int((run / "training_wall_seconds.txt").read_text().strip())
        runtime_rows.append(
            {
                "seed": seed,
                "training_wall_seconds": seconds,
                "training_hours": rounded(seconds / 3600.0),
            }
        )

    three_seed = {
        metric: summary([per_seed[str(seed)][metric] for seed in args.seeds])
        for metric in METRICS
    }
    deltas = {}
    for reference_key in ("temporal_v1", "mlr"):
        deltas[reference_key] = {}
        per_seed_key = (
            "delta_vs_temporal_v1"
            if reference_key == "temporal_v1"
            else "delta_vs_mlr"
        )
        for metric in METRICS:
            values = [
                per_seed[str(seed)][per_seed_key][metric]
                for seed in args.seeds
            ]
            deltas[reference_key][metric] = {
                **summary(values),
                "improved_seed_count": sum(
                    value < 0 if metric == "nbt" else value > 0
                    for value in values
                ),
            }

    delta_nbt = deltas["temporal_v1"]["nbt"]["mean"]
    delta_fwt = deltas["temporal_v1"]["fwt"]["mean"]
    delta_auc = deltas["temporal_v1"]["auc"]["mean"]
    nbt_improved = deltas["temporal_v1"]["nbt"]["improved_seed_count"]
    if (
        delta_nbt <= -1.0
        and nbt_improved >= 2
        and delta_fwt >= -1.5
        and delta_auc >= -0.5
    ):
        verdict = "ZERO_THRESHOLD_DIRECTIONAL_GUARD_FORMAL_SUPPORTED"
    elif (
        delta_nbt >= 1.0
        or nbt_improved == 0
        or delta_fwt <= -3.0
        or delta_auc <= -2.0
    ):
        verdict = "ZERO_THRESHOLD_DIRECTIONAL_GUARD_FORMAL_NOT_SUPPORTED"
    else:
        verdict = "ZERO_THRESHOLD_DIRECTIONAL_GUARD_FORMAL_INCONCLUSIVE"

    flags = []
    if deltas["mlr"]["nbt"]["mean"] < 0:
        flags.append("RETENTION_IMPROVES_VS_MLR")
    if delta_auc > 0:
        flags.append("AUC_IMPROVES_VS_TEMPORAL_V1")
    if delta_fwt > 0:
        flags.append("FWT_IMPROVES_VS_TEMPORAL_V1")

    eligible_rows = [row for row in guard_stage_rows if row["eligible_batches"]]
    total_eligible = sum(row["eligible_batches"] for row in eligible_rows)
    total_fires = sum(row["fire_count"] for row in eligible_rows)
    guard_overall_rows = []
    for seed in args.seeds:
        rows = [row for row in eligible_rows if row["seed"] == seed]
        eligible = sum(row["eligible_batches"] for row in rows)
        fires = sum(row["fire_count"] for row in rows)
        guard_overall_rows.append(
            {
                "seed": seed,
                "eligible_batches": eligible,
                "fire_count": fires,
                "fire_rate": rounded(fires / eligible),
                "mean_candidate_cosine": rounded(
                    sum(
                        row["mean_candidate_cosine"] * row["eligible_batches"]
                        for row in rows
                    )
                    / eligible
                ),
                "mean_fired_cosine": rounded(
                    sum(
                        (row["mean_fired_cosine"] or 0) * row["fire_count"]
                        for row in rows
                    )
                    / fires
                ) if fires else None,
            }
        )
    guard_mechanism = {
        "eligible_batches": total_eligible,
        "fire_count": total_fires,
        "fire_rate": rounded(total_fires / total_eligible),
        "mean_candidate_cosine": rounded(
            sum(
                row["mean_candidate_cosine"] * row["eligible_batches"]
                for row in eligible_rows
            )
            / total_eligible
        ),
        "mean_fired_cosine": rounded(
            sum(
                (row["mean_fired_cosine"] or 0) * row["fire_count"]
                for row in eligible_rows
            )
            / total_fires
        ) if total_fires else None,
    }

    branch = git_value("branch", "--show-current")
    source_sha = git_value("rev-parse", "HEAD")
    payload = {
        "benchmark": "LIBERO_OBJECT",
        "method": "zero_threshold_optimizer_aware_directional_guard",
        "seeds": args.seeds,
        "standard_deviation": "sample",
        "verdict": verdict,
        "flags": flags,
        "per_seed": per_seed,
        "three_seed_summary": three_seed,
        "delta_vs_temporal_v1": deltas["temporal_v1"],
        "delta_vs_mlr": deltas["mlr"],
        "guard_mechanism": guard_mechanism,
        "frozen_references": {
            "temporal_v1": temporal["three_seed_summary"],
            "mlr": baseline["three_seed_summary"]["mlr"],
        },
        "identity": {
            "branch": branch,
            "starting_sha": STARTING_SHA,
            "formal_experiment_source_sha": source_sha,
        },
    }

    write_csv(args.output_dir / "per_seed_metrics.csv", metric_rows[0].keys(), metric_rows)
    write_csv(args.output_dir / "per_stage_success.csv", stage_success_rows[0].keys(), stage_success_rows)
    write_csv(args.output_dir / "per_task_forgetting.csv", forgetting_rows[0].keys(), forgetting_rows)
    write_csv(args.output_dir / "selected_epochs.csv", selected_epoch_rows[0].keys(), selected_epoch_rows)
    clean_guard_stages = [
        {key: rounded(value) if isinstance(value, float) else value for key, value in row.items()}
        for row in guard_stage_rows
    ]
    write_csv(args.output_dir / "guard_stage_summary.csv", clean_guard_stages[0].keys(), clean_guard_stages)
    write_csv(args.output_dir / "guard_overall_summary.csv", guard_overall_rows[0].keys(), guard_overall_rows)
    write_csv(args.output_dir / "runtime_summary.csv", runtime_rows[0].keys(), runtime_rows)
    (args.output_dir / "zero_guard_3seed_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    def value(reference, metric):
        return reference[metric]["mean"]

    report = f"""# Zero-Threshold Optimizer-Aware Directional Guard Formal

## Identity

- Verdict: `{verdict}`
- Flags: `{', '.join(flags) if flags else 'none'}`
- Branch: `{branch}`
- Starting SHA: `{STARTING_SHA}`
- Formal experiment source SHA: `{source_sha}`

## Formal protocol

LIBERO_OBJECT, seeds 42/43/44, tasks 6-9, 100 epochs/stage, batch 10,
AdamW LR 1e-4, 20 rollout trials, Temporal V1 five-bin gate, IFA disabled.
The directional guard uses fresh old gradients every update, tau=0, scale=0.5,
and a deterministic 64-record/task subset of the existing replay buffer.
Formal score probes are a deterministic subset of the existing 50% replay
memory, not additional old-task data.

## Main metrics

| Method | FWT | NBT ↓ | AUC |
|---|---:|---:|---:|
| MLR | {value(baseline['three_seed_summary']['mlr'], 'fwt'):.3f} | {value(baseline['three_seed_summary']['mlr'], 'nbt'):.3f} | {value(baseline['three_seed_summary']['mlr'], 'auc'):.3f} |
| Temporal V1 | {value(temporal['three_seed_summary'], 'fwt'):.3f} | {value(temporal['three_seed_summary'], 'nbt'):.3f} | {value(temporal['three_seed_summary'], 'auc'):.3f} |
| Zero Guard | {three_seed['fwt']['mean']:.3f} ± {three_seed['fwt']['sample_std']:.3f} | {three_seed['nbt']['mean']:.3f} ± {three_seed['nbt']['sample_std']:.3f} | {three_seed['auc']['mean']:.3f} ± {three_seed['auc']['sample_std']:.3f} |

## Per seed

| Seed | Guard FWT | Guard NBT | Guard AUC | ΔNBT vs V1 | ΔFWT vs V1 | ΔAUC vs V1 |
|---:|---:|---:|---:|---:|---:|---:|
"""
    for seed in args.seeds:
        row = per_seed[str(seed)]
        delta = row["delta_vs_temporal_v1"]
        report += (
            f"| {seed} | {row['fwt']:.3f} | {row['nbt']:.3f} | "
            f"{row['auc']:.3f} | {delta['nbt']:+.3f} | "
            f"{delta['fwt']:+.3f} | {delta['auc']:+.3f} |\n"
        )
    report += "\n## Selected epochs\n\n"
    report += "| Seed | Stage 6 | Stage 7 | Stage 8 | Stage 9 |\n|---:|---:|---:|---:|---:|\n"
    for seed in args.seeds:
        epochs = per_seed[str(seed)]["selected_epochs"]
        report += f"| {seed} | {epochs['6']} | {epochs['7']} | {epochs['8']} | {epochs['9']} |\n"
    worst = max(forgetting_rows, key=lambda row: row["forgetting"])
    report += f"""

## Per-task forgetting

Complete values are in `per_task_forgetting.csv`. The largest observed
forgetting was seed {worst['seed']} task {worst['task']} at
{worst['forgetting']:.3f} percentage points.

## Guard mechanism

Across stages 7-9 the guard fired on {guard_mechanism['fire_rate']:.6f} of
{guard_mechanism['eligible_batches']} eligible updates. Stage 6 had no old-task
probe and zero eligible guard updates. All stage-level mechanism values were
finite; see `guard_stage_summary.csv`.

## Runtime

Per-seed training wall time and total observed GPU-hours are recorded in
`runtime_summary.csv`. Total observed training GPU-hours:
{sum(row['training_hours'] for row in runtime_rows):.3f}.

## Observed

The formal rollout metrics, selected checkpoints, per-task trajectories, and
guard fire statistics above were produced by the frozen three-seed protocol.

## Inferred

H17 is classified as `{verdict}` by the preregistered NBT, FWT, and AUC rules.
This determines whether the H16 short-horizon BC-loss retention benefit
translated into formal rollout retention without post-result tuning.

## Not established

Fresh old-task gradients are still recomputed every update; formal support
would establish efficacy, not efficiency.

The method currently uses 64 replay samples per old task as a fixed
score-gradient subset.

The score probes are within the existing replay memory, so no additional
old-task dataset access is introduced.

The guard full-commits AdamW moments while scaling only the parameter endpoint.

Three seeds establish replication in this benchmark, not universal
cross-benchmark generalization.

LIBERO_OBJECT success does not yet establish GOAL/LIBERO-50 transfer.
"""
    (args.output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
