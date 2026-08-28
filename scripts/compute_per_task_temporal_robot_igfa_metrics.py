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


def rounded(value):
    return None if value is None else round(value, 6)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
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


def summarize(values):
    return {
        "mean": rounded(statistics.mean(values)),
        "sample_std": rounded(statistics.stdev(values)),
    }


def weighted(rows, value_column, count_column):
    denominator = sum(int(row[count_column]) for row in rows)
    if not denominator:
        return None
    return sum(
        float(row[value_column]) * int(row[count_column])
        for row in rows if row[value_column] != ""
    ) / denominator


def aggregate_mechanism(run, seed):
    bins = read_csv(run / "run_001" / "per_task_temporal_igfa_gate_stats.csv")
    overall = read_csv(run / "run_001" / "per_task_temporal_igfa_gate_overall.csv")
    if len(bins) != 2000 or len(overall) != 400:
        raise ValueError(
            f"seed {seed} expected 2000 bin rows and 400 overall rows, "
            f"found {len(bins)} and {len(overall)}"
        )

    def aggregate(selected_bins, selected_overall, stage, bin_id="all"):
        candidates = sum(int(row["candidate_bins"]) for row in selected_bins)
        multi_old = sum(int(row["multi_old_candidate_bins"]) for row in selected_bins)
        present = sum(int(row["present_old_constraints"]) for row in selected_bins)
        conflicts = sum(int(row["initial_conflicting_constraints"]) for row in selected_bins)
        active = sum(int(row["active_constraints"]) for row in selected_bins)
        fired = sum(
            float(row["projection_fire_rate"] or 0.0) * int(row["candidate_bins"])
            for row in selected_bins
        )
        violations = sum(int(row["post_constraint_violation_count"]) for row in selected_bins)
        total_batches = sum(int(row["total_batches"]) for row in selected_overall)
        return {
            "seed": seed,
            "stage": stage,
            "bin": bin_id,
            "total_batches": total_batches,
            "candidate_bins": candidates,
            "multi_old_candidate_bins": multi_old,
            "present_old_constraints": present,
            "initial_conflicting_constraints": conflicts,
            "active_constraints": active,
            "initial_conflict_bin_rate": None if not candidates else rounded(fired / candidates),
            "initial_conflicting_constraint_rate": None if not present else rounded(conflicts / present),
            "mean_present_old_tasks": None if not candidates else rounded(present / candidates),
            "mean_active_constraints": None if not candidates else rounded(active / candidates),
            "mean_projection_cost": rounded(weighted(selected_bins, "mean_projection_cost", "candidate_bins")),
            "post_constraint_violation_count": violations,
            "post_constraint_violation_rate": None if not present else rounded(violations / present),
            "compatible_became_active_count": sum(int(row["compatible_became_active_count"]) for row in selected_bins),
            "zero_old_gradient_count": sum(int(row["zero_old_gradient_count"]) for row in selected_bins),
            "active_count_0": sum(int(row["active_count_0"]) for row in selected_bins),
            "active_count_1": sum(int(row["active_count_1"]) for row in selected_bins),
            "active_count_2": sum(int(row["active_count_2"]) for row in selected_bins),
            "active_count_3": sum(int(row["active_count_3"]) for row in selected_bins),
        }

    stages = []
    by_bin = []
    for stage in TASKS:
        stage_bins = [row for row in bins if int(row["stage"]) == stage]
        stage_overall = [row for row in overall if int(row["stage"]) == stage]
        stages.append(aggregate(stage_bins, stage_overall, stage))
        for bin_id in range(5):
            selected = [row for row in stage_bins if int(row["bin"]) == bin_id]
            by_bin.append(aggregate(selected, stage_overall, stage, bin_id))
    return stages, by_bin


def matrix_rows(seed, selected, matrix):
    rows = []
    for stage_index, stage in enumerate(TASKS):
        for task_index, task in enumerate(TASKS):
            if matrix[stage_index][task_index] is not None:
                rows.append({
                    "seed": seed, "stage": stage, "task": task,
                    "selected_epoch": selected[stage],
                    "success": rounded(matrix[stage_index][task_index]),
                })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--baseline-json", type=Path, default=Path("reports/baseline_reconciliation/object_3seed_metrics.json"))
    parser.add_argument("--static-json", type=Path, default=Path("reports/static_robot_igfa/static_robot_igfa_3seed_metrics.json"))
    parser.add_argument("--v1-json", type=Path, default=Path("reports/temporal_robot_igfa_v1/temporal_robot_igfa_3seed_metrics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports/per_task_temporal_robot_igfa_v2"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
    static = json.loads(args.static_json.read_text(encoding="utf-8"))
    v1 = json.loads(args.v1_json.read_text(encoding="utf-8"))

    per_seed = {}
    metric_rows = []
    stage_rows = []
    fwt_rows = []
    forgetting_rows = []
    mechanism_stage = []
    mechanism_bin = []
    for seed in args.seeds:
        run = args.runs_root / f"object_seed{seed}_mlr_per_task_temporal_igfa"
        selected, matrix = parse_evaluation(run / f"eval_seed{seed}.log")
        score = compute_metrics(matrix)
        references = {
            "v1": v1["per_seed"][str(seed)],
            "mlr": baseline["per_seed"]["mlr"][str(seed)],
        }
        deltas = {
            name: {metric: score[metric] - record[metric] for metric in METRICS}
            for name, record in references.items()
        }
        per_seed[str(seed)] = {
            **{metric: rounded(score[metric]) for metric in METRICS},
            "selected_epochs": {str(stage): selected[stage] for stage in TASKS},
            "r": [[rounded(value) for value in row] for row in matrix],
            "delta_vs_temporal_v1": {key: rounded(value) for key, value in deltas["v1"].items()},
            "delta_vs_mlr": {key: rounded(value) for key, value in deltas["mlr"].items()},
        }
        metric_rows.append({
            "seed": seed,
            **{metric: rounded(score[metric]) for metric in METRICS},
            **{f"delta_{metric}_vs_v1": rounded(deltas["v1"][metric]) for metric in METRICS},
            **{f"delta_{metric}_vs_mlr": rounded(deltas["mlr"][metric]) for metric in METRICS},
        })
        stage_rows.extend(matrix_rows(seed, selected, matrix))
        for index, task in enumerate(TASKS):
            diagonal = matrix[index][index]
            later = [matrix[stage][index] for stage in range(index + 1, 4)]
            fwt_rows.append({"seed": seed, "task": task, "fwt": rounded(diagonal)})
            forgetting_rows.append({
                "seed": seed, "task": task, "diagonal_success": rounded(diagonal),
                "mean_later_success": None if not later else rounded(statistics.mean(later)),
                "forgetting": None if not later else rounded(diagonal - statistics.mean(later)),
            })
        seed_stage, seed_bin = aggregate_mechanism(run, seed)
        mechanism_stage.extend(seed_stage)
        mechanism_bin.extend(seed_bin)

    summary = {
        metric: summarize([per_seed[str(seed)][metric] for seed in args.seeds])
        for metric in METRICS
    }
    delta_summary = {}
    for reference in ("temporal_v1", "mlr"):
        key = f"delta_vs_{reference}"
        delta_summary[reference] = {}
        for metric in METRICS:
            values = [per_seed[str(seed)][key][metric] for seed in args.seeds]
            delta_summary[reference][metric] = {
                **summarize(values),
                "wins": sum(value < 0 for value in values) if metric == "nbt" else sum(value > 0 for value in values),
            }

    candidates = sum(row["candidate_bins"] for row in mechanism_stage)
    present = sum(row["present_old_constraints"] for row in mechanism_stage)
    violations = sum(row["post_constraint_violation_count"] for row in mechanism_stage)
    mechanism = {
        "candidate_bins": candidates,
        "multi_old_candidate_bins": sum(row["multi_old_candidate_bins"] for row in mechanism_stage),
        "initial_conflict_bin_rate": rounded(sum((row["initial_conflict_bin_rate"] or 0) * row["candidate_bins"] for row in mechanism_stage) / candidates),
        "mean_active_constraints": rounded(sum(row["active_constraints"] for row in mechanism_stage) / candidates),
        "mean_projection_cost": rounded(sum((row["mean_projection_cost"] or 0) * row["candidate_bins"] for row in mechanism_stage) / candidates),
        "post_constraint_violation_count": violations,
        "post_constraint_violation_rate": rounded(violations / present),
        "compatible_became_active_count": sum(row["compatible_became_active_count"] for row in mechanism_stage),
        "zero_old_gradient_count": sum(row["zero_old_gradient_count"] for row in mechanism_stage),
        "active_count_distribution": {
            str(count): sum(row[f"active_count_{count}"] for row in mechanism_stage)
            for count in range(4)
        },
    }

    v1_delta = delta_summary["temporal_v1"]
    if (
        v1_delta["nbt"]["mean"] <= -2.0
        and v1_delta["nbt"]["wins"] >= 2
        and v1_delta["fwt"]["mean"] >= -1.0
        and v1_delta["auc"]["mean"] >= -1.0
        and mechanism["post_constraint_violation_rate"] <= 0.001
    ):
        verdict = "PER_TASK_TEMPORAL_ROBOT_IGFA_SUPPORTED"
    elif (
        v1_delta["nbt"]["mean"] >= 1.0
        or v1_delta["fwt"]["mean"] <= -3.0
        or v1_delta["auc"]["mean"] <= -3.0
    ):
        verdict = "PER_TASK_TEMPORAL_ROBOT_IGFA_NOT_SUPPORTED"
    else:
        verdict = "PER_TASK_TEMPORAL_ROBOT_IGFA_INCONCLUSIVE"
    mlr_delta = delta_summary["mlr"]
    practical = "SUPPORTED" if (
        mlr_delta["auc"]["mean"] >= 1.0
        and mlr_delta["auc"]["wins"] >= 2
        and mlr_delta["fwt"]["mean"] >= -1.0
    ) else "NOT ESTABLISHED"

    v2_seed44 = per_seed["44"]["r"]
    v1_seed44 = v1["per_seed"]["44"]["r"]
    seed44_task7 = {
        "v2_diagonal": v2_seed44[1][1],
        "v2_post_stage8": v2_seed44[2][1],
        "v2_post_stage9": v2_seed44[3][1],
        "v2_forgetting": rounded(v2_seed44[1][1] - statistics.mean([v2_seed44[2][1], v2_seed44[3][1]])),
        "v1_forgetting": rounded(v1_seed44[1][1] - statistics.mean([v1_seed44[2][1], v1_seed44[3][1]])),
    }
    seed44_task7["forgetting_delta_v2_minus_v1"] = rounded(
        seed44_task7["v2_forgetting"] - seed44_task7["v1_forgetting"]
    )

    payload = {
        "benchmark": "LIBERO_OBJECT",
        "method": "mlr_per_task_temporal_igfa",
        "seeds": args.seeds,
        "standard_deviation": "sample",
        "verdict": verdict,
        "practical_benefit_vs_mlr": practical,
        "per_seed": per_seed,
        "three_seed_summary": summary,
        "delta_vs_temporal_v1": delta_summary["temporal_v1"],
        "delta_vs_mlr": delta_summary["mlr"],
        "mechanism": mechanism,
        "seed44_task7": seed44_task7,
        "frozen_references": {
            "mlr": baseline["three_seed_summary"]["mlr"],
            "mlr_ifa": baseline["three_seed_summary"]["mlr_ifa"],
            "static": static["three_seed_summary"],
            "temporal_v1": v1["three_seed_summary"],
        },
    }
    (args.output_dir / "per_task_temporal_robot_igfa_3seed_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "per_seed_metrics.csv", metric_rows)
    write_csv(args.output_dir / "per_stage_success.csv", stage_rows)
    write_csv(args.output_dir / "per_task_fwt.csv", fwt_rows)
    write_csv(args.output_dir / "per_task_forgetting.csv", forgetting_rows)
    write_csv(args.output_dir / "mechanism_by_stage.csv", mechanism_stage)
    write_csv(args.output_dir / "mechanism_by_bin.csv", mechanism_bin)
    write_csv(args.output_dir / "mechanism_summary.csv", [{"scope": "three_seed", **mechanism}])

    def metric_text(record):
        return " / ".join(f"{record[name]:.3f}" for name in METRICS)

    lines = [
        "# Per-Task Temporal Robot-IGFA V2",
        "", "## Identity", "",
        f"- Verdict: `{verdict}`",
        f"- Practical benefit vs MLR: `{practical}`",
        "- Branch: `research/per-task-temporal-robot-igfa-v2`",
        "- Starting SHA: `83d98eb1e553fe08fae6019f0cbc874c18859115`",
        "", "## Method", "",
        "Policy-head only; five progress bins; one constraint per present old task; order-independent minimum-change projection by active-set enumeration (at most three constraints); alpha 1; Temporal V1 delta correction; transformer unchanged; IFA disabled; replay and historical scheduler semantics unchanged.",
        "", "## Tests and smoke", "",
        "Focused solver, gate=0 parity, single-old Temporal V1 parity, order-invariance, masked-conflict, feasibility, minimum-change, and no-anchor tests passed. The formal runs were admitted only after the tasks 6-8 smoke observed multi-old constraints with zero post-projection violations.",
        "", "## Per-seed metrics (FWT / NBT / AUC)", "",
    ]
    lines.extend(f"- seed{seed}: {metric_text(per_seed[str(seed)])}" for seed in args.seeds)
    lines += [
        "", "## Three-seed mean ± sample std", "",
        *[f"- {metric.upper()}: {summary[metric]['mean']:.3f} ± {summary[metric]['sample_std']:.3f}" for metric in METRICS],
        "", "## V2 vs Temporal V1", "",
        *[f"- Δ{metric.upper()}: {v1_delta[metric]['mean']:.3f}; same-seed wins {v1_delta[metric]['wins']}/3" for metric in METRICS],
        "", "## V2 vs MLR", "",
        *[f"- Δ{metric.upper()}: {mlr_delta[metric]['mean']:.3f}; same-seed wins {mlr_delta[metric]['wins']}/3" for metric in METRICS],
        "", "## Mechanism", "",
        f"- Candidate bins: {mechanism['candidate_bins']}",
        f"- Multi-old candidate bins: {mechanism['multi_old_candidate_bins']}",
        f"- Initial-conflict bin rate: {100 * mechanism['initial_conflict_bin_rate']:.3f}%",
        f"- Mean active constraints: {mechanism['mean_active_constraints']:.6f}",
        f"- Active-count distribution: {mechanism['active_count_distribution']}",
        f"- Mean projection cost: {mechanism['mean_projection_cost']:.6f}",
        f"- Post-constraint violation rate: {100 * mechanism['post_constraint_violation_rate']:.6f}%",
        f"- Compatible-became-active: {mechanism['compatible_became_active_count']}",
        f"- Zero-old-gradient count: {mechanism['zero_old_gradient_count']}",
        "", "## Seed44 task7", "",
        f"- V2 diagonal / post-stage8 / post-stage9: {seed44_task7['v2_diagonal']:.3f} / {seed44_task7['v2_post_stage8']:.3f} / {seed44_task7['v2_post_stage9']:.3f}",
        f"- V1 forgetting: {seed44_task7['v1_forgetting']:.3f}",
        f"- V2 forgetting: {seed44_task7['v2_forgetting']:.3f}",
        f"- Delta: {seed44_task7['forgetting_delta_v2_minus_v1']:.3f}",
        "", "## Observed", "",
        "All metrics and mechanism statements above come from the frozen three-seed formal protocol and complete 30-record evaluation logs.",
        "", "## Inferred", "",
        f"H7 verdict is `{verdict}` under the preregistered descriptive thresholds.",
        "", "## Not established", "",
        "These three seeds do not establish statistical significance or performance outside LIBERO-OBJECT.",
        "", "## Next action", "",
        "If supported, proceed to method consolidation/ablation; otherwise inspect projection cost and task/stage localization before changing alpha or method scope.",
        "", "## Exact successful commands", "",
        "```bash",
        "for seed in 42 43 44; do",
        "  SEED=$seed bash scripts/run_object_mlr_per_task_temporal_robot_igfa.sh",
        "  SEED=$seed bash test.sbatch mlr_per_task_temporal_igfa",
        "done",
        "python scripts/compute_per_task_temporal_robot_igfa_metrics.py",
        "```", "",
    ]
    (args.output_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
