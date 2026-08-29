#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import subprocess
from pathlib import Path

from libero.lifelong.analysis.candidate_directional_guard import (
    paired_drift_metrics,
)
from scripts.diagnose_online_first_order_margin import write_csv


SEEDS = (42, 43)
STAGES = (8, 9)
STARTING_SHA = "d6cdf1bfbe3a24b0c614075a83cf83a7d0736847"


PAIR_FIELDS = [
    "seed", "stage", "threshold", "guard_scale", "start_old_loss",
    "start_current_loss", "control_old_h100", "guard_old_h100",
    "control_old_drift_h100", "guard_old_drift_h100", "old_benefit_h100",
    "relative_old_benefit_h100", "control_current_h100",
    "guard_current_h100", "control_current_drift_h100",
    "guard_current_drift_h100", "current_harm_h100",
    "relative_current_harm_h100", "plasticity_preservation_h100",
    "control_old_h300", "guard_old_h300", "control_old_drift_h300",
    "guard_old_drift_h300", "old_benefit_h300",
    "relative_old_benefit_h300", "control_current_h300",
    "guard_current_h300", "control_current_drift_h300",
    "guard_current_drift_h300", "current_harm_h300",
    "relative_current_harm_h300", "plasticity_preservation_h300",
    "guard_fire_count_h100", "guard_fire_rate_h100",
    "guard_fire_count_h300", "guard_fire_rate_h300",
    "control_would_fire_rate_h100", "control_would_fire_rate_h300",
]
OVERALL_FIELDS = [
    "pairs", "positive_old_benefit_pairs_h100",
    "positive_old_benefit_pairs_h300", "mean_relative_old_benefit_h100",
    "median_relative_old_benefit_h100", "mean_relative_old_benefit_h300",
    "median_relative_old_benefit_h300", "mean_relative_current_harm_h100",
    "max_relative_current_harm_h100", "mean_relative_current_harm_h300",
    "max_relative_current_harm_h300", "pooled_guard_fire_rate_h100",
    "pooled_guard_fire_rate_h300", "pairs_with_at_least_5_fires_h100",
    "branch_start_parity", "all_metrics_finite", "verdict",
]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def current_head(root):
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * float(probability)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def probe_lookup(rows):
    result = {}
    for row in rows:
        key = (row["arm"], int(row["update"]))
        if key in result:
            raise ValueError(f"duplicate probe row {key}")
        result[key] = row
    return result


def finite_rows(rows, fields):
    for row in rows:
        for field in fields:
            value = row.get(field, "")
            if value == "" or not math.isfinite(float(value)):
                return False
    return True


def context_pair(seed, stage, steps, probes):
    lookup = probe_lookup(probes)
    required = {
        (arm, update) for arm in ("control", "guard") for update in (0, 100, 300)
    }
    if set(lookup) != required:
        raise ValueError(f"seed{seed} stage{stage} probe points differ: {set(lookup)}")
    starts = [lookup[(arm, 0)] for arm in ("control", "guard")]
    old_parity = abs(float(starts[0]["old_mean_loss"]) - float(starts[1]["old_mean_loss"]))
    current_parity = abs(float(starts[0]["current_loss"]) - float(starts[1]["current_loss"]))
    start_old = float(starts[0]["old_mean_loss"])
    start_current = float(starts[0]["current_loss"])
    row = {
        "seed": seed,
        "stage": stage,
        "threshold": float(starts[0]["threshold"]),
        "guard_scale": float(starts[0]["guard_scale"]),
        "start_old_loss": start_old,
        "start_current_loss": start_current,
    }
    for horizon in (100, 300):
        control = lookup[("control", horizon)]
        guard = lookup[("guard", horizon)]
        effect = paired_drift_metrics(
            start_old,
            start_current,
            {"old": control["old_mean_loss"], "current": control["current_loss"]},
            {"old": guard["old_mean_loss"], "current": guard["current_loss"]},
        )
        suffix = f"h{horizon}"
        row.update(
            {
                f"control_old_{suffix}": float(control["old_mean_loss"]),
                f"guard_old_{suffix}": float(guard["old_mean_loss"]),
                f"control_old_drift_{suffix}": effect["control_old_drift"],
                f"guard_old_drift_{suffix}": effect["guard_old_drift"],
                f"old_benefit_{suffix}": effect["old_benefit"],
                f"relative_old_benefit_{suffix}": effect["relative_old_benefit"],
                f"control_current_{suffix}": float(control["current_loss"]),
                f"guard_current_{suffix}": float(guard["current_loss"]),
                f"control_current_drift_{suffix}": effect["control_current_drift"],
                f"guard_current_drift_{suffix}": effect["guard_current_drift"],
                f"current_harm_{suffix}": effect["current_harm"],
                f"relative_current_harm_{suffix}": effect["relative_current_harm"],
                f"plasticity_preservation_{suffix}": effect["plasticity_preservation"],
            }
        )
    arm_steps = {arm: [item for item in steps if item["arm"] == arm] for arm in ("control", "guard")}
    if any(len(items) != 300 for items in arm_steps.values()):
        raise ValueError(f"seed{seed} stage{stage} does not have 300 steps per arm")
    for horizon in (100, 300):
        guard_items = [item for item in arm_steps["guard"] if int(item["update"]) <= horizon]
        control_items = [item for item in arm_steps["control"] if int(item["update"]) <= horizon]
        fire_count = sum(int(item["guard_fire"]) for item in guard_items)
        row[f"guard_fire_count_h{horizon}"] = fire_count
        row[f"guard_fire_rate_h{horizon}"] = fire_count / horizon
        row[f"control_would_fire_rate_h{horizon}"] = (
            sum(int(item["would_fire"]) for item in control_items) / horizon
        )
    return row, old_parity, current_parity


def task_specific_analysis(rows, pair_by_context):
    grouped = {}
    for row in rows:
        key = (
            int(row["seed"]), int(row["stage"]), row["arm"],
            int(row["update"]), int(row["old_task"]),
        )
        grouped[key] = float(row["loss"])
    worst = -float("inf")
    flags = []
    for (seed, stage), pair in pair_by_context.items():
        for old_task in range(6, stage):
            control_start = grouped[(seed, stage, "control", 0, old_task)]
            guard_start = grouped[(seed, stage, "guard", 0, old_task)]
            if abs(control_start - guard_start) > 1e-6:
                raise ValueError("per-task branch start parity failed")
            control_drift = grouped[(seed, stage, "control", 100, old_task)] - control_start
            guard_drift = grouped[(seed, stage, "guard", 100, old_task)] - guard_start
            relative_extra_harm = (guard_drift - control_drift) / (abs(control_start) + 1e-12)
            worst = max(worst, relative_extra_harm)
            if pair["old_benefit_h100"] > 0.0 and relative_extra_harm > 0.02:
                flags.append((seed, stage, old_task, relative_extra_harm))
    return flags, worst


def fmt(value, digits=8):
    return f"{float(value):.{digits}g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("reports/candidate_directional_guard_smoke"),
    )
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    raw_dir = report_dir / "raw"
    all_steps = []
    all_probes = []
    all_old_tasks = []
    pair_rows = []
    parity_values = []
    for seed in SEEDS:
        for stage in STAGES:
            prefix = f"seed{seed}_stage{stage}"
            steps = read_csv(raw_dir / f"{prefix}_steps.csv")
            probes = read_csv(raw_dir / f"{prefix}_probe.csv")
            old_tasks = read_csv(raw_dir / f"{prefix}_old_task_probe.csv")
            pair, old_parity, current_parity = context_pair(seed, stage, steps, probes)
            pair_rows.append(pair)
            parity_values.extend((old_parity, current_parity))
            all_steps.extend(steps)
            all_probes.extend(probes)
            all_old_tasks.extend(old_tasks)
    branch_parity = max(parity_values) <= 1e-6
    all_finite = (
        finite_rows(
            all_steps,
            [
                "training_loss", "candidate_step_score", "candidate_step_cosine",
                "candidate_delta_norm", "old_grad_norm", "predicted_applied_score",
                "predicted_applied_delta_norm", "lr", "arm_wall_seconds",
            ],
        )
        and finite_rows(all_probes, ["old_mean_loss", "current_loss"])
        and finite_rows(all_old_tasks, ["loss"])
    )
    positive_h100 = sum(row["old_benefit_h100"] > 0.0 for row in pair_rows)
    positive_h300 = sum(row["old_benefit_h300"] > 0.0 for row in pair_rows)
    relative_old_h100 = [row["relative_old_benefit_h100"] for row in pair_rows]
    relative_old_h300 = [row["relative_old_benefit_h300"] for row in pair_rows]
    relative_current_h100 = [row["relative_current_harm_h100"] for row in pair_rows]
    relative_current_h300 = [row["relative_current_harm_h300"] for row in pair_rows]
    pooled_fire_h100 = sum(row["guard_fire_count_h100"] for row in pair_rows) / 400
    pooled_fire_h300 = sum(row["guard_fire_count_h300"] for row in pair_rows) / 1200
    pairs_five = sum(row["guard_fire_count_h100"] >= 5 for row in pair_rows)
    evidence_floor = (
        branch_parity and all_finite and pairs_five == 4
        and 0.05 <= pooled_fire_h100 <= 0.80
    )
    mean_old_h100 = statistics.mean(relative_old_h100)
    mean_old_h300 = statistics.mean(relative_old_h300)
    mean_current_h100 = statistics.mean(relative_current_h100)
    max_current_h100 = max(relative_current_h100)
    retention_failure = (
        (positive_h100 <= 1 and mean_old_h100 <= 0.0)
        or mean_old_h100 <= -0.001
    )
    plasticity_failure = (
        mean_current_h100 > 0.02
        or sum(value > 0.02 for value in relative_current_h100) >= 2
    )
    supported = (
        positive_h100 >= 3
        and mean_old_h100 >= 0.001
        and mean_old_h300 >= 0.0
        and mean_current_h100 <= 0.005
        and max_current_h100 <= 0.02
    )
    if not evidence_floor:
        verdict = "CANDIDATE_DIRECTIONAL_GUARD_SMOKE_INCONCLUSIVE"
    elif retention_failure or plasticity_failure:
        verdict = "CANDIDATE_DIRECTIONAL_GUARD_SMOKE_NOT_SUPPORTED"
    elif supported:
        verdict = "CANDIDATE_DIRECTIONAL_GUARD_SMOKE_SUPPORTED"
    else:
        verdict = "CANDIDATE_DIRECTIONAL_GUARD_SMOKE_INCONCLUSIVE"
    overall = {
        "pairs": 4,
        "positive_old_benefit_pairs_h100": positive_h100,
        "positive_old_benefit_pairs_h300": positive_h300,
        "mean_relative_old_benefit_h100": mean_old_h100,
        "median_relative_old_benefit_h100": statistics.median(relative_old_h100),
        "mean_relative_old_benefit_h300": mean_old_h300,
        "median_relative_old_benefit_h300": statistics.median(relative_old_h300),
        "mean_relative_current_harm_h100": mean_current_h100,
        "max_relative_current_harm_h100": max_current_h100,
        "mean_relative_current_harm_h300": statistics.mean(relative_current_h300),
        "max_relative_current_harm_h300": max(relative_current_h300),
        "pooled_guard_fire_rate_h100": pooled_fire_h100,
        "pooled_guard_fire_rate_h300": pooled_fire_h300,
        "pairs_with_at_least_5_fires_h100": pairs_five,
        "branch_start_parity": "PASS" if branch_parity else "FAIL",
        "all_metrics_finite": int(all_finite),
        "verdict": verdict,
    }
    pair_by_context = {(row["seed"], row["stage"]): row for row in pair_rows}
    task_flags, worst_task_harm = task_specific_analysis(
        all_old_tasks, pair_by_context
    )
    flags = ["TASK_SPECIFIC_RETENTION_TRADEOFF_OBSERVED"] if task_flags else []
    guard_steps = [row for row in all_steps if row["arm"] == "guard"]
    candidate_cosines = [float(row["candidate_step_cosine"]) for row in all_steps]
    fired = [row for row in guard_steps if int(row["guard_fire"]) == 1]
    fired_cosines = [float(row["candidate_step_cosine"]) for row in fired]
    score_mean = statistics.mean(float(row["candidate_step_score"]) for row in all_steps)
    predicted_reduction = sum(
        (1.0 - float(row["guard_scale"])) * float(row["candidate_step_score"])
        for row in fired
    )
    runtime_by_arm = {}
    for row in all_steps:
        key = (int(row["seed"]), int(row["stage"]), row["arm"])
        runtime_by_arm[key] = float(row["arm_wall_seconds"])
    control_runtime = sum(value for key, value in runtime_by_arm.items() if key[2] == "control")
    guard_runtime = sum(value for key, value in runtime_by_arm.items() if key[2] == "guard")

    write_csv(report_dir / "step_raw.csv", list(all_steps[0]), all_steps)
    write_csv(report_dir / "probe_curve.csv", list(all_probes[0]), all_probes)
    write_csv(
        report_dir / "old_task_probe_curve.csv", list(all_old_tasks[0]), all_old_tasks
    )
    write_csv(report_dir / "pair_summary.csv", PAIR_FIELDS, pair_rows)
    write_csv(report_dir / "overall_summary.csv", OVERALL_FIELDS, [overall])

    root = report_dir.parents[1]
    head = current_head(root)
    lines = [
        "# Candidate Directional Guard Intervention Smoke",
        "",
        f"Verdict: `{verdict}`",
        f"Flags: `{', '.join(flags) if flags else 'NONE'}`",
        "Branch: `research/candidate-directional-guard-smoke`",
        f"Starting SHA: `{STARTING_SHA}`",
        f"Analysis code SHA at report generation: `{head}`",
        "Final SHA: branch HEAD reported in the final handoff after committing this report.",
        "",
        "## Protocol",
        "",
        "Temporal V1; seeds 42/43; stages 8/9; previous-stage ep95 entry proxy; "
        "10 common warmup epochs; exact paired Control/Guard branch restore; "
        "300 intervention updates per arm; H100 primary and H300 secondary. "
        "The fixed guard scale is 0.5. Thresholds are prior H13 full/H100/"
        "endpoint_harmful/actual_step_cosine valid LOSO thresholds: "
        "seed42=0.002288496925602411 and seed43=0.004145668183330901. "
        "Old label probes use 256 records/task, fresh old score gradients use "
        "64 records/task at every step, and the fixed current probe uses 256 records.",
        "",
        f"Branch start parity: `{'PASS' if branch_parity else 'FAIL'}` "
        f"(maximum absolute probe difference {fmt(max(parity_values))}).",
        f"All required metrics finite: `{all_finite}`.",
        "",
        "## H100 paired results",
        "",
        "| Seed | Stage | Fires | Fire rate | Control old drift | Guard old drift | Old benefit | Relative old benefit | Control current drift | Guard current drift | Relative current harm |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pair_rows:
        lines.append(
            f"| {row['seed']} | {row['stage']} | {row['guard_fire_count_h100']} | "
            f"{fmt(row['guard_fire_rate_h100'])} | {fmt(row['control_old_drift_h100'])} | "
            f"{fmt(row['guard_old_drift_h100'])} | {fmt(row['old_benefit_h100'])} | "
            f"{fmt(row['relative_old_benefit_h100'])} | "
            f"{fmt(row['control_current_drift_h100'])} | "
            f"{fmt(row['guard_current_drift_h100'])} | "
            f"{fmt(row['relative_current_harm_h100'])} |"
        )
    lines.extend(
        [
            "",
            f"Positive retention pairs: `{positive_h100}/4`; mean/median relative old benefit: "
            f"`{fmt(mean_old_h100)}` / `{fmt(statistics.median(relative_old_h100))}`; "
            f"mean/max relative current harm: `{fmt(mean_current_h100)}` / "
            f"`{fmt(max_current_h100)}`; pooled fire rate: `{fmt(pooled_fire_h100)}`.",
            "",
            "## H300 persistence",
            "",
            "| Seed | Stage | Fires | Fire rate | Old benefit | Relative old benefit | Current harm | Relative current harm |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pair_rows:
        lines.append(
            f"| {row['seed']} | {row['stage']} | {row['guard_fire_count_h300']} | "
            f"{fmt(row['guard_fire_rate_h300'])} | {fmt(row['old_benefit_h300'])} | "
            f"{fmt(row['relative_old_benefit_h300'])} | {fmt(row['current_harm_h300'])} | "
            f"{fmt(row['relative_current_harm_h300'])} |"
        )
    lines.extend(
        [
            "",
            f"Positive retention pairs: `{positive_h300}/4`; mean relative old benefit: "
            f"`{fmt(mean_old_h300)}`; mean relative current harm: "
            f"`{fmt(statistics.mean(relative_current_h300))}`; pooled fire rate: "
            f"`{fmt(pooled_fire_h300)}`.",
            "",
            "## Per-task and mechanism observations",
            "",
            f"Task-specific retention tradeoff: `{'OBSERVED' if task_flags else 'NONE'}`; "
            f"worst task-specific relative extra harm at H100: `{fmt(worst_task_harm)}`.",
            f"Candidate cosine over all instrumented steps: mean `{fmt(statistics.mean(candidate_cosines))}`, "
            f"median `{fmt(statistics.median(candidate_cosines))}`, q25 `{fmt(percentile(candidate_cosines, 0.25))}`, "
            f"q75 `{fmt(percentile(candidate_cosines, 0.75))}`.",
            f"Fired Guard-step cosine: mean `{fmt(statistics.mean(fired_cosines)) if fired_cosines else 'NA'}`, "
            f"median `{fmt(statistics.median(fired_cosines)) if fired_cosines else 'NA'}`. "
            f"Mean candidate first-order score: `{fmt(score_mean)}`; summed predicted first-order "
            f"score reduction over fired steps: `{fmt(predicted_reduction)}`.",
            f"Intervention wall time: Control `{fmt(control_runtime)}` seconds; Guard "
            f"`{fmt(guard_runtime)}` seconds. Both arms recomputed fresh old gradients.",
            "",
            "## Scientific conclusion",
            "",
        ]
    )
    if verdict.endswith("SUPPORTED") and "NOT_SUPPORTED" not in verdict:
        lines.append(
            "H15 is supported under the preregistered smoke criteria: scaling "
            "high-risk candidate displacement showed paired short-horizon retention "
            "benefit without material current-probe harm, and the mean benefit did "
            "not reverse at H300. The next action is Guard Threshold / Refresh "
            "Strategy Diagnosis; do not start formal guard training."
        )
    elif verdict.endswith("NOT_SUPPORTED"):
        lines.append(
            "H15 is not supported under the preregistered smoke criteria. The "
            "directional score remains predictive, but this fixed magnitude-control "
            "intervention lacks causal utility or causes material plasticity harm. "
            "Stop the scale/reject route; next examine a post-update directional "
            "monitor or BC-loss versus rollout-success alignment."
        )
    else:
        lines.append(
            "H15 is inconclusive under the preregistered smoke criteria. Follow the "
            "specific evidence-floor, persistence, or plasticity limitation shown "
            "above; do not tune the threshold or scale after observing this result."
        )
    lines.extend(
        [
            "",
            "## Scope limits",
            "",
            "This is a short paired BC-loss intervention smoke, not lifelong rollout "
            "performance evidence. Seed-specific LOSO thresholds are diagnostic "
            "calibration values, not a final deployable universal threshold. The "
            "intervention keeps AdamW moments on the full-gradient trajectory while "
            "scaling only parameter displacement; this is an experimental control "
            "semantics, not yet the final optimizer design. Fresh old-task gradients "
            "are recomputed every step, so runtime cost is intentionally not solved "
            "here. The old-task probe may not perfectly represent rollout retention. "
            "Ep95 remains a late-stage proxy.",
            "",
            "## Exact successful commands",
            "",
            "```bash",
            "CUDA_VISIBLE_DEVICES=1 bash scripts/run_candidate_directional_guard.sh smoke",
            "CUDA_VISIBLE_DEVICES=1 bash scripts/run_candidate_directional_guard.sh formal",
            "python -m pytest -q tests/test_candidate_directional_guard.py",
            "```",
        ]
    )
    (report_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] verdict={verdict} pairs=4 branch_parity={branch_parity}")


if __name__ == "__main__":
    main()
