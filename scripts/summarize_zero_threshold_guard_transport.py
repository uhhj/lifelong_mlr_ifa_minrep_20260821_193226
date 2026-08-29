#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import subprocess
from pathlib import Path

from libero.lifelong.analysis.candidate_directional_guard import paired_drift_metrics
from scripts.diagnose_online_first_order_margin import write_csv


SEEDS = (42, 43, 44)
STAGES = (8, 9)
STARTING_SHA = "d5d8759152563021483f570683aeaf7046a21201"

PAIR_FIELDS = [
    "seed", "stage", "threshold_mode", "threshold", "guard_scale",
    "start_old_loss", "start_current_loss", "guard_fire_count_h100",
    "guard_fire_rate_h100", "control_old_drift_h100", "guard_old_drift_h100",
    "old_benefit_h100", "relative_old_benefit_h100",
    "control_current_drift_h100", "guard_current_drift_h100",
    "current_harm_h100", "relative_current_harm_h100",
    "guard_fire_count_h300", "guard_fire_rate_h300",
    "control_old_drift_h300", "guard_old_drift_h300", "old_benefit_h300",
    "relative_old_benefit_h300", "control_current_drift_h300",
    "guard_current_drift_h300", "current_harm_h300",
    "relative_current_harm_h300", "control_would_fire_rate_h100",
    "control_would_fire_rate_h300",
]
SEED_FIELDS = [
    "seed", "positive_pairs_h100", "mean_relative_old_benefit_h100",
    "mean_relative_current_harm_h100", "mean_relative_old_benefit_h300",
    "mean_relative_current_harm_h300", "pooled_fire_rate_h100",
    "pooled_fire_rate_h300",
]
OVERALL_FIELDS = [
    "pairs", "positive_old_benefit_pairs_h100",
    "mean_relative_old_benefit_h100", "median_relative_old_benefit_h100",
    "positive_old_benefit_pairs_h300", "mean_relative_old_benefit_h300",
    "median_relative_old_benefit_h300", "mean_relative_current_harm_h100",
    "max_relative_current_harm_h100", "mean_relative_current_harm_h300",
    "max_relative_current_harm_h300", "pooled_fire_rate_h100",
    "pooled_fire_rate_h300", "seed42_mean_old_benefit_h100",
    "seed43_mean_old_benefit_h100", "seed44_mean_old_benefit_h100",
    "branch_start_parity", "all_metrics_finite", "verdict",
]


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentile(values, probability):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * float(probability)
    lower, upper = math.floor(index), math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def finite_rows(rows, fields):
    return all(
        row.get(field, "") != "" and math.isfinite(float(row[field]))
        for row in rows
        for field in fields
    )


def context_pair(seed, stage, steps, probes):
    lookup = {}
    for row in probes:
        key = (row["arm"], int(row["update"]))
        if key in lookup:
            raise ValueError(f"duplicate probe row {key}")
        lookup[key] = row
    required = {
        (arm, update) for arm in ("control", "guard") for update in (0, 100, 300)
    }
    if set(lookup) != required:
        raise ValueError(f"seed{seed} stage{stage} probe points differ")
    if any(row.get("threshold_mode") != "zero" for row in probes + steps):
        raise ValueError("formal transport rows must use threshold_mode=zero")
    if any(float(row["threshold"]) != 0.0 for row in probes + steps):
        raise ValueError("formal transport threshold must equal zero")
    control_start = lookup[("control", 0)]
    guard_start = lookup[("guard", 0)]
    old_parity = abs(
        float(control_start["old_mean_loss"]) - float(guard_start["old_mean_loss"])
    )
    current_parity = abs(
        float(control_start["current_loss"]) - float(guard_start["current_loss"])
    )
    start_old = float(control_start["old_mean_loss"])
    start_current = float(control_start["current_loss"])
    row = {
        "seed": int(seed),
        "stage": int(stage),
        "threshold_mode": "zero",
        "threshold": 0.0,
        "guard_scale": float(control_start["guard_scale"]),
        "start_old_loss": start_old,
        "start_current_loss": start_current,
    }
    arm_steps = {
        arm: [item for item in steps if item["arm"] == arm]
        for arm in ("control", "guard")
    }
    if any(len(items) != 300 for items in arm_steps.values()):
        raise ValueError(f"seed{seed} stage{stage} does not have 300 steps per arm")
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
        guard_block = [
            item for item in arm_steps["guard"] if int(item["update"]) <= horizon
        ]
        control_block = [
            item for item in arm_steps["control"] if int(item["update"]) <= horizon
        ]
        fire_count = sum(int(item["guard_fire"]) for item in guard_block)
        row.update(
            {
                f"guard_fire_count_{suffix}": fire_count,
                f"guard_fire_rate_{suffix}": fire_count / horizon,
                f"control_old_drift_{suffix}": effect["control_old_drift"],
                f"guard_old_drift_{suffix}": effect["guard_old_drift"],
                f"old_benefit_{suffix}": effect["old_benefit"],
                f"relative_old_benefit_{suffix}": effect["relative_old_benefit"],
                f"control_current_drift_{suffix}": effect["control_current_drift"],
                f"guard_current_drift_{suffix}": effect["guard_current_drift"],
                f"current_harm_{suffix}": effect["current_harm"],
                f"relative_current_harm_{suffix}": effect["relative_current_harm"],
                f"control_would_fire_rate_{suffix}": sum(
                    int(item["would_fire"]) for item in control_block
                ) / horizon,
            }
        )
    return row, old_parity, current_parity


def build_seed_summary(pair_rows):
    output = []
    for seed in SEEDS:
        rows = [row for row in pair_rows if int(row["seed"]) == seed]
        if len(rows) != 2:
            raise ValueError(f"seed {seed} does not have two stage pairs")
        output.append(
            {
                "seed": seed,
                "positive_pairs_h100": sum(
                    float(row["old_benefit_h100"]) > 0.0 for row in rows
                ),
                "mean_relative_old_benefit_h100": statistics.mean(
                    float(row["relative_old_benefit_h100"]) for row in rows
                ),
                "mean_relative_current_harm_h100": statistics.mean(
                    float(row["relative_current_harm_h100"]) for row in rows
                ),
                "mean_relative_old_benefit_h300": statistics.mean(
                    float(row["relative_old_benefit_h300"]) for row in rows
                ),
                "mean_relative_current_harm_h300": statistics.mean(
                    float(row["relative_current_harm_h300"]) for row in rows
                ),
                "pooled_fire_rate_h100": sum(
                    int(row["guard_fire_count_h100"]) for row in rows
                ) / 200,
                "pooled_fire_rate_h300": sum(
                    int(row["guard_fire_count_h300"]) for row in rows
                ) / 600,
            }
        )
    return output


def evaluate_transport(pair_rows, branch_parity=True, all_finite=True):
    seed_rows = build_seed_summary(pair_rows)
    positive_h100 = sum(float(row["old_benefit_h100"]) > 0.0 for row in pair_rows)
    positive_h300 = sum(float(row["old_benefit_h300"]) > 0.0 for row in pair_rows)
    old_h100 = [float(row["relative_old_benefit_h100"]) for row in pair_rows]
    old_h300 = [float(row["relative_old_benefit_h300"]) for row in pair_rows]
    current_h100 = [float(row["relative_current_harm_h100"]) for row in pair_rows]
    current_h300 = [float(row["relative_current_harm_h300"]) for row in pair_rows]
    pooled_h100 = sum(int(row["guard_fire_count_h100"]) for row in pair_rows) / 600
    pooled_h300 = sum(int(row["guard_fire_count_h300"]) for row in pair_rows) / 1800
    seed_by_id = {int(row["seed"]): row for row in seed_rows}
    evidence_floor = (
        len(pair_rows) == 6
        and bool(branch_parity)
        and bool(all_finite)
        and all(int(row["guard_fire_count_h100"]) >= 5 for row in pair_rows)
        and 0.05 <= pooled_h100 <= 0.80
    )
    mean_old_h100 = statistics.mean(old_h100)
    mean_old_h300 = statistics.mean(old_h300)
    mean_current_h100 = statistics.mean(current_h100)
    max_current_h100 = max(current_h100)
    supported = (
        positive_h100 >= 5
        and mean_old_h100 >= 0.001
        and all(float(row["mean_relative_old_benefit_h100"]) > 0.0 for row in seed_rows)
        and int(seed_by_id[44]["positive_pairs_h100"]) >= 1
        and float(seed_by_id[44]["mean_relative_old_benefit_h100"]) > 0.0
        and positive_h300 >= 4
        and mean_old_h300 >= 0.0
        and mean_current_h100 <= 0.005
        and max_current_h100 <= 0.02
    )
    retention_failure = (
        (positive_h100 <= 2 and mean_old_h100 <= 0.0)
        or mean_old_h100 <= -0.001
    )
    seed44_failure = (
        int(seed_by_id[44]["positive_pairs_h100"]) == 0
        and float(seed_by_id[44]["mean_relative_old_benefit_h100"]) <= 0.0
    )
    plasticity_failure = (
        mean_current_h100 > 0.02
        or sum(value > 0.02 for value in current_h100) >= 3
    )
    if not evidence_floor:
        verdict = "ZERO_THRESHOLD_DIRECTIONAL_GUARD_TRANSPORT_INCONCLUSIVE"
    elif retention_failure or seed44_failure or plasticity_failure:
        verdict = "ZERO_THRESHOLD_DIRECTIONAL_GUARD_TRANSPORT_NOT_SUPPORTED"
    elif supported:
        verdict = "ZERO_THRESHOLD_DIRECTIONAL_GUARD_TRANSPORT_SUPPORTED"
    else:
        verdict = "ZERO_THRESHOLD_DIRECTIONAL_GUARD_TRANSPORT_INCONCLUSIVE"
    return {
        "verdict": verdict,
        "evidence_floor": evidence_floor,
        "seed_rows": seed_rows,
        "positive_h100": positive_h100,
        "positive_h300": positive_h300,
        "mean_old_h100": mean_old_h100,
        "median_old_h100": statistics.median(old_h100),
        "mean_old_h300": mean_old_h300,
        "median_old_h300": statistics.median(old_h300),
        "mean_current_h100": mean_current_h100,
        "max_current_h100": max_current_h100,
        "mean_current_h300": statistics.mean(current_h300),
        "max_current_h300": max(current_h300),
        "pooled_h100": pooled_h100,
        "pooled_h300": pooled_h300,
    }


def task_tradeoff(old_rows):
    lookup = {
        (
            int(row["seed"]), int(row["stage"]), row["arm"],
            int(row["update"]), int(row["old_task"]),
        ): float(row["loss"])
        for row in old_rows
    }
    worst = -float("inf")
    flagged = []
    for seed in SEEDS:
        for stage in STAGES:
            for task in range(6, stage):
                control_start = lookup[(seed, stage, "control", 0, task)]
                guard_start = lookup[(seed, stage, "guard", 0, task)]
                if abs(control_start - guard_start) > 1e-6:
                    raise ValueError("per-task branch start parity failed")
                control_drift = lookup[(seed, stage, "control", 100, task)] - control_start
                guard_drift = lookup[(seed, stage, "guard", 100, task)] - guard_start
                harm = (guard_drift - control_drift) / (abs(control_start) + 1e-12)
                worst = max(worst, harm)
                if harm > 0.02:
                    flagged.append((seed, stage, task, harm))
    return flagged, worst


def fmt(value, digits=8):
    return f"{float(value):.{digits}g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("reports/zero_threshold_guard_transport"),
    )
    args = parser.parse_args()
    report_dir = args.report_dir.resolve()
    raw_dir = report_dir / "raw"
    all_steps, all_probes, all_old = [], [], []
    pairs, parity = [], []
    for seed in SEEDS:
        for stage in STAGES:
            prefix = f"seed{seed}_stage{stage}"
            steps = read_csv(raw_dir / f"{prefix}_steps.csv")
            probes = read_csv(raw_dir / f"{prefix}_probe.csv")
            old = read_csv(raw_dir / f"{prefix}_old_task_probe.csv")
            pair, old_diff, current_diff = context_pair(seed, stage, steps, probes)
            pairs.append(pair)
            parity.extend((old_diff, current_diff))
            all_steps.extend(steps)
            all_probes.extend(probes)
            all_old.extend(old)
    branch_parity = max(parity) <= 1e-6
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
        and finite_rows(all_old, ["loss"])
    )
    result = evaluate_transport(pairs, branch_parity, all_finite)
    verdict = result["verdict"]
    seed_rows = result["seed_rows"]
    seed_by_id = {int(row["seed"]): row for row in seed_rows}
    task_flags, worst_task_harm = task_tradeoff(all_old)
    flags = []
    if task_flags:
        flags.append("TASK_SPECIFIC_RETENTION_TRADEOFF_OBSERVED")
    if result["pooled_h100"] > 0.60:
        flags.append("ZERO_THRESHOLD_HIGH_INTERVENTION_RATE")
    if result["mean_old_h100"] > 0.0 and result["mean_old_h300"] < 0.0:
        flags.append("H300_RETENTION_REVERSAL_OBSERVED")
    overall = {
        "pairs": 6,
        "positive_old_benefit_pairs_h100": result["positive_h100"],
        "mean_relative_old_benefit_h100": result["mean_old_h100"],
        "median_relative_old_benefit_h100": result["median_old_h100"],
        "positive_old_benefit_pairs_h300": result["positive_h300"],
        "mean_relative_old_benefit_h300": result["mean_old_h300"],
        "median_relative_old_benefit_h300": result["median_old_h300"],
        "mean_relative_current_harm_h100": result["mean_current_h100"],
        "max_relative_current_harm_h100": result["max_current_h100"],
        "mean_relative_current_harm_h300": result["mean_current_h300"],
        "max_relative_current_harm_h300": result["max_current_h300"],
        "pooled_fire_rate_h100": result["pooled_h100"],
        "pooled_fire_rate_h300": result["pooled_h300"],
        "seed42_mean_old_benefit_h100": seed_by_id[42]["mean_relative_old_benefit_h100"],
        "seed43_mean_old_benefit_h100": seed_by_id[43]["mean_relative_old_benefit_h100"],
        "seed44_mean_old_benefit_h100": seed_by_id[44]["mean_relative_old_benefit_h100"],
        "branch_start_parity": "PASS" if branch_parity else "FAIL",
        "all_metrics_finite": int(all_finite),
        "verdict": verdict,
    }
    write_csv(report_dir / "step_raw.csv", list(all_steps[0]), all_steps)
    write_csv(report_dir / "probe_curve.csv", list(all_probes[0]), all_probes)
    write_csv(report_dir / "old_task_probe_curve.csv", list(all_old[0]), all_old)
    write_csv(report_dir / "pair_summary.csv", PAIR_FIELDS, pairs)
    write_csv(report_dir / "seed_summary.csv", SEED_FIELDS, seed_rows)
    write_csv(report_dir / "overall_summary.csv", OVERALL_FIELDS, [overall])

    cosines = [float(row["candidate_step_cosine"]) for row in all_steps]
    guard_steps = [row for row in all_steps if row["arm"] == "guard"]
    fired = [row for row in guard_steps if int(row["guard_fire"]) == 1]
    fired_cosines = [float(row["candidate_step_cosine"]) for row in fired]
    predicted_reduction = sum(
        (1.0 - float(row["guard_scale"])) * float(row["candidate_step_score"])
        for row in fired
    )
    runtimes = {}
    for row in all_steps:
        runtimes[(int(row["seed"]), int(row["stage"]), row["arm"])] = float(
            row["arm_wall_seconds"]
        )
    control_seconds = sum(value for key, value in runtimes.items() if key[2] == "control")
    guard_seconds = sum(value for key, value in runtimes.items() if key[2] == "guard")
    root = report_dir.parents[1]
    code_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    lines = [
        "# Zero-Threshold Candidate Directional Guard Transport Diagnosis",
        "",
        f"Verdict: `{verdict}`",
        f"Flags: `{', '.join(flags) if flags else 'NONE'}`",
        "Branch: `research/zero-threshold-directional-guard-transport`",
        f"Starting SHA: `{STARTING_SHA}`",
        f"Analysis code SHA at report generation: `{code_sha}`",
        "Final SHA: branch HEAD reported in the final handoff after committing this report.",
        "",
        "## Protocol and threshold provenance",
        "",
        "Temporal V1; seeds 42/43/44; stages 8/9; previous-stage ep95 entry "
        "proxy; 10 common warmup epochs; exact paired Control/Guard restore; "
        "300 intervention updates per arm; H100 primary and H300 secondary; "
        "guard scale 0.5; old score probe 64/task with a fresh gradient every "
        "step; old label probe 256/task; current probe 256.",
        "",
        "Threshold mode is `zero` and threshold is exactly `0.0`. No "
        "future-label-derived threshold artifact was read for guard decision.",
        f"Branch start parity: `{'PASS' if branch_parity else 'FAIL'}` "
        f"(maximum absolute probe difference `{fmt(max(parity))}`). All primary "
        f"metrics finite: `{all_finite}`.",
        "",
        "## H100 paired transport",
        "",
        "| Seed | Stage | Fire rate | Control old drift | Guard old drift | Rel. old benefit | Rel. current harm |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in pairs:
        lines.append(
            f"| {row['seed']} | {row['stage']} | {fmt(row['guard_fire_rate_h100'])} | "
            f"{fmt(row['control_old_drift_h100'])} | {fmt(row['guard_old_drift_h100'])} | "
            f"{fmt(row['relative_old_benefit_h100'])} | "
            f"{fmt(row['relative_current_harm_h100'])} |"
        )
    lines.extend(
        [
            "",
            f"Positive pairs: `{result['positive_h100']}/6`; mean/median relative "
            f"old benefit: `{fmt(result['mean_old_h100'])}` / "
            f"`{fmt(result['median_old_h100'])}`; mean/max relative current harm: "
            f"`{fmt(result['mean_current_h100'])}` / "
            f"`{fmt(result['max_current_h100'])}`; pooled fire rate: "
            f"`{fmt(result['pooled_h100'])}`.",
            "",
            "## Per-seed transport",
            "",
            "| Seed | Positive pairs | Mean H100 rel. old benefit | Mean rel. current harm | Fire rate |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in seed_rows:
        lines.append(
            f"| {row['seed']} | {row['positive_pairs_h100']}/2 | "
            f"{fmt(row['mean_relative_old_benefit_h100'])} | "
            f"{fmt(row['mean_relative_current_harm_h100'])} | "
            f"{fmt(row['pooled_fire_rate_h100'])} |"
        )
    lines.extend(
        [
            "",
            "## H300 persistence",
            "",
            "| Seed | Stage | Fire rate | Control old drift | Guard old drift | Rel. old benefit | Rel. current harm |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in pairs:
        lines.append(
            f"| {row['seed']} | {row['stage']} | {fmt(row['guard_fire_rate_h300'])} | "
            f"{fmt(row['control_old_drift_h300'])} | {fmt(row['guard_old_drift_h300'])} | "
            f"{fmt(row['relative_old_benefit_h300'])} | "
            f"{fmt(row['relative_current_harm_h300'])} |"
        )
    lines.extend(
        [
            "",
            f"Positive pairs: `{result['positive_h300']}/6`; mean relative old "
            f"benefit: `{fmt(result['mean_old_h300'])}`; mean relative current "
            f"harm: `{fmt(result['mean_current_h300'])}`; pooled fire rate: "
            f"`{fmt(result['pooled_h300'])}`.",
            "",
            "## Mechanism, runtime, and per-task checks",
            "",
            f"Candidate cosine mean/median/q25/q75: `{fmt(statistics.mean(cosines))}` / "
            f"`{fmt(statistics.median(cosines))}` / `{fmt(percentile(cosines, 0.25))}` / "
            f"`{fmt(percentile(cosines, 0.75))}`. Fired cosine mean/median/q25/q75: "
            f"`{fmt(statistics.mean(fired_cosines))}` / "
            f"`{fmt(statistics.median(fired_cosines))}` / "
            f"`{fmt(percentile(fired_cosines, 0.25))}` / "
            f"`{fmt(percentile(fired_cosines, 0.75))}`.",
            f"Summed predicted first-order score reduction: `{fmt(predicted_reduction)}`. "
            f"Control/Guard intervention wall time: `{fmt(control_seconds)}` / "
            f"`{fmt(guard_seconds)}` seconds; mean sec/update: "
            f"`{fmt(control_seconds / 1800)}` / `{fmt(guard_seconds / 1800)}`.",
            f"Task-specific retention tradeoff: `{'OBSERVED' if task_flags else 'NONE'}`; "
            f"worst relative task harm: `{fmt(worst_task_harm)}`.",
            "",
            "## Scientific conclusion",
            "",
        ]
    )
    if verdict == "ZERO_THRESHOLD_DIRECTIONAL_GUARD_TRANSPORT_SUPPORTED":
        lines.append(
            "H16 is supported under the preregistered criteria. The label-free "
            "candidate-cosine > 0 boundary preserved paired causal retention benefit "
            "across all seeds, including seed44, without material H100 current-probe "
            "harm and without mean H300 reversal. Threshold calibration is closed for "
            "this mechanism. Next run Guard Gradient Refresh Feasibility Diagnosis, "
            "comparing fresh gradients with refresh every 10 updates; do not reduce "
            "probe size or start formal guard training."
        )
    elif verdict == "ZERO_THRESHOLD_DIRECTIONAL_GUARD_TRANSPORT_NOT_SUPPORTED":
        lines.append(
            "H16 is not supported under the preregistered criteria. Tau=0 did not "
            "transport across seeds or caused material plasticity harm. The next step "
            "is label-free online distributional calibration or threshold selectivity "
            "diagnosis, not a lambda sweep."
        )
    else:
        lines.append(
            "H16 is inconclusive under the preregistered criteria. Follow the observed "
            "dose, effect-size, seed44, or H300 limitation without tuning tau after "
            "seeing these results."
        )
    lines.extend(
        [
            "",
            "## Not established",
            "",
            "This remains a short paired BC-loss intervention diagnosis, not formal "
            "lifelong rollout evidence. Tau=0 removes future-label threshold "
            "calibration, but fresh old-task gradient estimation remains expensive. "
            "The fixed score probes may not perfectly represent rollout retention. "
            "The intervention full-commits AdamW moments while scaling only the "
            "parameter endpoint. H300 is a medium-horizon BC-loss probe, not proof of "
            "long-term continual-learning benefit. Ep95 remains a late-stage proxy.",
            "",
            "## Exact successful commands",
            "",
            "```bash",
            "python -m pytest -q tests/test_zero_threshold_guard_transport.py tests/test_candidate_directional_guard.py",
            "CUDA_VISIBLE_DEVICES=1 bash scripts/run_zero_threshold_guard_transport.sh smoke",
            "CUDA_VISIBLE_DEVICES=1 bash scripts/run_zero_threshold_guard_transport.sh formal",
            "```",
        ]
    )
    (report_dir / "FINAL_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[summary] verdict={verdict} pairs=6 branch_parity={branch_parity} "
        f"fire_h100={result['pooled_h100']:.6g}"
    )


if __name__ == "__main__":
    main()
