#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from libero.lifelong.analysis.bc_rollout_retention_alignment import (
    METHODS,
    alignment_metrics,
    build_adjacent_transitions,
    build_longitudinal_relations,
    build_nbt_method_contrasts,
    build_nbt_surrogate_rows,
    build_unique_index,
    decide_flags,
    decide_verdict,
    matched_method_contrasts,
    parse_selected_epochs,
    summarize_by_seed,
    summarize_by_stage,
)


STARTING_SHA = "80dfb5a5b25cf2b7ed18ed530b8291703c8e3cdf"
RUN_SUFFIX = {
    "temporal_v1": "mlr_temporal_igfa",
    "zero_guard": "mlr_zero_directional_guard",
}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def clean(value):
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [{key: clean(value) for key, value in row.items()} for row in rows]
        )


def metric(value):
    return "NA" if not math.isfinite(float(value)) else f"{float(value):.6f}"


def longitudinal_summary(relations, transitions, method):
    selected = [row for row in relations if row["method"] == method]
    valid = [
        float(row["spearman"])
        for row in selected
        if math.isfinite(float(row["spearman"]))
    ]
    adjacent = [
        row
        for row in transitions
        if row["method"] == method and row["sign_concordant"] != ""
    ]
    return {
        "relations": len(selected),
        "valid_relations": len(valid),
        "median_spearman": statistics.median(valid) if valid else float("nan"),
        "adjacent_non_tie": len(adjacent),
        "adjacent_concordance": (
            sum(int(row["sign_concordant"]) for row in adjacent) / len(adjacent)
            if adjacent
            else float("nan")
        ),
        "bc_improve_rollout_worse": sum(
            float(row["bc_improvement"]) > 0
            and float(row["success_improvement_pp"]) < 0
            for row in transitions
            if row["method"] == method
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("reports/bc_rollout_retention_alignment"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/bc_rollout_retention_alignment"),
    )
    parser.add_argument("--push-status", default="PUSH_DEFERRED_TLS")
    args = parser.parse_args()
    root = args.root.resolve()
    input_dir = args.input_dir if args.input_dir.is_absolute() else root / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    bc_rows = read_rows(input_dir / "bc_loss_checkpoint_grid.csv")
    rollout_rows = read_rows(input_dir / "rollout_checkpoint_grid.csv")
    if len(bc_rows) != 324 or len(rollout_rows) != 324:
        raise ValueError(
            f"formal grids require 324 rows each, found BC={len(bc_rows)} "
            f"rollout={len(rollout_rows)}"
        )
    if set(build_unique_index(bc_rows)) != set(build_unique_index(rollout_rows)):
        raise ValueError("formal BC and rollout keys differ")
    if any(not math.isfinite(float(row["mean_bc_loss"])) for row in bc_rows):
        raise ValueError("non-finite BC grid value")
    if any(not 0.0 <= float(row["success_rate"]) <= 1.0 for row in rollout_rows):
        raise ValueError("rollout success outside [0,1]")

    contrasts = matched_method_contrasts(bc_rows, rollout_rows)
    summary = alignment_metrics(contrasts)
    seed_summary = summarize_by_seed(contrasts)
    stage_summary = summarize_by_stage(contrasts)
    longitudinal = build_longitudinal_relations(bc_rows, rollout_rows)
    transitions = build_adjacent_transitions(bc_rows, rollout_rows)

    selected_epochs = {}
    selected_rows = []
    for method in METHODS:
        suffix = RUN_SUFFIX[method]
        for seed in (42, 43, 44):
            path = root / "runs" / f"object_seed{seed}_{suffix}" / f"eval_seed{seed}.log"
            selected = parse_selected_epochs(path)
            selected_epochs[(method, seed)] = selected
            for stage, epoch in selected.items():
                selected_rows.append(
                    {
                        "method": method,
                        "seed": seed,
                        "stage": stage,
                        "selected_epoch": epoch,
                    }
                )
    nbt_rows = build_nbt_surrogate_rows(
        bc_rows, rollout_rows, selected_epochs
    )
    nbt_contrasts = build_nbt_method_contrasts(nbt_rows)

    evidence_ok = (
        len(contrasts) == 108
        and summary["rollout_non_tie_rows"] >= 36
        and all(
            int(row["matched_rows"]) == 36
            and int(row["rollout_non_tie_rows"]) >= 8
            for row in seed_summary
        )
    )
    verdict = decide_verdict(summary, seed_summary, evidence_ok=evidence_ok)
    flags = decide_flags(summary, seed_summary, longitudinal, nbt_contrasts)

    write_rows(output_dir / "matched_checkpoint_contrast.csv", contrasts)
    write_rows(output_dir / "alignment_summary.csv", [summary])
    write_rows(output_dir / "seed_alignment_summary.csv", seed_summary)
    write_rows(output_dir / "stage_alignment_summary.csv", stage_summary)
    write_rows(output_dir / "longitudinal_relation_summary.csv", longitudinal)
    write_rows(output_dir / "adjacent_transition_grid.csv", transitions)
    write_rows(output_dir / "selected_epochs.csv", selected_rows)
    write_rows(output_dir / "nbt_surrogate_alignment.csv", nbt_rows)
    write_rows(output_dir / "nbt_method_contrast.csv", nbt_contrasts)

    long_by_method = {
        method: longitudinal_summary(longitudinal, transitions, method)
        for method in METHODS
    }
    nbt_non_tie = [row for row in nbt_contrasts if int(row["rollout_non_tie"])]
    nbt_concordance = (
        sum(int(row["sign_concordant"]) for row in nbt_non_tie)
        / len(nbt_non_tie)
        if nbt_non_tie
        else float("nan")
    )
    nbt_dangerous = sum(int(row["dangerous_mismatch"]) for row in nbt_contrasts)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], text=True
    ).strip()
    source_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    report = f"""# BC-Loss–Rollout Retention Alignment Diagnosis

## Identity

- Verdict: `{verdict}`
- Flags: `{', '.join(flags) if flags else 'none'}`
- Branch: `{branch}`
- Starting SHA: `{STARTING_SHA}`
- Analysis source SHA: `{source_sha}`
- Final SHA: recorded by the result commit and final handoff
- Push status: `{args.push_status}`

## Protocol

No retraining was performed. The analysis compares Temporal V1 and Zero Guard
for seeds 42/43/44 at epochs 70,75,80,85,90,95. BC loss uses the frozen
256-record/task manifest with batch size 32. Rollout success uses 20 trials per
checkpoint-task and the existing formal LIBERO_OBJECT evaluation path.

## Primary alignment

| Metric | Value |
|---|---:|
| Matched rows | {summary['matched_rows']} |
| Rollout non-tie rows | {summary['rollout_non_tie_rows']} |
| Pearson | {metric(summary['pearson'])} |
| Spearman | {metric(summary['spearman'])} |
| ROC-AUC, non-tie | {metric(summary['roc_auc_non_tie'])} |
| Average precision, non-tie | {metric(summary['average_precision_non_tie'])} |
| Sign concordance, non-tie | {metric(summary['sign_concordance_non_tie'])} |
| BB / BW / WB / WW / ties | {summary['BB']} / {summary['BW']} / {summary['WB']} / {summary['WW']} / {summary['ties']} |
| Dangerous BW rate | {metric(summary['dangerous_BW_rate'])} |

## Per seed

| Seed | Spearman | AUC | Concordance | BW rate |
|---:|---:|---:|---:|---:|
"""
    for row in seed_summary:
        report += (
            f"| {row['seed']} | {metric(row['spearman'])} | "
            f"{metric(row['roc_auc_non_tie'])} | "
            f"{metric(row['sign_concordance_non_tie'])} | "
            f"{metric(row['dangerous_BW_rate'])} |\n"
        )
    report += "\n## Per stage\n\n| Stage | Spearman | AUC | Concordance | BW rate |\n|---:|---:|---:|---:|---:|\n"
    for row in stage_summary:
        report += (
            f"| {row['stage']} | {metric(row['spearman'])} | "
            f"{metric(row['roc_auc_non_tie'])} | "
            f"{metric(row['sign_concordance_non_tie'])} | "
            f"{metric(row['dangerous_BW_rate'])} |\n"
        )
    report += "\n## Longitudinal\n\n| Method | Valid relations | Median Spearman | Adjacent concordance | BC-improve / rollout-worse |\n|---|---:|---:|---:|---:|\n"
    for method in METHODS:
        row = long_by_method[method]
        report += (
            f"| {method} | {row['valid_relations']} | "
            f"{metric(row['median_spearman'])} | "
            f"{metric(row['adjacent_concordance'])} | "
            f"{row['bc_improve_rollout_worse']} |\n"
        )
    report += f"""

## NBT surrogate

The analysis contains {len(nbt_rows)} method-task rows and
{len(nbt_contrasts)} matched Guard-vs-V1 task contrasts. Non-tie sign
concordance is {metric(nbt_concordance)}; BC-better/rollout-worse occurs in
{nbt_dangerous} contrasts.

## Observed

The complete checkpoint grids, matched quadrants, longitudinal transitions,
selected epochs, and NBT analogues are retained in the CSV files in this
directory. The evidence floor was {'met' if evidence_ok else 'not met'}.

## Inferred

H18 is classified as `{verdict}` using the preregistered thresholds.
"""
    if verdict == "BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_SUPPORTED":
        report += (
            "Mean old-task BC loss is not sufficiently aligned with rollout "
            "retention to be the sole control surrogate for the current Guard.\n"
        )
    elif verdict == "BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_NOT_SUPPORTED":
        report += (
            "Mean old-task BC loss is strongly aligned with rollout retention; "
            "the Guard failure is therefore more consistent with cumulative "
            "intervention path distortion than surrogate mismatch.\n"
        )
    else:
        report += (
            "The available checkpoint grid does not decisively resolve the "
            "surrogate-mismatch hypothesis.\n"
        )
    report += """

## Not established

Weak alignment does not prove BC loss is useless for imitation learning.

This concerns mean BC loss on fixed probes, not all possible state-conditioned
or trajectory-conditioned behavioral losses.

20-trial rollout has 5-pp resolution and can create ties.

Correlation/mismatch does not identify a unique causal mechanism.

Analysis is limited to LIBERO_OBJECT and ep70-95 checkpoints.

No new training was performed.
"""
    (output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(
        f"[alignment] verdict={verdict} flags={flags} "
        f"matched={len(contrasts)} non_tie={summary['rollout_non_tie_rows']}"
    )


if __name__ == "__main__":
    main()
