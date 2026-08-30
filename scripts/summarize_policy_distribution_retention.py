#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import subprocess
from collections import defaultdict
from pathlib import Path

from libero.lifelong.analysis.bc_rollout_retention_alignment import spearman
from libero.lifelong.analysis.policy_distribution_retention import (
    BC_CONTROLS,
    PRIMARY_SIGNALS,
    decide_flags,
    decide_h19,
    matched_signal_contrasts,
    rank_primary_signals,
    signal_alignment_metrics,
    summarize_groups,
)


STARTING_SHA = "e8093bfb8c7e967d08a2346b0a70a7c1cf85e42e"
SECONDARY_SIGNALS = (
    "rollout_symkl_global_mean",
    "rollout_mean_l2_global_mean",
    "rollout_mode_switch_rate",
    "full_gmm_fwdkl_global_mean",
    "full_gmm_fwdkl_global_q90",
    "rollout_kl_phase_mean",
    "rollout_w2_phase_mean",
)
FAMILY = {
    **{name: "primary_policy" for name in PRIMARY_SIGNALS},
    **{name: "bc_control" for name in BC_CONTROLS},
    **{name: "secondary_policy" for name in SECONDARY_SIGNALS},
}


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "" if isinstance(value, float) and not math.isfinite(value) else value
                for key, value in row.items()
            })


def baseline_metrics(path):
    row = read_rows(path)[0]
    return {
        "rows": int(row["matched_rows"]),
        "non_tie_rows": int(row["rollout_non_tie_rows"]),
        "pearson": float(row["pearson"]),
        "spearman": float(row["spearman"]),
        "roc_auc": float(row["roc_auc_non_tie"]),
        "average_precision": float(row["average_precision_non_tie"]),
        "sign_concordance": float(row["sign_concordance_non_tie"]),
        "dangerous_rate": float(row["dangerous_BW_rate"]),
        "BB": int(row["BB"]), "BW": int(row["BW"]),
        "WB": int(row["WB"]), "WW": int(row["WW"]),
        "ties": int(row["ties"]),
    }


def summary_row(signal, metrics, baseline):
    return {
        "signal": signal,
        "signal_family": FAMILY[signal],
        **metrics,
        "delta_spearman": float(metrics["spearman"]) - baseline["spearman"],
        "delta_auc": float(metrics["roc_auc"]) - baseline["roc_auc"],
        "delta_ap": float(metrics["average_precision"]) - baseline["average_precision"],
        "delta_concordance": float(metrics["sign_concordance"])
        - baseline["sign_concordance"],
        "delta_dangerous_rate": float(metrics["dangerous_rate"])
        - baseline["dangerous_rate"],
    }


def load_selected(path):
    result = {}
    for row in read_rows(path):
        result[(row["method"], int(row["seed"]), int(row["stage"]))] = int(
            row["selected_epoch"]
        )
    return result


def row_index(rows):
    return {
        (
            row["method"], int(row["seed"]), int(row["stage"]),
            int(row["epoch"]), int(row["task"]),
        ): row
        for row in rows
    }


def best_longitudinal(grid, rollout, signal):
    signals = row_index(grid)
    successes = row_index(
        row for row in rollout if int(row["task"]) < int(row["stage"])
    )
    relations = sorted({(key[0], key[1], key[2], key[4]) for key in signals})
    rows = []
    for method, seed, stage, task in relations:
        epochs = (70, 75, 80, 85, 90, 95)
        values = [float(signals[(method, seed, stage, epoch, task)][signal]) for epoch in epochs]
        outcomes = [float(successes[(method, seed, stage, epoch, task)]["success_pct"]) for epoch in epochs]
        rows.append({
            "signal": signal, "method": method, "seed": seed,
            "stage": stage, "task": task, "points": 6,
            "success_unique_values": len(set(outcomes)),
            "spearman_neg_signal_vs_success": spearman(
                [-value for value in values], outcomes
            ),
            "signal_ep70": values[0], "signal_ep95": values[-1],
            "success_ep70": outcomes[0], "success_ep95": outcomes[-1],
        })
    if len(rows) != 36:
        raise ValueError(f"expected 36 longitudinal relations, found {len(rows)}")
    return rows


def best_nbt(grid, rollout, selected, signal):
    signals = row_index(grid)
    successes = row_index(rollout)
    method_rows = {}
    for method in ("temporal_v1", "zero_guard"):
        for seed in (42, 43, 44):
            for task in (6, 7, 8):
                acquisition_epoch = selected[(method, seed, task)]
                acquisition_success = float(
                    successes[(method, seed, task, acquisition_epoch, task)]["success_pct"]
                )
                later_stages = list(range(task + 1, 10))
                later_signal = [
                    float(signals[(method, seed, stage, selected[(method, seed, stage)], task)][signal])
                    for stage in later_stages
                ]
                later_success = [
                    float(successes[(method, seed, stage, selected[(method, seed, stage)], task)]["success_pct"])
                    for stage in later_stages
                ]
                method_rows[(method, seed, task)] = {
                    "mean_later_signal": statistics.mean(later_signal),
                    "acquisition_success_pct": acquisition_success,
                    "mean_later_success_pct": statistics.mean(later_success),
                    "rollout_forgetting_pp": acquisition_success
                    - statistics.mean(later_success),
                }
    rows = []
    for seed in (42, 43, 44):
        for task in (6, 7, 8):
            v1 = method_rows[("temporal_v1", seed, task)]
            guard = method_rows[("zero_guard", seed, task)]
            signal_advantage = v1["mean_later_signal"] - guard["mean_later_signal"]
            rollout_advantage = v1["rollout_forgetting_pp"] - guard["rollout_forgetting_pp"]
            rows.append({
                "signal": signal, "seed": seed, "task": task,
                "v1_mean_later_divergence": v1["mean_later_signal"],
                "guard_mean_later_divergence": guard["mean_later_signal"],
                "nbt_signal_advantage": signal_advantage,
                "v1_rollout_forgetting_pp": v1["rollout_forgetting_pp"],
                "guard_rollout_forgetting_pp": guard["rollout_forgetting_pp"],
                "rollout_retention_advantage_pp": rollout_advantage,
                "rollout_non_tie": int(rollout_advantage != 0),
                "sign_concordant": "" if rollout_advantage == 0 else int(
                    (signal_advantage > 0 and rollout_advantage > 0)
                    or (signal_advantage < 0 and rollout_advantage < 0)
                ),
                "dangerous_mismatch": int(signal_advantage > 0 and rollout_advantage < 0),
            })
    return rows


def fmt(value):
    value = float(value)
    return "NA" if not math.isfinite(value) else f"{value:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("reports/success_critical_retention_signal"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/success_critical_retention_signal"),
    )
    parser.add_argument("--push-status", default="PUSH_DEFERRED_TLS")
    args = parser.parse_args()
    root = args.root.resolve()
    input_dir = args.input_dir if args.input_dir.is_absolute() else root / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = read_rows(input_dir / "signal_checkpoint_grid.csv")
    rollout = read_rows(
        root / "reports/bc_rollout_retention_alignment/rollout_checkpoint_grid.csv"
    )
    baseline = baseline_metrics(
        root / "reports/bc_rollout_retention_alignment/alignment_summary.csv"
    )
    if len(grid) != 216:
        raise ValueError(f"expected 216 signal rows, found {len(grid)}")
    if any(not math.isfinite(float(row[name])) for row in grid for name in PRIMARY_SIGNALS):
        raise ValueError("non-finite primary signal")
    all_contrasts = []
    summaries = []
    seed_summaries = []
    stage_summaries = []
    for signal in (*PRIMARY_SIGNALS, *BC_CONTROLS, *SECONDARY_SIGNALS):
        contrasts = matched_signal_contrasts(grid, rollout, signal, FAMILY[signal])
        if len(contrasts) != 108:
            raise ValueError(f"signal {signal} has {len(contrasts)} contrasts")
        all_contrasts.extend(contrasts)
        metrics = signal_alignment_metrics(contrasts)
        if signal == "bc_global_mean":
            metrics = dict(baseline)
        summaries.append(summary_row(signal, metrics, baseline))
        for row in summarize_groups(contrasts, "seed"):
            seed_summaries.append({"signal": signal, "signal_family": FAMILY[signal], **row})
        for row in summarize_groups(contrasts, "stage"):
            stage_summaries.append({"signal": signal, "signal_family": FAMILY[signal], **row})
    ranking = rank_primary_signals(summaries)
    for index, row in enumerate(ranking, 1):
        row["rank"] = index
    best = ranking[0]
    phase_rows = read_rows(input_dir / "phase_probe_manifest.csv")
    coverage_ok = (
        len(phase_rows) == 1440
        and len(grid) == 216
        and all(int(row["global_probe_count"]) == 256 for row in grid)
        and all(int(row["phase_probe_count"]) == 160 for row in grid)
        and baseline["non_tie_rows"] == 87
    )
    verdict = decide_h19(summaries, baseline, seed_summaries, coverage_ok)
    flags = decide_flags(summaries, best)
    best_contrasts = [row for row in all_contrasts if row["signal"] == best["signal"]]
    longitudinal = best_longitudinal(grid, rollout, best["signal"])
    selected = load_selected(
        root / "reports/bc_rollout_retention_alignment/selected_epochs.csv"
    )
    nbt = best_nbt(grid, rollout, selected, best["signal"])
    nbt_non_tie = [row for row in nbt if int(row["rollout_non_tie"])]
    nbt_concordance = sum(int(row["sign_concordant"]) for row in nbt_non_tie) / len(nbt_non_tie)
    valid_long = [
        float(row["spearman_neg_signal_vs_success"]) for row in longitudinal
        if math.isfinite(float(row["spearman_neg_signal_vs_success"]))
    ]
    write_rows(output_dir / "signal_matched_contrasts.csv", all_contrasts)
    write_rows(output_dir / "signal_alignment_summary.csv", summaries)
    write_rows(output_dir / "signal_seed_summary.csv", seed_summaries)
    write_rows(output_dir / "signal_stage_summary.csv", stage_summaries)
    write_rows(output_dir / "signal_gain_vs_bc.csv", summaries)
    write_rows(output_dir / "bc_control_summary.csv", [row for row in summaries if row["signal"] in BC_CONTROLS])
    write_rows(output_dir / "primary_signal_ranking.csv", ranking)
    write_rows(output_dir / "best_signal_checkpoint_contrast.csv", best_contrasts)
    write_rows(output_dir / "best_signal_longitudinal.csv", longitudinal)
    write_rows(output_dir / "best_signal_nbt_alignment.csv", nbt)
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    next_action = {
        "SUCCESS_CRITICAL_BEHAVIORAL_RETENTION_SIGNAL_SUPPORTED": (
            "Deployment-Anchored Functional Guard Smoke"
            if "POLICY_DISTRIBUTION_ADDS_VALUE" in flags
            else "Success-Critical Local BC Guard Smoke"
        ),
        "SUCCESS_CRITICAL_BEHAVIORAL_RETENTION_SIGNAL_NOT_SUPPORTED": (
            "On-Policy Deployment-State Retention Diagnosis"
        ),
        "SUCCESS_CRITICAL_BEHAVIORAL_RETENTION_SIGNAL_INCONCLUSIVE": (
            "50-trial targeted re-evaluation of the best-signal disagreement subset"
        ),
    }[verdict]
    report = f"""# Success-Critical Behavioral Retention Signal Diagnosis

## Identity

- Verdict: `{verdict}`
- Flags: `{', '.join(flags)}`
- Branch: `{branch}`
- Starting SHA: `{STARTING_SHA}`
- Analysis source SHA: `{source_sha}`
- Final SHA: recorded by the result commit and final handoff
- Push status: `{args.push_status}`

## Protocol and coverage

No retraining, `optimizer.step`, or new rollout was performed. Existing
checkpoints and the locked 4320-trial rollout grid were reused.

- Phase manifest: {len(phase_rows)} rows (32 per seed/task/bin)
- Signal checkpoint grid: {len(grid)} rows
- Matched rows per signal: 108
- Rollout non-ties: {baseline['non_tie_rows']}
- Coverage floor: {'met' if coverage_ok else 'not met'}

## Locked global-mean BC baseline

| Spearman | AUC | AP | Concordance | Dangerous rate |
|---:|---:|---:|---:|---:|
| {fmt(baseline['spearman'])} | {fmt(baseline['roc_auc'])} | {fmt(baseline['average_precision'])} | {fmt(baseline['sign_concordance'])} | {fmt(baseline['dangerous_rate'])} |

## P1-P8 primary signals

| Rank | Signal | Spearman | AUC | AP | Concordance | Dangerous | Delta AUC |
|---:|---|---:|---:|---:|---:|---:|---:|
"""
    for row in ranking:
        report += (
            f"| {row['rank']} | {row['signal']} | {fmt(row['spearman'])} | "
            f"{fmt(row['roc_auc'])} | {fmt(row['average_precision'])} | "
            f"{fmt(row['sign_concordance'])} | {fmt(row['dangerous_rate'])} | "
            f"{fmt(row['delta_auc'])} |\n"
        )
    report += "\n## C0-C4 BC controls\n\n| Signal | Spearman | AUC | Concordance | Dangerous |\n|---|---:|---:|---:|---:|\n"
    for row in summaries:
        if row["signal"] in BC_CONTROLS:
            report += (
                f"| {row['signal']} | {fmt(row['spearman'])} | {fmt(row['roc_auc'])} | "
                f"{fmt(row['sign_concordance'])} | {fmt(row['dangerous_rate'])} |\n"
            )
    best_seed = [row for row in seed_summaries if row["signal"] == best["signal"]]
    best_stage = [row for row in stage_summaries if row["signal"] == best["signal"]]
    full = next(row for row in summaries if row["signal"] == "full_gmm_fwdkl_global_mean")
    report += f"""

## Best primary

- Signal: `{best['signal']}`
- Spearman / AUC / AP: {fmt(best['spearman'])} / {fmt(best['roc_auc'])} / {fmt(best['average_precision'])}
- Concordance / dangerous: {fmt(best['sign_concordance'])} / {fmt(best['dangerous_rate'])}
- Delta AUC vs locked BC: {fmt(best['delta_auc'])}
- Seed AUCs: {', '.join(str(row['seed']) + '=' + fmt(row['roc_auc']) for row in best_seed)}
- Stage AUCs: {', '.join(str(row['stage']) + '=' + fmt(row['roc_auc']) for row in best_stage)}

## Secondary

- Full-GMM forward-KL AUC: {fmt(full['roc_auc'])}
- Longitudinal relations: {len(longitudinal)}; median Spearman: {fmt(statistics.median(valid_long))}
- NBT contrasts: {len(nbt)}; non-tie concordance: {fmt(nbt_concordance)}

## Observed

The fixed expert-state probes provide the measurements recorded in the CSV
artifacts. H19 is classified as `{verdict}` under the preregistered thresholds.

## Inferred

The ranking distinguishes whether localization, rollout-mode policy drift, or
neither materially improves on global mean BC alignment. Flags: `{', '.join(flags)}`.

## Not established

These signals are still evaluated on stored expert-feature states, not on
states generated by the current policy during deployment.

A strong offline policy-distribution signal does not yet prove that
intervening on it improves FWT/NBT/AUC.

Normalized progress bins are not semantic reach/grasp/contact/release labels.

Full-GMM KL is Monte-Carlo estimated; selected rollout-mode KL/W2 are analytic.

Analysis is limited to LIBERO_OBJECT and existing ep70-95 checkpoints.

No new training or rollout evaluation was performed.

## Next action

`{next_action}`
"""
    (output_dir / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(
        f"[success-critical] verdict={verdict} flags={flags} "
        f"best={best['signal']} auc={best['roc_auc']:.6f} coverage={coverage_ok}"
    )


if __name__ == "__main__":
    main()
