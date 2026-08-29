#!/usr/bin/env python3
import argparse
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero.lifelong.analysis.optimizer_trajectory_signal import (
    FORMAL_PREDICTORS,
    build_anchor_index,
    build_horizon_transition_summary,
    build_trajectory_cases,
    ranking_metrics,
    read_csv,
    run_loso,
    score_distribution,
    summarize_scale_controls,
    summarize_seed_stage,
    write_csv,
)


FEATURES = (
    "single_raw",
    "ema3_raw",
    "mean2_raw",
    "mean3_raw",
    "slope_raw",
    "acceleration_raw",
    "single_relative",
    "ema3_relative",
    "single_cosine",
    "delta_norm",
    "old_grad_norm",
    "norm_product",
)


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def pct(value):
    return "NA" if not finite(value) else f"{100.0 * float(value):.2f}%"


def number(value):
    return "NA" if not finite(value) else f"{float(value):.4f}"


def add_horizon(rows, horizon):
    return [{"horizon": horizon, **row} for row in rows]


def analyze_horizon(rows, horizon):
    feature_rows = []
    fold_rows = []
    loso_predictions = {}
    loso_summary = {}
    rankings = {feature: ranking_metrics(rows, feature) for feature in FEATURES}
    for feature in FORMAL_PREDICTORS:
        pooled, folds, predictions = run_loso(rows, feature)
        loso_predictions[feature] = predictions
        loso_summary[feature] = pooled
        for fold in folds:
            fold_rows.append(
                {
                    "horizon": horizon,
                    "feature": feature,
                    **fold,
                }
            )

    baseline_auc = rankings["single_raw"]["roc_auc"]
    baseline_ba = loso_summary["single_raw"]["balanced_accuracy"]
    for feature in FEATURES:
        ranking = rankings[feature]
        pooled = loso_summary.get(feature)
        predictions = loso_predictions.get(feature, [])
        feature_rows.append(
            {
                "horizon": horizon,
                "feature": feature,
                **ranking,
                "loso_balanced_accuracy": "" if pooled is None else pooled["balanced_accuracy"],
                "loso_false_safe": "" if pooled is None else pooled["false_safe_rate_all"],
                "loso_false_alarm": "" if pooled is None else pooled["false_alarm_rate_all"],
                "loso_sensitivity": "" if pooled is None else pooled["sensitivity"],
                "loso_specificity": "" if pooled is None else pooled["specificity"],
                "heldout_seeds_ba_ge_0_60": "" if pooled is None else sum(
                    fold["test_BA"] >= 0.60
                    for fold in fold_rows
                    if fold["horizon"] == horizon and fold["feature"] == feature
                ),
                "delta_auc_vs_single_raw": ranking["roc_auc"] - baseline_auc,
                "delta_loso_ba_pp_vs_single_raw": "" if pooled is None else 100.0 * (
                    pooled["balanced_accuracy"] - baseline_ba
                ),
            }
        )
    seed_stage = add_horizon(
        summarize_seed_stage(rows, FEATURES, loso_predictions), horizon
    )
    scale = add_horizon(summarize_scale_controls(rows), horizon)
    distribution = add_horizon(
        score_distribution(rows, FORMAL_PREDICTORS), horizon
    )
    return feature_rows, fold_rows, seed_stage, scale, distribution


def lookup(rows, horizon, feature):
    return next(
        row for row in rows
        if int(row["horizon"]) == horizon and row["feature"] == feature
    )


def build_verdict(feature_rows, complete100_per_seed, heterogeneity):
    single100 = lookup(feature_rows, 100, "single_raw")
    ema100 = lookup(feature_rows, 100, "ema3_raw")
    ema300 = lookup(feature_rows, 300, "ema3_raw")
    evidence_floor = (
        ema100["cases"] >= 220
        and all(complete100_per_seed[seed] >= 65 for seed in (42, 43, 44))
    )
    supported = (
        evidence_floor
        and ema100["roc_auc"] >= 0.72
        and ema100["delta_auc_vs_single_raw"] >= 0.05
        and ema100["loso_balanced_accuracy"] >= 0.68
        and ema100["delta_loso_ba_pp_vs_single_raw"] >= 5.0
        and ema100["heldout_seeds_ba_ge_0_60"] >= 2
    )
    not_supported = (
        ema100["delta_auc_vs_single_raw"] < 0.02
        and ema100["delta_loso_ba_pp_vs_single_raw"] < 2.0
        and ema300["roc_auc"] < 0.72
        and ema300["loso_balanced_accuracy"] < 0.65
    )
    delayed = (
        not supported
        and ema300["roc_auc"] >= 0.75
        and ema300["loso_balanced_accuracy"] >= 0.68
    )
    if supported:
        verdict = "OPTIMIZER_AWARE_TRAJECTORY_SIGNAL_SUPPORTED"
    elif not_supported:
        verdict = "OPTIMIZER_AWARE_TRAJECTORY_SIGNAL_NOT_SUPPORTED"
    else:
        verdict = "OPTIMIZER_AWARE_TRAJECTORY_SIGNAL_INCONCLUSIVE"
    if not evidence_floor and verdict == "OPTIMIZER_AWARE_TRAJECTORY_SIGNAL_SUPPORTED":
        verdict = "OPTIMIZER_AWARE_TRAJECTORY_SIGNAL_INCONCLUSIVE"

    if verdict == "OPTIMIZER_AWARE_TRAJECTORY_SIGNAL_SUPPORTED":
        next_action = "Run Dense-Anchor Trajectory Feasibility at a fixed 20- or 50-update interval before any intervention."
    elif verdict == "OPTIMIZER_AWARE_TRAJECTORY_SIGNAL_NOT_SUPPORTED":
        next_action = "Validate BC-loss drift against rollout-success forgetting; close the local and short-trajectory surrogate route."
    elif delayed:
        next_action = "Diagnose a periodic long-horizon retention monitor; do not implement a one-step gate."
    elif heterogeneity == "SEED_STAGE_HETEROGENEITY_HIGH":
        next_action = "Localize the seed-stage failures before any conditioned model or margin intervention."
    else:
        next_action = "Resolve cross-seed and horizon instability before any trajectory intervention."
    return verdict, delayed, evidence_floor, next_action


def write_report(
    output_dir,
    anchors,
    raw100,
    raw300,
    trajectory100,
    trajectory300,
    feature_rows,
    seed_stage,
    transitions,
    scale_rows,
):
    complete100_per_seed = Counter(int(row["seed"]) for row in trajectory100)
    complete300_per_seed = Counter(int(row["seed"]) for row in trajectory300)
    subgroup_auc = [
        float(row["roc_auc"])
        for row in seed_stage
        if int(row["horizon"]) == 100 and row["feature"] == "ema3_raw"
        and finite(row["roc_auc"])
    ]
    auc_range = max(subgroup_auc) - min(subgroup_auc)
    heterogeneity = (
        "SEED_STAGE_HETEROGENEITY_HIGH"
        if auc_range >= 0.20
        else "SEED_STAGE_HETEROGENEITY_MODERATE"
    )
    verdict, delayed, evidence_floor, next_action = build_verdict(
        feature_rows, complete100_per_seed, heterogeneity
    )
    single100 = lookup(feature_rows, 100, "single_raw")
    ema100 = lookup(feature_rows, 100, "ema3_raw")
    ema300 = lookup(feature_rows, 300, "ema3_raw")
    single300 = lookup(feature_rows, 300, "single_raw")
    transition_all = next(row for row in transitions if row["scope"] == "ALL")

    norm100 = {
        row["feature"]: row for row in scale_rows if int(row["horizon"]) == 100
    }
    norm_confounding = any(
        abs(row["roc_auc"] - norm100["actual_step_score"]["roc_auc"]) <= 0.03
        for name, row in norm100.items() if name in ("delta_norm", "norm_product")
    )
    normalized_better = any(
        norm100[name]["roc_auc"] >= norm100["actual_step_score"]["roc_auc"] + 0.05
        for name in ("actual_step_cosine", "actual_step_relative")
    )
    scale_inference = []
    if norm_confounding:
        scale_inference.append("norm-scale confounding is plausible")
    if normalized_better:
        scale_inference.append("scale normalization improves ranking")
    if not scale_inference:
        scale_inference.append("neither preregistered scale diagnostic condition was met")

    subgroup = {
        (int(row["horizon"]), int(row["seed"]), int(row["stage"]), row["feature"]): row
        for row in seed_stage
    }
    seed_lines = []
    for seed in (42, 43, 44):
        for stage in (8, 9):
            seed_lines.append(
                f"| {seed} | {stage} | "
                f"{number(subgroup[(100, seed, stage, 'single_raw')]['roc_auc'])} | "
                f"{number(subgroup[(100, seed, stage, 'ema3_raw')]['roc_auc'])} | "
                f"{number(subgroup[(300, seed, stage, 'single_raw')]['roc_auc'])} | "
                f"{number(subgroup[(300, seed, stage, 'ema3_raw')]['roc_auc'])} |"
            )

    def feature_table(horizon):
        lines = []
        for feature in (
            "single_raw", "ema3_raw", "mean2_raw", "mean3_raw",
            "single_relative", "ema3_relative",
        ):
            row = lookup(feature_rows, horizon, feature)
            lines.append(
                f"| {feature} | {number(row['roc_auc'])} | {number(row['average_precision'])} | "
                f"{pct(row['loso_balanced_accuracy']) if row['loso_balanced_accuracy'] != '' else 'NA'} | "
                f"{pct(row['loso_false_safe']) if row['loso_false_safe'] != '' else 'NA'} |"
            )
        return lines

    scale_lines = [
        f"| {row['feature']} | {number(row['roc_auc'])} | {number(row['average_precision'])} | {row['delta_auc_vs_actual_step_score']:+.4f} |"
        for row in scale_rows if int(row["horizon"]) == 100
    ]
    lines = [
        "# Optimizer-Aware Trajectory Signal Stability Diagnosis", "",
        "## Identity", "",
        f"- Verdict: `{verdict}`",
        "- Branch: `research/optimizer-aware-trajectory-signal`",
        "- Starting SHA: `9795a284a835da0d20c63e22f1bac4bc45113493`",
        "- Final SHA: branch HEAD containing this report (reported in the external handoff)", "",
        "## Input", "",
        f"- Anchors: {len(anchors)}",
        f"- Raw H100 cases: {len(raw100)}",
        f"- Raw H300 cases: {len(raw300)}",
        "- Source SHA: `9795a284a835da0d20c63e22f1bac4bc45113493`", "",
        "## History completeness", "",
        f"- H100 complete: {len(trajectory100)}; per seed: {dict(sorted(complete100_per_seed.items()))}",
        f"- H300 complete: {len(trajectory300)}; per seed: {dict(sorted(complete300_per_seed.items()))}",
        f"- H12 evidence floor: {'met' if evidence_floor else 'not met'}", "",
        "## Seed-stage table", "",
        "| Seed | Stage | Single H100 AUC | EMA3 H100 AUC | Single H300 AUC | EMA3 H300 AUC |",
        "|---:|---:|---:|---:|---:|---:|",
        *seed_lines, "",
        "## H100", "",
        "| Feature | AUC | AP | LOSO BA | LOSO false-safe |",
        "|---|---:|---:|---:|---:|",
        *feature_table(100), "",
        f"EMA3 AUC gain: {ema100['delta_auc_vs_single_raw']:+.4f}; LOSO BA gain: {ema100['delta_loso_ba_pp_vs_single_raw']:+.2f} pp.", "",
        "## H300", "",
        "| Feature | AUC | AP | LOSO BA | LOSO false-safe |",
        "|---|---:|---:|---:|---:|",
        *feature_table(300), "",
        f"Single/EMA3 AUC: {single300['roc_auc']:.4f}/{ema300['roc_auc']:.4f}; single/EMA3 LOSO BA: {pct(single300['loso_balanced_accuracy'])}/{pct(ema300['loso_balanced_accuracy'])}.", "",
        "## Horizon transition", "",
        f"- Paired cases: {transition_all['paired_cases']}",
        f"- safe100 -> harmful300: {transition_all['safe100_harmful300']}",
        f"- Delayed harmful rate among H100-safe: {pct(transition_all['delayed_harmful_rate_among_h100_safe'])}",
        f"- harmful100 -> safe300: {transition_all['harmful100_safe300']}",
        f"- Delayed trajectory finding: {'DELAYED_TRAJECTORY_SIGNAL' if delayed else 'not established'}", "",
        "## Scale", "",
        "| Feature | AUC | AP | Delta AUC vs raw |",
        "|---|---:|---:|---:|",
        *scale_lines, "",
        "Inference: " + "; ".join(scale_inference) + ".", "",
        "## Heterogeneity", "",
        f"`{heterogeneity}`; valid six-group EMA3 H100 AUC range = {auc_range:.4f}.", "",
        "## Observed", "",
        "The fixed EMA3, auxiliary trajectory rankings, LOSO folds, scale controls, and H100/H300 transitions are direct computations from the committed anchor and future-case CSV files.", "",
        "## Inferred", "",
        f"H12 is `{verdict}`. The fixed recent-history aggregation "
        + ("meets" if verdict.endswith("_SUPPORTED") and "NOT" not in verdict else "does not meet")
        + " the preregistered support rule.", "",
        "## Not established", "",
        "Anchor spacing is 100 updates. Trajectory features come from targeted continuation, not the exact original training trajectory. H100/H300 labels are BC-loss retention, not rollout success. LOSO across three seeds is descriptive. EMA3 is a diagnostic predictor, not a validated training method.", "",
        "## Next action", "",
        next_action, "",
    ]
    (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict, heterogeneity, delayed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--h100", type=Path, required=True)
    parser.add_argument("--h300", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_paths = (args.anchor, args.h100, args.h300)
    original_text = {path: path.read_text(encoding="utf-8") for path in source_paths}
    anchors = read_csv(args.anchor)
    raw100 = read_csv(args.h100)
    raw300 = read_csv(args.h300)
    if (len(anchors), len(raw100), len(raw300)) != (267, 261, 249):
        raise ValueError(
            f"unexpected input sizes: {(len(anchors), len(raw100), len(raw300))}"
        )
    anchor_index = build_anchor_index(anchors)
    trajectory100 = build_trajectory_cases(raw100, anchor_index)
    trajectory300 = build_trajectory_cases(raw300, anchor_index)

    for source, trajectory in ((raw100, trajectory100), (raw300, trajectory300)):
        source_index = {
            (int(row["seed"]), int(row["stage"]), int(row["global_update"])): row
            for row in source
        }
        for row in trajectory:
            original = source_index[(int(row["seed"]), int(row["stage"]), int(row["global_update"]))]
            if float(row["single_raw"]) != float(original["actual_step_score"]):
                raise AssertionError("single baseline parity failed")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "trajectory_cases_h100.csv", trajectory100)
    write_csv(output_dir / "trajectory_cases_h300.csv", trajectory300)
    analyses = [analyze_horizon(trajectory100, 100), analyze_horizon(trajectory300, 300)]
    feature_rows = analyses[0][0] + analyses[1][0]
    fold_rows = analyses[0][1] + analyses[1][1]
    seed_stage = analyses[0][2] + analyses[1][2]
    scale_rows = analyses[0][3] + analyses[1][3]
    distributions = analyses[0][4] + analyses[1][4]
    transitions, _ = build_horizon_transition_summary(trajectory100, trajectory300)
    write_csv(output_dir / "feature_summary.csv", feature_rows)
    write_csv(output_dir / "loso_fold_metrics.csv", fold_rows)
    write_csv(output_dir / "seed_stage_summary.csv", seed_stage)
    write_csv(output_dir / "horizon_transition_summary.csv", transitions)
    write_csv(output_dir / "scale_diagnostic_summary.csv", scale_rows)
    write_csv(output_dir / "score_distribution_summary.csv", distributions)
    verdict, heterogeneity, delayed = write_report(
        output_dir,
        anchors,
        raw100,
        raw300,
        trajectory100,
        trajectory300,
        feature_rows,
        seed_stage,
        transitions,
        scale_rows,
    )
    for path, text in original_text.items():
        if path.read_text(encoding="utf-8") != text:
            raise AssertionError(f"input CSV changed: {path}")
    print(
        f"[formal] anchors={len(anchors)} raw_h100={len(raw100)} "
        f"complete_h100={len(trajectory100)} raw_h300={len(raw300)} "
        f"complete_h300={len(trajectory300)}"
    )
    print(f"verdict={verdict}")
    print(f"heterogeneity={heterogeneity}")
    print(f"delayed_trajectory_signal={delayed}")


if __name__ == "__main__":
    main()
