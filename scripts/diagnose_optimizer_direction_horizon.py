#!/usr/bin/env python3
import argparse
import math
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero.lifelong.analysis.optimizer_direction_horizon import (
    build_unique_index,
    distribution_summary,
    enrich_case,
    pair_horizons,
    ranking_metrics,
    read_csv,
    reversal_summary,
    run_loso,
    subgroup_rows,
    summarize_seed_stage,
    write_csv,
)


PREDICTORS = (
    "actual_step_score",
    "actual_step_cosine",
    "actual_step_relative",
)
CONTROL_FEATURES = PREDICTORS + ("delta_norm", "old_grad_norm", "norm_product")


def is_finite(value):
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def fmt(value):
    return "NA" if not is_finite(value) else f"{float(value):.4f}"


def pct(value):
    return "NA" if not is_finite(value) else f"{100.0 * float(value):.2f}%"


def analyze_full(rows, horizon):
    summaries = []
    folds_out = []
    prediction_cache = {}
    by_feature = {}
    for feature in PREDICTORS:
        ranking = ranking_metrics(rows, feature)
        pooled, folds, predictions = run_loso(rows, feature)
        prediction_cache[feature] = predictions
        by_feature[feature] = (ranking, pooled, folds)
        for fold in folds:
            folds_out.append(
                {
                    "scope": "full",
                    "horizon": horizon,
                    "task": "endpoint_harmful",
                    "feature": feature,
                    **fold,
                }
            )
    raw_ranking, raw_pooled, _ = by_feature["actual_step_score"]
    for feature in PREDICTORS:
        ranking, pooled, folds = by_feature[feature]
        summaries.append(
            {
                "horizon": horizon,
                "feature": feature,
                "cases": ranking["cases"],
                "harmful_prevalence": ranking["positive_prevalence"],
                "roc_auc": ranking["roc_auc"],
                "average_precision": ranking["average_precision"],
                "loso_balanced_accuracy": pooled["balanced_accuracy"],
                "loso_sensitivity": pooled["sensitivity"],
                "loso_specificity": pooled["specificity"],
                "loso_false_safe": pooled["false_safe_rate_all"],
                "loso_false_alarm": pooled["false_alarm_rate_all"],
                "heldout_seeds_ba_ge_0_60": sum(
                    float(fold["test_BA"]) >= 0.60
                    for fold in folds if int(fold["valid_fold"])
                ),
                "delta_auc_vs_raw": ranking["roc_auc"] - raw_ranking["roc_auc"],
                "delta_loso_ba_pp_vs_raw": 100.0 * (
                    pooled["balanced_accuracy"] - raw_pooled["balanced_accuracy"]
                ),
                "delta_false_safe_pp_vs_raw": 100.0 * (
                    pooled["false_safe_rate_all"] - raw_pooled["false_safe_rate_all"]
                ),
            }
        )
    seed_stage = summarize_seed_stage(
        rows, PREDICTORS, horizon, prediction_cache["actual_step_cosine"]
    )
    return summaries, folds_out, seed_stage


def analyze_reversal_tasks(paired):
    output = []
    fold_rows = []
    for task in ("stable_endpoint", "delayed_harmful", "persistent_harmful"):
        rows = subgroup_rows(paired, task)
        label_fn = lambda row: bool(int(row["subgroup_positive"]))
        for feature in PREDICTORS:
            ranking = ranking_metrics(rows, feature, label_fn)
            pooled, folds, _ = run_loso(rows, feature, label_fn)
            valid_folds = [fold for fold in folds if int(fold["valid_fold"])]
            output.append(
                {
                    "task": task,
                    "feature": feature,
                    "cases": ranking["cases"],
                    "positive_cases": ranking["positive_cases"],
                    "positive_prevalence": ranking["positive_prevalence"],
                    "roc_auc": ranking["roc_auc"],
                    "average_precision": ranking["average_precision"],
                    "loso_balanced_accuracy": "NA" if pooled is None else pooled["balanced_accuracy"],
                    "loso_sensitivity": "NA" if pooled is None else pooled["sensitivity"],
                    "loso_specificity": "NA" if pooled is None else pooled["specificity"],
                    "valid_loso_folds": len(valid_folds),
                    "heldout_folds_ba_ge_0_60": sum(
                        float(fold["test_BA"]) >= 0.60 for fold in valid_folds
                    ),
                }
            )
            for fold in folds:
                fold_rows.append(
                    {
                        "scope": "reversal_subgroup",
                        "horizon": "PAIRED",
                        "task": task,
                        "feature": feature,
                        **fold,
                    }
                )
    return output, fold_rows


def scale_controls(rows, horizon):
    cosine_auc = ranking_metrics(rows, "actual_step_cosine")["roc_auc"]
    output = []
    for feature in CONTROL_FEATURES:
        ranking = ranking_metrics(rows, feature)
        output.append(
            {
                "horizon": horizon,
                "feature": feature,
                "cases": ranking["cases"],
                "roc_auc": ranking["roc_auc"],
                "average_precision": ranking["average_precision"],
                "delta_auc_vs_cosine": ranking["roc_auc"] - cosine_auc,
            }
        )
    return output


def lookup(rows, **criteria):
    return next(
        row for row in rows
        if all(str(row[key]) == str(value) for key, value in criteria.items())
    )


def decide(directional, reversal_rankings, reversal_rows, scale_rows, per_seed):
    primary = lookup(directional, horizon="H100", feature="actual_step_cosine")
    raw = lookup(directional, horizon="H100", feature="actual_step_score")
    h300 = lookup(directional, horizon="H300", feature="actual_step_cosine")
    supported = (
        primary["cases"] >= 250
        and min(per_seed.values()) >= 80
        and primary["roc_auc"] >= 0.70
        and primary["loso_balanced_accuracy"] >= 0.65
        and primary["loso_sensitivity"] >= 0.60
        and primary["heldout_seeds_ba_ge_0_60"] >= 2
        and primary["roc_auc"] - raw["roc_auc"] >= 0.03
    )
    not_supported = (
        primary["roc_auc"] < 0.60
        and primary["loso_balanced_accuracy"] < 0.58
        and h300["roc_auc"] < 0.65
        and h300["loso_balanced_accuracy"] < 0.60
    )
    if supported:
        verdict = "OPTIMIZER_STEP_DIRECTIONAL_RISK_SUPPORTED"
    elif not_supported:
        verdict = "OPTIMIZER_STEP_DIRECTIONAL_RISK_NOT_SUPPORTED"
    else:
        verdict = "OPTIMIZER_STEP_DIRECTIONAL_RISK_INCONCLUSIVE"

    overall = lookup(reversal_rows, scope="ALL")
    stable = lookup(
        reversal_rankings, task="stable_endpoint", feature="actual_step_cosine"
    )
    delayed = lookup(
        reversal_rankings, task="delayed_harmful", feature="actual_step_cosine"
    )
    persistent = lookup(
        reversal_rankings, task="persistent_harmful", feature="actual_step_cosine"
    )
    norm_product = lookup(scale_rows, horizon="H100", feature="norm_product")
    flags = []
    if overall["flip_rate"] >= 0.15:
        flags.append("HORIZON_REVERSAL_MATERIAL")
    if (
        stable["roc_auc"] >= 0.75
        and stable["roc_auc"] - primary["roc_auc"] >= 0.05
        and overall["flip_rate"] >= 0.15
    ):
        flags.append("REVERSAL_EXPLAINS_DIRECTIONAL_DEGRADATION")
    delayed_ba = delayed["loso_balanced_accuracy"]
    if (
        delayed["cases"] >= 100 and delayed["positive_cases"] >= 15
        and delayed["roc_auc"] >= 0.70
        and delayed_ba != "NA" and delayed_ba >= 0.65
        and delayed["heldout_folds_ba_ge_0_60"] >= 2
    ):
        flags.append("DELAYED_HARMFUL_DIRECTIONAL_SIGNAL")
    persistent_ba = persistent["loso_balanced_accuracy"]
    if (
        persistent["cases"] >= 60 and persistent["positive_cases"] >= 20
        and persistent["roc_auc"] >= 0.70
        and persistent_ba != "NA" and persistent_ba >= 0.65
    ):
        flags.append("PERSISTENT_HARMFUL_DIRECTIONAL_SIGNAL")
    if (
        verdict != "OPTIMIZER_STEP_DIRECTIONAL_RISK_SUPPORTED"
        and h300["roc_auc"] >= 0.72
        and h300["loso_balanced_accuracy"] >= 0.65
        and h300["heldout_seeds_ba_ge_0_60"] >= 2
    ):
        flags.append("DELAYED_DIRECTIONAL_SIGNAL")
    if (
        primary["roc_auc"] - raw["roc_auc"] >= 0.03
        and primary["roc_auc"] - norm_product["roc_auc"] >= 0.10
    ):
        flags.append("DIRECTIONAL_NORMALIZATION_ADVANTAGE")
    if not any(
        flag in flags for flag in (
            "HORIZON_REVERSAL_MATERIAL",
            "DELAYED_HARMFUL_DIRECTIONAL_SIGNAL",
            "PERSISTENT_HARMFUL_DIRECTIONAL_SIGNAL",
        )
    ):
        flags.append("NO_CLEAR_REVERSAL_STRUCTURE")

    if verdict == "OPTIMIZER_STEP_DIRECTIONAL_RISK_SUPPORTED":
        next_action = "Run Candidate AdamW-Step Cosine Feasibility Diagnosis before any guard training."
    elif (
        verdict == "OPTIMIZER_STEP_DIRECTIONAL_RISK_INCONCLUSIVE"
        and "DELAYED_HARMFUL_DIRECTIONAL_SIGNAL" in flags
    ):
        next_action = "Test a periodic optimizer-direction retention monitor or denser-anchor targeted diagnosis; do not build a one-step gate."
    elif "REVERSAL_EXPLAINS_DIRECTIONAL_DEGRADATION" in flags:
        next_action = "Test BC-loss path alignment with rollout-success retention before optimizing any single-horizon classifier."
    elif verdict == "OPTIMIZER_STEP_DIRECTIONAL_RISK_NOT_SUPPORTED":
        next_action = "Close first-order score engineering and validate BC-loss drift against rollout-success forgetting."
    else:
        next_action = "Resolve the remaining direction and horizon instability before any intervention."
    return verdict, flags, next_action


def write_report(
    output_dir, directional, folds, seed_stage, reversal_rows,
    reversal_rankings, scale_rows, verdict, flags, heterogeneity, next_action,
    counts,
):
    def directional_table(horizon):
        lines = []
        for feature, label in (
            ("actual_step_score", "raw"),
            ("actual_step_cosine", "cosine"),
            ("actual_step_relative", "relative"),
        ):
            row = lookup(directional, horizon=horizon, feature=feature)
            lines.append(
                f"| {label} | {fmt(row['roc_auc'])} | {fmt(row['average_precision'])} | "
                f"{pct(row['loso_balanced_accuracy'])} | {pct(row['loso_sensitivity'])} | {pct(row['loso_false_safe'])} |"
            )
        return lines

    cosine_folds = [
        row for row in folds
        if row["scope"] == "full" and row["horizon"] == "H100"
        and row["feature"] == "actual_step_cosine"
    ]
    fold_lines = [
        f"| {row['heldout_seed']} | {float(row['threshold']):.6g} | {pct(float(row['test_BA']))} | "
        f"{pct(float(row['test_sensitivity']))} | {pct(float(row['test_specificity']))} |"
        for row in cosine_folds
    ]
    subgroup = {
        (row["task"], row["feature"]): row for row in reversal_rankings
    }
    overall = lookup(reversal_rows, scope="ALL")
    seed_stage_index = {
        (row["horizon"], int(row["seed"]), int(row["stage"]), row["feature"]): row
        for row in seed_stage
    }
    heterogeneity_aucs = [
        float(row["roc_auc"]) for row in seed_stage
        if row["horizon"] == "H100" and row["feature"] == "actual_step_cosine"
        and is_finite(row["roc_auc"])
    ]
    heterogeneity_range = max(heterogeneity_aucs) - min(heterogeneity_aucs)
    seed_lines = []
    for seed in (42, 43, 44):
        for stage in (8, 9):
            seed_lines.append(
                f"| {seed} | {stage} | {fmt(seed_stage_index[('H100', seed, stage, 'actual_step_cosine')]['roc_auc'])} | "
                f"{fmt(seed_stage_index[('H300', seed, stage, 'actual_step_cosine')]['roc_auc'])} |"
            )
    scale_lines = [
        f"| {row['feature']} | {fmt(row['roc_auc'])} | {fmt(row['average_precision'])} |"
        for row in scale_rows if row["horizon"] == "H100"
    ]
    primary = lookup(directional, horizon="H100", feature="actual_step_cosine")
    raw = lookup(directional, horizon="H100", feature="actual_step_score")

    def subgroup_line(task):
        row = subgroup[(task, "actual_step_cosine")]
        return (
            f"Cases/positives {row['cases']}/{row['positive_cases']}; cosine AUC/AP "
            f"{fmt(row['roc_auc'])}/{fmt(row['average_precision'])}; LOSO BA "
            f"{pct(row['loso_balanced_accuracy']) if row['loso_balanced_accuracy'] != 'NA' else 'NA'}; "
            f"valid folds {row['valid_loso_folds']}/3."
        )

    lines = [
        "# Optimizer-Step Direction and Horizon Reversal Diagnosis", "",
        "## Identity", "",
        f"- Verdict: `{verdict}`",
        f"- Flags: {', '.join(f'`{flag}`' for flag in flags)}",
        "- Branch: `research/optimizer-step-direction-horizon-reversal`",
        "- Starting SHA: `9691aca9dbebc4481aaa92fba192b6ff5d2b1b84`",
        "- Final SHA: branch HEAD containing this report (reported in the external handoff)", "",
        "## Inputs", "",
        f"- Anchors: {counts['anchors']}",
        f"- H100: {counts['h100']}",
        f"- H300: {counts['h300']}",
        f"- Paired: {counts['paired']}", "",
        "## Primary H100", "",
        "| Score | AUC | AP | LOSO BA | Sensitivity | False-safe |",
        "|---|---:|---:|---:|---:|---:|",
        *directional_table("H100"), "",
        f"Cosine gain versus raw: AUC {primary['delta_auc_vs_raw']:+.4f}; LOSO BA {primary['delta_loso_ba_pp_vs_raw']:+.2f} pp.", "",
        "## H300", "",
        "| Score | AUC | AP | LOSO BA | Sensitivity | False-safe |",
        "|---|---:|---:|---:|---:|---:|",
        *directional_table("H300"), "",
        "## LOSO cosine folds", "",
        "| Seed | tau | BA | Sensitivity | Specificity |",
        "|---:|---:|---:|---:|---:|",
        *fold_lines, "",
        "## Seed-stage", "",
        "| Seed | Stage | H100 cosine AUC | H300 cosine AUC |",
        "|---:|---:|---:|---:|",
        *seed_lines, "",
        f"Direction heterogeneity: `{heterogeneity}`; six-group H100 cosine AUC range = {heterogeneity_range:.4f}.", "",
        "## Horizon transition", "",
        f"SS={overall['SS']}, SH={overall['SH']}, HH={overall['HH']}, HS={overall['HS']}; flip rate {pct(overall['flip_rate'])}; delayed rate among H100-safe {pct(overall['delayed_harmful_rate_among_h100_safe'])}; recovery rate among H100-harmful {pct(overall['recovery_rate_among_h100_harmful'])}.", "",
        "## Stable-label ranking", "",
        subgroup_line("stable_endpoint"), "",
        "## Delayed harmful", "",
        subgroup_line("delayed_harmful"), "",
        "## Persistent harmful", "",
        subgroup_line("persistent_harmful"), "",
        "## Scale controls", "",
        "| Feature | AUC | AP |",
        "|---|---:|---:|",
        *scale_lines, "",
        "## Observed", "",
        "The directional rankings, cross-seed thresholds, transition counts, stable/delayed/persistent subgroup rankings, and norm controls are direct computations from the committed anchor and raw H100/H300 CSVs.", "",
        "## Inferred", "",
        f"H13 is `{verdict}`. Direction normalization "
        + ("meets" if verdict.endswith("_SUPPORTED") and "NOT" not in verdict else "does not meet")
        + " the preregistered primary support rule. Horizon interpretations are limited to the listed descriptive flags.", "",
        "## Not established", "",
        "Actual-step cosine is a post-update diagnostic score, not a pre-update deployable gate. The targeted continuations are distribution-matched, not the exact historical lifelong trajectory. H100/H300 labels are old-task BC-loss drift, not rollout-success forgetting. Three-seed LOSO is descriptive, not a statistical significance claim. Horizon reversal does not by itself prove optimizer path dependence as the unique cause.", "",
        "## Next action", "",
        next_action, "",
    ]
    (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--h100", type=Path, required=True)
    parser.add_argument("--h300", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    sources = (args.anchor, args.h100, args.h300)
    source_text = {path: path.read_text(encoding="utf-8") for path in sources}
    anchors = read_csv(args.anchor)
    raw100 = read_csv(args.h100)
    raw300 = read_csv(args.h300)
    if (len(anchors), len(raw100), len(raw300)) != (267, 261, 249):
        raise ValueError(f"unexpected input counts: {(len(anchors), len(raw100), len(raw300))}")
    anchor_index = build_unique_index(anchors)
    build_unique_index(raw100)
    build_unique_index(raw300)
    h100 = [enrich_case(row, anchor_index) for row in raw100]
    h300 = [enrich_case(row, anchor_index) for row in raw300]

    d100, f100, s100 = analyze_full(h100, "H100")
    d300, f300, s300 = analyze_full(h300, "H300")
    directional = d100 + d300
    seed_stage = s100 + s300
    paired = pair_horizons(h100, h300, anchor_index)
    if len(paired) != 249:
        raise ValueError(f"expected 249 paired cases, found {len(paired)}")
    reversal_rows = reversal_summary(paired)
    reversal_rankings, reversal_folds = analyze_reversal_tasks(paired)
    folds = f100 + f300 + reversal_folds
    scales = scale_controls(h100, "H100") + scale_controls(h300, "H300")
    distributions = distribution_summary(paired, PREDICTORS)
    per_seed = Counter(int(row["seed"]) for row in h100)
    cosine_subgroups = [
        float(row["roc_auc"]) for row in seed_stage
        if row["horizon"] == "H100" and row["feature"] == "actual_step_cosine"
        and is_finite(row["roc_auc"])
    ]
    auc_range = max(cosine_subgroups) - min(cosine_subgroups)
    heterogeneity = (
        "DIRECTION_HETEROGENEITY_HIGH"
        if auc_range >= 0.20 else "DIRECTION_HETEROGENEITY_MODERATE"
    )
    verdict, flags, next_action = decide(
        directional, reversal_rankings, reversal_rows, scales, per_seed
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "directional_summary.csv", directional)
    write_csv(output / "loso_fold_metrics.csv", folds)
    write_csv(output / "seed_stage_summary.csv", seed_stage)
    write_csv(output / "paired_horizon_cases.csv", paired)
    write_csv(output / "horizon_reversal_summary.csv", reversal_rows)
    write_csv(output / "reversal_ranking_summary.csv", reversal_rankings)
    write_csv(output / "score_distribution_summary.csv", distributions)
    write_csv(output / "scale_control_summary.csv", scales)
    write_report(
        output, directional, folds, seed_stage, reversal_rows,
        reversal_rankings, scales, verdict, flags, heterogeneity, next_action,
        {"anchors": len(anchors), "h100": len(h100), "h300": len(h300), "paired": len(paired)},
    )
    for path, text in source_text.items():
        if path.read_text(encoding="utf-8") != text:
            raise AssertionError(f"input changed: {path}")
    print(f"[formal] anchors={len(anchors)} h100={len(h100)} h300={len(h300)} paired={len(paired)}")
    print(f"verdict={verdict}")
    print("flags=" + ",".join(flags))
    print(f"heterogeneity={heterogeneity} range={auc_range:.6f}")


if __name__ == "__main__":
    main()
