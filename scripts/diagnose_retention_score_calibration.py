#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from libero.lifelong.analysis.retention_score_calibration import (
    CASE_KEY,
    binary_metrics,
    classification_metrics,
    loso_predict,
    pooled_loso_metrics,
    quantile,
    ranking_metrics,
    select_balanced_threshold,
    unique_cases,
)


FORMAL_RHOS = (0.05, 0.10, 0.25, 0.50)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_flag(value):
    return str(value).strip().lower() in ("1", "true")


def validate_input(rows):
    if len(rows) != 720:
        raise ValueError(f"expected 720 input rows, found {len(rows)}")
    if any(row["interval_type"] != "short" for row in rows):
        raise ValueError("input contains a non-short transition")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[field] for field in CASE_KEY)].append(row)
    if len(grouped) != 180:
        raise ValueError(f"expected 180 unique cases, found {len(grouped)}")
    for key, selected in grouped.items():
        rhos = sorted(float(row["rho"]) for row in selected)
        if rhos != list(FORMAL_RHOS):
            raise ValueError(f"case {key} has invalid horizons: {rhos}")
    cases = unique_cases(rows)
    counts = Counter(row["method"] for row in cases)
    if counts != {"V1": 90, "V2": 90}:
        raise ValueError(f"unexpected method case counts: {dict(counts)}")
    for method in ("V1", "V2"):
        seed_counts = Counter(
            row["seed"] for row in cases if row["method"] == method
        )
        if seed_counts != {"42": 30, "43": 30, "44": 30}:
            raise ValueError(f"unexpected {method} seed counts: {dict(seed_counts)}")
    return cases


def score_specs(rows, methods, rhos):
    specs = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        specs.append(
            {
                "method": method,
                "score_type": "first_order",
                "rho": "NA",
                "rows": unique_cases(method_rows),
                "score_field": "pred_all",
            }
        )
        for rho in rhos:
            selected = [
                row for row in method_rows
                if abs(float(row["rho"]) - float(rho)) < 1e-12
            ]
            specs.append(
                {
                    "method": method,
                    "score_type": "finite_step",
                    "rho": f"{rho:.2f}",
                    "rows": selected,
                    "score_field": "probe_delta",
                }
            )
    return specs


def distribution_rows(specs):
    output = []
    for spec in specs:
        for label_name, label in (("safe", False), ("harmful", True)):
            values = [
                float(row[spec["score_field"]])
                for row in spec["rows"]
                if (float(row["actual_delta"]) > 0.0) == label
            ]
            output.append(
                {
                    "method": spec["method"],
                    "score_type": spec["score_type"],
                    "rho": spec["rho"],
                    "endpoint_label": label_name,
                    "n": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "q25": quantile(values, 0.25),
                    "q75": quantile(values, 0.75),
                }
            )
    return output


def ranking_rows(specs):
    output = []
    stages = []
    for spec in specs:
        scopes = [("ALL", spec["rows"])]
        scopes.extend(
            (seed, [row for row in spec["rows"] if row["seed"] == seed])
            for seed in ("42", "43", "44")
        )
        for seed, selected in scopes:
            metrics = ranking_metrics(selected, spec["score_field"])
            material = [row for row in selected if parse_flag(row["material_change"])]
            material_metrics = ranking_metrics(material, spec["score_field"])
            output.append(
                {
                    "method": spec["method"],
                    "seed": seed,
                    "score_type": spec["score_type"],
                    "rho": spec["rho"],
                    **metrics,
                    "material_cases": len(material),
                    "material_roc_auc": material_metrics["roc_auc"],
                    "material_average_precision": material_metrics["average_precision"],
                }
            )
        for stage in (7, 8, 9):
            selected = [row for row in spec["rows"] if int(row["stage"]) == stage]
            metrics = ranking_metrics(selected, spec["score_field"])
            stages.append(
                {
                    "method": spec["method"],
                    "stage": stage,
                    "score_type": spec["score_type"],
                    "rho": spec["rho"],
                    **metrics,
                }
            )
    return output, stages


def calibration_rows(specs):
    summaries = []
    fold_rows = []
    threshold_rows = []
    for spec in specs:
        rows = spec["rows"]
        scores = [float(row[spec["score_field"]]) for row in rows]
        labels = [float(row["actual_delta"]) > 0.0 for row in rows]
        ranking = ranking_metrics(rows, spec["score_field"])
        zero = classification_metrics(scores, labels, 0.0)
        predictions, folds = loso_predict(rows, spec["score_field"])
        loso = pooled_loso_metrics(predictions)
        for fold in folds:
            fold_rows.append(
                {
                    "method": spec["method"],
                    "score_type": spec["score_type"],
                    "rho": spec["rho"],
                    "heldout_seed": fold["heldout_seed"],
                    "train_seeds": fold["train_seeds"],
                    "threshold": fold["threshold"],
                    "train_balanced_accuracy": fold["train_balanced_accuracy"],
                    "train_sensitivity": fold["train_sensitivity"],
                    "train_specificity": fold["train_specificity"],
                    "test_cases": fold["test_cases"],
                    "test_balanced_accuracy": fold["test_balanced_accuracy"],
                    "test_sensitivity": fold["test_sensitivity"],
                    "test_specificity": fold["test_specificity"],
                    "test_false_safe_rate_all": fold["test_false_safe_rate_all"],
                    "test_false_alarm_rate_all": fold["test_false_alarm_rate_all"],
                    "test_false_safe_rate_among_harmful": fold["test_false_safe_rate_among_harmful"],
                    "test_false_alarm_rate_among_safe": fold["test_false_alarm_rate_among_safe"],
                }
            )
        summaries.append(
            {
                "method": spec["method"],
                "score_type": spec["score_type"],
                "rho": spec["rho"],
                **ranking,
                "zero_balanced_accuracy": zero["balanced_accuracy"],
                "zero_false_safe_rate_all": zero["false_safe_rate_all"],
                "zero_false_alarm_rate_all": zero["false_alarm_rate_all"],
                "loso_balanced_accuracy": loso["balanced_accuracy"],
                "loso_false_safe_rate_all": loso["false_safe_rate_all"],
                "loso_false_alarm_rate_all": loso["false_alarm_rate_all"],
                "loso_sensitivity": loso["sensitivity"],
                "loso_specificity": loso["specificity"],
                "loso_ba_gain_pp": 100.0 * (
                    loso["balanced_accuracy"] - zero["balanced_accuracy"]
                ),
                "loso_false_safe_reduction_pp": 100.0 * (
                    zero["false_safe_rate_all"] - loso["false_safe_rate_all"]
                ),
                "loso_false_alarm_change_pp": 100.0 * (
                    loso["false_alarm_rate_all"] - zero["false_alarm_rate_all"]
                ),
                "heldout_seeds_ba_ge_0_60": sum(
                    fold["test_balanced_accuracy"] >= 0.60 for fold in folds
                ),
                "finite_vs_first_order_roc_auc_gain": "",
                "finite_vs_first_order_ap_gain": "",
                "finite_vs_first_order_loso_ba_gain_pp": "",
                "finite_vs_first_order_false_safe_reduction_pp": "",
            }
        )
        oracle = select_balanced_threshold(scores, labels)
        for kind, descriptive_oracle, threshold, metrics in (
            ("zero", 0, 0.0, zero),
            ("pooled_best", 1, oracle["threshold"], oracle["metrics"]),
        ):
            threshold_rows.append(
                {
                    "method": spec["method"],
                    "score_type": spec["score_type"],
                    "rho": spec["rho"],
                    "calibration_kind": kind,
                    "descriptive_oracle": descriptive_oracle,
                    "threshold": threshold,
                    **metrics,
                }
            )
    by_method_fo = {
        row["method"]: row for row in summaries if row["score_type"] == "first_order"
    }
    for row in summaries:
        if row["score_type"] != "finite_step":
            continue
        baseline = by_method_fo[row["method"]]
        row["finite_vs_first_order_roc_auc_gain"] = row["roc_auc"] - baseline["roc_auc"]
        row["finite_vs_first_order_ap_gain"] = (
            row["average_precision"] - baseline["average_precision"]
        )
        row["finite_vs_first_order_loso_ba_gain_pp"] = 100.0 * (
            row["loso_balanced_accuracy"] - baseline["loso_balanced_accuracy"]
        )
        row["finite_vs_first_order_false_safe_reduction_pp"] = 100.0 * (
            baseline["loso_false_safe_rate_all"] - row["loso_false_safe_rate_all"]
        )
    return summaries, fold_rows, threshold_rows


def find_summary(rows, method, score_type, rho):
    return next(
        row for row in rows
        if row["method"] == method and row["score_type"] == score_type
        and row["rho"] == rho
    )


def pct(value):
    return f"{100.0 * float(value):.2f}%"


def pp(value):
    return f"{float(value):+.2f} pp"


def write_report(output_dir, summaries, folds, rankings, distributions, stages, formal):
    if not formal:
        primary = find_summary(summaries, "V1", "finite_step", "0.10")
        lines = [
            "# Retention Score Ranking and Calibration Smoke", "",
            "- Verdict: `SMOKE_ONLY_NOT_SCIENTIFIC`",
            f"- Cases: {primary['cases']}",
            f"- ROC-AUC: {primary['roc_auc']:.6f}",
            f"- LOSO folds: {sum(row['method'] == 'V1' for row in folds)}", "",
        ]
        (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
        return "SMOKE_ONLY_NOT_SCIENTIFIC", "UNRESOLVED_SCORE_SOURCE"

    primary = find_summary(summaries, "V1", "finite_step", "0.10")
    rho25 = find_summary(summaries, "V1", "finite_step", "0.25")
    rho50 = find_summary(summaries, "V1", "finite_step", "0.50")
    first_order = find_summary(summaries, "V1", "first_order", "NA")
    supported = (
        primary["roc_auc"] >= 0.70
        and primary["loso_balanced_accuracy"] >= 0.65
        and primary["loso_ba_gain_pp"] >= 10.0
        and primary["loso_false_safe_reduction_pp"] >= 10.0
        and primary["heldout_seeds_ba_ge_0_60"] >= 2
    )
    not_supported = (
        primary["roc_auc"] < 0.60
        and primary["loso_balanced_accuracy"] < 0.58
        and rho25["roc_auc"] < 0.65
        and rho25["loso_balanced_accuracy"] < 0.60
        and rho50["roc_auc"] < 0.65
        and rho50["loso_balanced_accuracy"] < 0.60
    )
    if supported:
        verdict = "LOCAL_RETENTION_SCORE_CALIBRATION_SUPPORTED"
    elif not_supported:
        verdict = "LOCAL_RETENTION_SCORE_CALIBRATION_NOT_SUPPORTED"
    else:
        verdict = "LOCAL_RETENTION_SCORE_CALIBRATION_INCONCLUSIVE"
    if (
        primary["roc_auc"] - first_order["roc_auc"] >= 0.05
        and primary["loso_balanced_accuracy"] - first_order["loso_balanced_accuracy"] >= 0.05
    ):
        attribution = "FINITE_STEP_ADDS_VALUE"
    elif (
        first_order["roc_auc"] >= 0.70
        and first_order["loso_balanced_accuracy"] >= 0.65
        and first_order["heldout_seeds_ba_ge_0_60"] >= 2
    ):
        attribution = "FIRST_ORDER_MAGNITUDE_SUFFICIENT"
    elif (
        first_order["roc_auc"] < 0.60
        and primary["roc_auc"] < 0.60
        and rho25["roc_auc"] < 0.65
        and rho50["roc_auc"] < 0.65
    ):
        attribution = "NO_USEFUL_LOCAL_RANKING"
    else:
        attribution = "UNRESOLVED_SCORE_SOURCE"

    if verdict == "LOCAL_RETENTION_SCORE_CALIBRATION_SUPPORTED" and attribution == "FINITE_STEP_ADDS_VALUE":
        next_action = "Test online candidate-step feasibility for a calibrated finite functional retention score on Temporal V1."
    elif verdict == "LOCAL_RETENTION_SCORE_CALIBRATION_SUPPORTED" and attribution == "FIRST_ORDER_MAGNITUDE_SUFFICIENT":
        next_action = "Test a margin-aware calibrated first-order retention gate on Temporal V1."
    elif verdict == "LOCAL_RETENTION_SCORE_CALIBRATION_NOT_SUPPORTED" and attribution == "NO_USEFUL_LOCAL_RANKING":
        next_action = "Start REALIZED TRAINING TRAJECTORY DIAGNOSIS: measure path-direction rotation, old-loss slope reversals, displacement cosine, and late harmful onset."
    else:
        next_action = "Inspect which held-out seed and late horizon make calibration unstable before any intervention training."

    table_specs = [
        ("V1", "first_order", "NA"),
        ("V1", "finite_step", "0.05"),
        ("V1", "finite_step", "0.10"),
        ("V1", "finite_step", "0.25"),
        ("V1", "finite_step", "0.50"),
        ("V2", "first_order", "NA"),
        ("V2", "finite_step", "0.10"),
    ]
    main_rows = []
    for method, score_type, rho in table_specs:
        row = find_summary(summaries, method, score_type, rho)
        main_rows.append(
            f"| {method} | {score_type} | {rho} | {row['roc_auc']:.4f} | "
            f"{row['average_precision']:.4f} | {pct(row['zero_balanced_accuracy'])} | "
            f"{pct(row['loso_balanced_accuracy'])} | {pct(row['zero_false_safe_rate_all'])} | "
            f"{pct(row['loso_false_safe_rate_all'])} |"
        )
    primary_folds = [
        row for row in folds
        if row["method"] == "V1" and row["score_type"] == "finite_step"
        and row["rho"] == "0.10"
    ]
    fold_lines = [
        f"| {row['heldout_seed']} | {row['train_seeds']} | {float(row['threshold']):.6f} | "
        f"{pct(row['test_balanced_accuracy'])} | {pct(row['test_sensitivity'])} | "
        f"{pct(row['test_specificity'])} | {pct(row['test_false_safe_rate_all'])} | "
        f"{pct(row['test_false_safe_rate_among_harmful'])} |"
        for row in primary_folds
    ]
    horizon_lines = []
    for rho in ("0.05", "0.10", "0.25", "0.50"):
        row = find_summary(summaries, "V1", "finite_step", rho)
        horizon_lines.append(
            f"| {rho} | {row['roc_auc']:.4f} | {row['average_precision']:.4f} | "
            f"{pct(row['loso_balanced_accuracy'])} | {pct(row['loso_false_safe_rate_all'])} |"
        )
    distribution_lookup = {
        (row["method"], row["score_type"], row["rho"], row["endpoint_label"]): row
        for row in distributions
    }
    safe_dist = distribution_lookup[("V1", "finite_step", "0.10", "safe")]
    harmful_dist = distribution_lookup[("V1", "finite_step", "0.10", "harmful")]
    primary_rank = next(
        row for row in rankings
        if row["method"] == "V1" and row["seed"] == "ALL"
        and row["score_type"] == "finite_step" and row["rho"] == "0.10"
    )
    material_consistent = (
        (primary_rank["roc_auc"] >= 0.5) == (primary_rank["material_roc_auc"] >= 0.5)
    )
    v2_fo = find_summary(summaries, "V2", "first_order", "NA")
    v2_primary = find_summary(summaries, "V2", "finite_step", "0.10")
    stage_lines = []
    for stage in (7, 8, 9):
        row = next(
            value for value in stages
            if value["method"] == "V1" and value["stage"] == stage
            and value["score_type"] == "finite_step" and value["rho"] == "0.10"
        )
        stage_lines.append(
            f"- Stage {stage}: ROC-AUC {row['roc_auc']:.4f}, AP {row['average_precision']:.4f}, prevalence {pct(row['harmful_prevalence'])}."
        )
    lines = [
        "# Retention Score Ranking and Calibration Diagnosis", "",
        "## Verdict", "",
        f"`{verdict}`", "",
        "## Attribution", "",
        f"`{attribution}`", "",
        "## Identity", "",
        "- Branch: `research/retention-score-ranking-calibration`",
        "- Starting SHA: `ff710880fc8c1fee68a4ff44aa63c4d1711e5aa2`",
        "- Final SHA: branch HEAD at handoff", "",
        "## Input", "",
        "- Source: `reports/finite_step_functional_probe_validity/finite_step_probe_raw.csv`",
        "- Rows: 720",
        "- Unique short cases: 180",
        "- V1/V2 cases: 90/90", "",
        "## Primary", "",
        "Method V1, finite-step rho=0.10, short transitions, leave-one-seed-out threshold calibration.",
        f"Endpoint-harmful prevalence is {pct(primary['harmful_prevalence'])}; AP is {primary['average_precision']:.4f}, or {primary['ap_gain_over_prevalence']:+.4f} above prevalence.",
        f"LOSO all-case false-safe is {pct(primary['loso_false_safe_rate_all'])}; harmful-conditional miss rate is {pct(1.0 - primary['loso_sensitivity'])}. These denominators are not interchangeable.", "",
        "## Ranking and calibration", "",
        "| Method | Score | rho | ROC-AUC | AP | Zero BA | LOSO BA | Zero FS | LOSO FS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        *main_rows, "",
        "## Horizon ranking curve", "",
        "| rho | ROC-AUC | AP | LOSO BA | LOSO FS |",
        "|---:|---:|---:|---:|---:|",
        *horizon_lines, "",
        "## Calibration held-out folds", "",
        "| Held-out seed | Threshold learned from | tau | Test BA | Sensitivity | Specificity | All-case false-safe | Harmful miss |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
        *fold_lines, "",
        "## First-order vs finite", "",
        f"- ROC-AUC: {first_order['roc_auc']:.4f} vs {primary['roc_auc']:.4f} (delta {primary['roc_auc'] - first_order['roc_auc']:+.4f}).",
        f"- AP: {first_order['average_precision']:.4f} vs {primary['average_precision']:.4f} (delta {primary['average_precision'] - first_order['average_precision']:+.4f}).",
        f"- LOSO BA: {pct(first_order['loso_balanced_accuracy'])} vs {pct(primary['loso_balanced_accuracy'])} ({pp(primary['finite_vs_first_order_loso_ba_gain_pp'])}).",
        f"- LOSO all-case false-safe: {pct(first_order['loso_false_safe_rate_all'])} vs {pct(primary['loso_false_safe_rate_all'])} (finite reduction {pp(primary['finite_vs_first_order_false_safe_reduction_pp'])}).", "",
        "## Distribution", "",
        f"- Harmful V1 rho0.10 scores: n={harmful_dist['n']}, median={float(harmful_dist['median']):.6f}, q25/q75={float(harmful_dist['q25']):.6f}/{float(harmful_dist['q75']):.6f}.",
        f"- Safe V1 rho0.10 scores: n={safe_dist['n']}, median={float(safe_dist['median']):.6f}, q25/q75={float(safe_dist['q25']):.6f}/{float(safe_dist['q75']):.6f}.", "",
        "## Secondary", "",
        f"- V2 first-order ROC-AUC/LOSO BA: {v2_fo['roc_auc']:.4f}/{pct(v2_fo['loso_balanced_accuracy'])}.",
        f"- V2 finite rho0.10 ROC-AUC/LOSO BA: {v2_primary['roc_auc']:.4f}/{pct(v2_primary['loso_balanced_accuracy'])}.",
        f"- Material-subset ranking direction is {'consistent' if material_consistent else 'inconsistent'} with the full V1 rho0.10 set (material ROC-AUC {primary_rank['material_roc_auc']:.4f}).",
        *stage_lines, "",
        "## Observed", "",
        "ROC-AUC, AP, zero-threshold classification, LOSO fold thresholds, held-out confusion metrics, distributions, and secondary slices are direct deterministic calculations from the committed CSV.", "",
        "## Inferred", "",
        f"H10 is `{verdict}` and the preregistered attribution is `{attribution}`. Oracle pooled thresholds are recorded only as descriptive upper bounds in `threshold_summary.csv` and are not verdict evidence.", "",
        "## Not established", "",
        "LOSO across three seeds is a small-sample descriptive generalization test, not a statistical significance claim. A calibrated offline threshold does not establish stability during online training. The finite-step score uses the future checkpoint chord and is not deployable online. This diagnosis concerns old BC-loss drift, not rollout-success retention.", "",
        "## Verdict", "",
        f"`{verdict}`", "",
        "## Next action", "",
        next_action, "",
    ]
    (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict, attribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path,
        default=Path("reports/finite_step_functional_probe_validity/finite_step_probe_raw.csv"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/retention_score_ranking_calibration"),
    )
    parser.add_argument("--methods", nargs="+", choices=("V1", "V2"), default=["V1", "V2"])
    parser.add_argument("--rhos", nargs="+", type=float, default=list(FORMAL_RHOS))
    args = parser.parse_args()
    rows = read_csv(args.input)
    cases = validate_input(rows)
    if any(rho not in FORMAL_RHOS for rho in args.rhos):
        raise ValueError(f"rhos must be selected from {FORMAL_RHOS}")
    specs = score_specs(rows, args.methods, args.rhos)
    for spec in specs:
        if len(spec["rows"]) != 90:
            raise ValueError(f"{spec['method']} {spec['score_type']} {spec['rho']} has {len(spec['rows'])} cases")
    rankings, stages = ranking_rows(specs)
    summaries, folds, thresholds = calibration_rows(specs)
    distributions = distribution_rows(specs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "ranking_summary.csv", rankings)
    write_csv(args.output_dir / "loso_fold_metrics.csv", folds)
    write_csv(args.output_dir / "loso_summary.csv", summaries)
    write_csv(args.output_dir / "threshold_summary.csv", thresholds)
    write_csv(args.output_dir / "stage_summary.csv", stages)
    write_csv(args.output_dir / "score_distribution_summary.csv", distributions)
    formal = set(args.methods) == {"V1", "V2"} and tuple(sorted(args.rhos)) == FORMAL_RHOS
    verdict, attribution = write_report(
        args.output_dir, summaries, folds, rankings, distributions, stages, formal
    )
    print(
        f"[complete] input_rows={len(rows)} unique_cases={len(cases)} "
        f"specs={len(specs)} loso_folds={len(folds)}"
    )
    print(f"verdict={verdict}")
    print(f"attribution={attribution}")


if __name__ == "__main__":
    main()
