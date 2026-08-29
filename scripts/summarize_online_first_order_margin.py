#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import Counter
from pathlib import Path

from libero.lifelong.analysis.online_first_order_margin import (
    build_future_cases,
    incremental_margin_counts,
)
from libero.lifelong.analysis.retention_score_calibration import (
    binary_metrics,
    classification_metrics,
    loso_predict,
    pooled_loso_metrics,
    quantile,
    ranking_metrics,
)


SCORES = (
    "native_raw_max",
    "native_cos_max",
    "actual_step_score",
    "actual_step_relative",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def merge_raw(raw_dir, suffix):
    paths = sorted(Path(raw_dir).glob(f"seed*_stage*_{suffix}.csv"))
    if len(paths) != 6:
        raise ValueError(f"expected six {suffix} files, found {len(paths)}")
    rows = []
    for path in paths:
        selected = read_csv(path)
        if not selected:
            raise ValueError(f"empty raw file: {path}")
        rows.extend(selected)
    return rows


def validate_anchors(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault((int(row["seed"]), int(row["stage"])), []).append(row)
    if set(grouped) != {
        (seed, stage) for seed in (42, 43, 44) for stage in (8, 9)
    }:
        raise ValueError(f"unexpected seed-stage groups: {sorted(grouped)}")
    for key, selected in grouped.items():
        updates = [int(row["global_update"]) for row in selected]
        if updates != sorted(updates) or len(updates) != len(set(updates)):
            raise ValueError(f"non-monotonic or duplicate anchor updates: {key}")
        if any(right - left != 100 for left, right in zip(updates[:-1], updates[1:])):
            raise ValueError(f"anchor interval is not 100: {key}")


def summarize_score(cases, horizon, score_type, total_cases):
    selected = [row for row in cases if row[score_type] != ""]
    ranking = ranking_metrics(selected, score_type)
    scores = [float(row[score_type]) for row in selected]
    labels = [float(row["actual_delta"]) > 0.0 for row in selected]
    zero = classification_metrics(scores, labels, 0.0)
    predictions, folds = loso_predict(selected, score_type)
    loso = pooled_loso_metrics(predictions)
    return {
        "horizon": horizon,
        "score_type": score_type,
        "scope": "ALL",
        "cases": len(selected),
        "candidate_coverage": len(selected) / total_cases,
        **ranking,
        "zero_balanced_accuracy": zero["balanced_accuracy"],
        "zero_false_safe": zero["false_safe_rate_all"],
        "zero_false_alarm": zero["false_alarm_rate_all"],
        "loso_balanced_accuracy": loso["balanced_accuracy"],
        "loso_false_safe": loso["false_safe_rate_all"],
        "loso_false_alarm": loso["false_alarm_rate_all"],
        "loso_sensitivity": loso["sensitivity"],
        "loso_specificity": loso["specificity"],
        "ba_gain_pp": 100.0 * (loso["balanced_accuracy"] - zero["balanced_accuracy"]),
        "false_safe_reduction_pp": 100.0 * (
            zero["false_safe_rate_all"] - loso["false_safe_rate_all"]
        ),
        "heldout_seeds_ba_ge_0_60": sum(
            fold["test_balanced_accuracy"] >= 0.60 for fold in folds
        ),
    }, predictions, folds


def fold_output(horizon, score_type, predictions, folds):
    output = []
    for fold in folds:
        heldout = int(fold["heldout_seed"])
        test_predictions = [
            row for row in predictions if int(row["heldout_seed"]) == heldout
        ]
        margin = None
        if score_type == "native_raw_max":
            margin = incremental_margin_counts(
                test_predictions, {heldout: float(fold["threshold"])}
            )
        output.append(
            {
                "horizon": horizon,
                "score_type": score_type,
                "heldout_seed": heldout,
                "train_seeds": fold["train_seeds"],
                "threshold": fold["threshold"],
                "train_BA": fold["train_balanced_accuracy"],
                "test_cases": fold["test_cases"],
                "test_BA": fold["test_balanced_accuracy"],
                "test_sensitivity": fold["test_sensitivity"],
                "test_specificity": fold["test_specificity"],
                "test_false_safe": fold["test_false_safe_rate_all"],
                "test_false_alarm": fold["test_false_alarm_rate_all"],
                "currently_ungated": "" if margin is None else margin["currently_ungated"],
                "currently_ungated_harmful": "" if margin is None else margin["currently_ungated_harmful"],
                "newly_flagged": "" if margin is None else margin["newly_flagged"],
                "newly_flagged_harmful": "" if margin is None else margin["newly_flagged_harmful"],
                "incremental_harmful_recall": "" if margin is None else margin["incremental_harmful_recall"],
                "incremental_false_alarms": "" if margin is None else margin["incremental_false_alarms"],
                "harmful_still_missed_after_margin": "" if margin is None else margin["harmful_still_missed_after_margin"],
            }
        )
    return output


def distribution_summary(cases, horizon):
    output = []
    for score_type in SCORES:
        available = [row for row in cases if row[score_type] != ""]
        for seed in (42, 43, 44):
            for stage in (8, 9):
                for label_name, harmful in (("safe", False), ("harmful", True)):
                    values = [
                        float(row[score_type])
                        for row in available
                        if int(row["seed"]) == seed and int(row["stage"]) == stage
                        and (float(row["actual_delta"]) > 0.0) == harmful
                    ]
                    if not values:
                        continue
                    output.append(
                        {
                            "horizon": horizon,
                            "score_type": score_type,
                            "seed": seed,
                            "stage": stage,
                            "endpoint_label": label_name,
                            "n": len(values),
                            "mean": statistics.mean(values),
                            "median": statistics.median(values),
                            "q25": quantile(values, 0.25),
                            "q75": quantile(values, 0.75),
                            "min": min(values),
                            "max": max(values),
                        }
                    )
    return output


def stage_summary(cases, horizon, prediction_cache):
    output = []
    for score_type in SCORES:
        predictions = prediction_cache[(horizon, score_type)]
        for stage in (8, 9):
            selected = [
                row for row in cases
                if row[score_type] != "" and int(row["stage"]) == stage
            ]
            ranking = ranking_metrics(selected, score_type)
            stage_predictions = [
                row for row in predictions if int(row["stage"]) == stage
            ]
            metrics = binary_metrics(
                [bool(int(row["predicted_harmful"])) for row in stage_predictions],
                [bool(int(row["endpoint_harmful"])) for row in stage_predictions],
            )
            output.append(
                {
                    "horizon": horizon,
                    "stage": stage,
                    "score_type": score_type,
                    **ranking,
                    "loso_balanced_accuracy": metrics["balanced_accuracy"],
                    "loso_false_safe": metrics["false_safe_rate_all"],
                }
            )
    return output


def find_summary(rows, horizon, score_type):
    return next(
        row for row in rows
        if int(row["horizon"]) == int(horizon) and row["score_type"] == score_type
    )


def pct(value):
    return f"{100.0 * float(value):.2f}%"


def pp(value):
    return f"{float(value):+.2f} pp"


def write_report(output_dir, anchors, cases100, cases300, summaries, folds, stages):
    raw = find_summary(summaries, 100, "native_raw_max")
    cosine = find_summary(summaries, 100, "native_cos_max")
    step = find_summary(summaries, 100, "actual_step_score")
    raw300 = find_summary(summaries, 300, "native_raw_max")
    cosine300 = find_summary(summaries, 300, "native_cos_max")
    step300 = find_summary(summaries, 300, "actual_step_score")
    anchors_with_candidates = sum(int(row["native_candidate_bins"]) > 0 for row in anchors)
    anchor_coverage = anchors_with_candidates / len(anchors)
    per_seed_cases = Counter(
        int(row["seed"]) for row in cases100 if row["native_raw_max"] != ""
    )

    def supported(row):
        return (
            row["candidate_coverage"] >= 0.80
            and row["roc_auc"] >= 0.70
            and row["loso_balanced_accuracy"] >= 0.65
            and row["ba_gain_pp"] >= 10.0
            and row["false_safe_reduction_pp"] >= 10.0
            and row["heldout_seeds_ba_ge_0_60"] >= 2
            and row["cases"] >= 150
            and all(per_seed_cases[seed] >= 40 for seed in (42, 43, 44))
        )

    sufficient = (
        raw["cases"] >= 150
        and all(per_seed_cases[seed] >= 40 for seed in (42, 43, 44))
    )
    if supported(raw):
        verdict = "ONLINE_FIRST_ORDER_MARGIN_VALIDITY_SUPPORTED"
        attribution = "RAW_TEMPORAL_MARGIN_TRANSPORTS"
    elif supported(cosine):
        verdict = "ONLINE_FIRST_ORDER_MARGIN_VALIDITY_SUPPORTED"
        attribution = "COSINE_TEMPORAL_MARGIN_REQUIRED"
    elif (
        step["roc_auc"] >= 0.70
        and step["loso_balanced_accuracy"] >= 0.65
        and step["heldout_seeds_ba_ge_0_60"] >= 2
    ):
        verdict = "ONLINE_FIRST_ORDER_MARGIN_VALIDITY_INCONCLUSIVE"
        attribution = "OPTIMIZER_AWARE_SIGNAL_ONLY"
    elif (
        raw["roc_auc"] < 0.60 and raw["loso_balanced_accuracy"] < 0.58
        and cosine["roc_auc"] < 0.60 and cosine["loso_balanced_accuracy"] < 0.58
        and step["roc_auc"] < 0.65 and step["loso_balanced_accuracy"] < 0.60
        and sufficient
    ):
        verdict = "ONLINE_FIRST_ORDER_MARGIN_VALIDITY_NOT_SUPPORTED"
        attribution = "ONLINE_MARGIN_SIGNAL_NOT_ESTABLISHED"
    else:
        verdict = "ONLINE_FIRST_ORDER_MARGIN_VALIDITY_INCONCLUSIVE"
        attribution = "ONLINE_MARGIN_SIGNAL_NOT_ESTABLISHED"

    raw_folds = [
        row for row in folds
        if int(row["horizon"]) == 100 and row["score_type"] == "native_raw_max"
    ]
    thresholds = {int(row["heldout_seed"]): float(row["threshold"]) for row in raw_folds}
    incremental = incremental_margin_counts(
        [row for row in cases100 if row["native_raw_max"] != ""], thresholds
    )
    if attribution == "RAW_TEMPORAL_MARGIN_TRANSPORTS":
        next_action = "Implement Margin-Aware Temporal Robot-IGFA from Temporal V1, changing only risk > 0 to risk > calibrated tau."
    elif attribution == "COSINE_TEMPORAL_MARGIN_REQUIRED":
        next_action = "Implement Cosine-Margin Temporal Robot-IGFA from Temporal V1 using the calibrated cosine margin only."
    elif attribution == "OPTIMIZER_AWARE_SIGNAL_ONLY":
        next_action = "Test an optimizer-aware global retention guard before changing the native temporal gate."
    elif verdict == "ONLINE_FIRST_ORDER_MARGIN_VALIDITY_NOT_SUPPORTED":
        next_action = "Start realized training trajectory and old-loss-slope diagnosis; close the local online-margin bridge."
    else:
        next_action = "Resolve cross-seed or horizon instability before any margin intervention."

    fold_lines = [
        f"| {row['heldout_seed']} | {row['train_seeds']} | {float(row['threshold']):.6g} | "
        f"{pct(row['test_BA'])} | {pct(row['test_sensitivity'])} | "
        f"{pct(row['test_specificity'])} | {pct(row['test_false_safe'])} |"
        for row in raw_folds
    ]
    stage_lines = []
    for stage in (8, 9):
        parts = []
        for score_type in ("native_raw_max", "native_cos_max", "actual_step_score"):
            row = next(
                value for value in stages
                if int(value["horizon"]) == 100 and int(value["stage"]) == stage
                and value["score_type"] == score_type
            )
            parts.append(
                f"{score_type} AUC/BA={row['roc_auc']:.4f}/{pct(row['loso_balanced_accuracy'])}"
            )
        stage_lines.append(f"- Stage {stage}: " + "; ".join(parts) + ".")
    delayed = (
        raw300["roc_auc"] >= 0.70 and raw300["loso_balanced_accuracy"] >= 0.65
    )
    lines = [
        "# Online First-Order Margin Validity Diagnosis", "",
        "## Identity", "",
        f"- Verdict: `{verdict}`",
        f"- Attribution: `{attribution}`",
        "- Branch: `research/online-first-order-margin-validity`",
        "- Starting SHA: `d725842ed792e40b30d1965a395315829c29256b`", "",
        "## Protocol", "",
        "Temporal V1 targeted continuations used seeds 42/43/44, stages 8/9, 15 epochs, batch size 10, replay rate 0.5, five bins, gate strength 1.0, and IFA disabled. Entry checkpoints were previous-stage ep95 proxies. The first two epochs were burn-in; anchors were sampled every 100 optimizer updates. Label and score-gradient probes used 256 and 64 fixed records per old task.", "",
        "## Training parity", "",
        "Instrumentation parity passed before formal execution. The shadow score restores CPU/CUDA RNG and train/eval mode, while real updates call the committed `MLRTemporalRobotIGFA.observe()` directly. Optimizer and scheduler semantics match the release, including 100-epoch total-step construction and epoch-end scheduler stepping.", "",
        "## Coverage", "",
        f"- All anchors: {len(anchors)}",
        f"- Anchors with native candidate bins: {anchors_with_candidates}",
        f"- Anchor candidate coverage: {pct(anchor_coverage)}",
        f"- H100 native cases: {raw['cases']} of {len(cases100)} ({pct(raw['candidate_coverage'])})",
        f"- H100 native cases per seed: {dict(sorted(per_seed_cases.items()))}", "",
        "## Primary native raw", "",
        f"- Harmful prevalence: {pct(raw['harmful_prevalence'])}",
        f"- ROC-AUC / AP: {raw['roc_auc']:.4f} / {raw['average_precision']:.4f}",
        f"- Threshold0 BA / false-safe / false-alarm: {pct(raw['zero_balanced_accuracy'])} / {pct(raw['zero_false_safe'])} / {pct(raw['zero_false_alarm'])}",
        f"- LOSO BA / false-safe / false-alarm: {pct(raw['loso_balanced_accuracy'])} / {pct(raw['loso_false_safe'])} / {pct(raw['loso_false_alarm'])}",
        f"- BA gain / false-safe reduction: {pp(raw['ba_gain_pp'])} / {pp(raw['false_safe_reduction_pp'])}",
        f"- Held-out seeds BA>=0.60: {raw['heldout_seeds_ba_ge_0_60']}/3", "",
        "| Held-out seed | Train seeds | tau | Test BA | Sensitivity | Specificity | False-safe |",
        "|---:|---|---:|---:|---:|---:|---:|",
        *fold_lines, "",
        "## Native cosine", "",
        f"ROC-AUC/AP {cosine['roc_auc']:.4f}/{cosine['average_precision']:.4f}; LOSO BA {pct(cosine['loso_balanced_accuracy'])}; false-safe reduction {pp(cosine['false_safe_reduction_pp'])}.", "",
        "## Actual optimizer-step bridge", "",
        f"ROC-AUC/AP {step['roc_auc']:.4f}/{step['average_precision']:.4f}; LOSO BA {pct(step['loso_balanced_accuracy'])}; LOSO false-safe {pct(step['loso_false_safe'])}.", "",
        "## Incremental margin analysis", "",
        f"- Currently ungated / harmful: {incremental['currently_ungated']} / {incremental['currently_ungated_harmful']}",
        f"- Newly flagged / harmful: {incremental['newly_flagged']} / {incremental['newly_flagged_harmful']}",
        f"- Incremental harmful recall: {pct(incremental['incremental_harmful_recall'])}",
        f"- Incremental false alarms: {incremental['incremental_false_alarms']}",
        f"- Harmful still missed: {incremental['harmful_still_missed_after_margin']}", "",
        "## H300 secondary", "",
        f"- Native raw AUC/LOSO BA: {raw300['roc_auc']:.4f}/{pct(raw300['loso_balanced_accuracy'])}",
        f"- Native cosine AUC/LOSO BA: {cosine300['roc_auc']:.4f}/{pct(cosine300['loso_balanced_accuracy'])}",
        f"- Actual-step AUC/LOSO BA: {step300['roc_auc']:.4f}/{pct(step300['loso_balanced_accuracy'])}",
        f"- Delayed raw signal criterion: {'met' if delayed else 'not met'}", "",
        "## Stage8/9", "",
        *stage_lines, "",
        "## Observed", "",
        "Anchor losses, native pre-gate risks, realized optimizer-step scores, future loss deltas, ranking metrics, LOSO folds, and incremental margin counts are direct outputs of the targeted continuations.", "",
        "## Inferred", "",
        f"H11 is `{verdict}` with preregistered attribution `{attribution}`.", "",
        "## Not established", "",
        "Ep95 is a late-stage entry proxy, not the exact ep99 boundary. Continuations are distribution-matched, not exact historical minibatch replay. LOSO over three seeds is descriptive rather than a significance claim. BC-loss retention is not rollout-success retention, and diagnostic tau is not yet a validated formal training hyperparameter.", "",
        "## Verdict", "",
        f"`{verdict}`", "",
        "## Attribution", "",
        f"`{attribution}`", "",
        "## Next action", "",
        next_action, "",
    ]
    (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return verdict, attribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("reports/online_first_order_margin_validity"),
    )
    args = parser.parse_args()
    raw_dir = args.report_dir / "raw"
    anchors = merge_raw(raw_dir, "anchor")
    task_anchors = merge_raw(raw_dir, "task")
    validate_anchors(anchors)
    cases100 = build_future_cases(anchors, 100)
    cases300 = build_future_cases(anchors, 300)
    write_csv(args.report_dir / "anchor_raw.csv", anchors)
    write_csv(args.report_dir / "task_anchor_raw.csv", task_anchors)
    write_csv(args.report_dir / "cases_h100.csv", cases100)
    write_csv(args.report_dir / "cases_h300.csv", cases300)
    summaries = []
    folds = []
    prediction_cache = {}
    for horizon, cases in ((100, cases100), (300, cases300)):
        for score_type in SCORES:
            summary, predictions, score_folds = summarize_score(
                cases, horizon, score_type, len(cases)
            )
            summaries.append(summary)
            prediction_cache[(horizon, score_type)] = predictions
            folds.extend(fold_output(horizon, score_type, predictions, score_folds))
    distributions = distribution_summary(cases100, 100) + distribution_summary(cases300, 300)
    stages = stage_summary(cases100, 100, prediction_cache) + stage_summary(
        cases300, 300, prediction_cache
    )
    write_csv(args.report_dir / "ranking_summary.csv", summaries)
    write_csv(args.report_dir / "loso_fold_metrics.csv", folds)
    write_csv(args.report_dir / "score_distribution_summary.csv", distributions)
    write_csv(args.report_dir / "stage_summary.csv", stages)
    verdict, attribution = write_report(
        args.report_dir, anchors, cases100, cases300, summaries, folds, stages
    )
    print(
        f"[summary] anchors={len(anchors)} task_anchors={len(task_anchors)} "
        f"h100={len(cases100)} h300={len(cases300)}"
    )
    print(f"verdict={verdict}")
    print(f"attribution={attribution}")


if __name__ == "__main__":
    main()
