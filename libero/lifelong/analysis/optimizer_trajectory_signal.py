import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from libero.lifelong.analysis.retention_score_calibration import (
    average_precision,
    binary_metrics,
    classification_metrics,
    quantile,
    roc_auc,
    select_balanced_threshold,
)


FORMAL_PREDICTORS = (
    "single_raw",
    "ema3_raw",
    "single_relative",
    "ema3_relative",
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def build_anchor_index(anchor_rows):
    index = {}
    grouped_updates = defaultdict(list)
    for row in anchor_rows:
        key = (
            int(row["seed"]),
            int(row["stage"]),
            int(row["global_update"]),
        )
        if key in index:
            raise ValueError(f"duplicate anchor {key}")
        index[key] = row
        grouped_updates[key[:2]].append(key[2])
    for group, updates in grouped_updates.items():
        if updates != sorted(updates) or len(updates) != len(set(updates)):
            raise ValueError(f"anchor updates are not strictly increasing: {group}")
    return index


def case_key(row):
    return (
        int(row["seed"]),
        int(row["stage"]),
        int(row["global_update"]),
    )


def enrich_case_with_history(case, anchor_index, spacing=100):
    seed, stage, update = case_key(case)
    current = anchor_index.get((seed, stage, update))
    if current is None:
        raise ValueError(f"case has no current anchor: {(seed, stage, update)}")
    prev1 = anchor_index.get((seed, stage, update - spacing))
    prev2 = anchor_index.get((seed, stage, update - 2 * spacing))
    if prev1 is None or prev2 is None:
        return None

    s0 = float(current["actual_step_score"])
    s1 = float(prev1["actual_step_score"])
    s2 = float(prev2["actual_step_score"])
    r0 = float(current["actual_step_relative"])
    r1 = float(prev1["actual_step_relative"])
    r2 = float(prev2["actual_step_relative"])
    delta_norm = float(current["actual_step_delta_norm"])
    old_grad_norm = float(current["actual_step_old_grad_norm"])
    enriched = {
        **case,
        "single_raw": s0,
        "ema3_raw": 0.50 * s0 + 0.25 * s1 + 0.25 * s2,
        "mean2_raw": 0.50 * (s0 + s1),
        "mean3_raw": (s0 + s1 + s2) / 3.0,
        "slope_raw": s0 - s1,
        "acceleration_raw": s0 - 2.0 * s1 + s2,
        "single_relative": r0,
        "ema3_relative": 0.50 * r0 + 0.25 * r1 + 0.25 * r2,
        "single_cosine": float(current["actual_step_cosine"]),
        "delta_norm": delta_norm,
        "old_grad_norm": old_grad_norm,
        "norm_product": delta_norm * old_grad_norm,
        "score_prev1": s1,
        "score_prev2": s2,
        "relative_prev1": r1,
        "relative_prev2": r2,
    }
    for field, value in enriched.items():
        if field in case:
            continue
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"non-finite {field} for {(seed, stage, update)}")
    return enriched


def build_trajectory_cases(case_rows, anchor_index, spacing=100):
    output = []
    for case in case_rows:
        enriched = enrich_case_with_history(case, anchor_index, spacing=spacing)
        if enriched is not None:
            output.append(enriched)
    return output


def horizon_transition_key(harmful100, harmful300):
    if not harmful100 and not harmful300:
        return "safe100_safe300"
    if harmful100 and harmful300:
        return "harmful100_harmful300"
    if not harmful100 and harmful300:
        return "safe100_harmful300"
    return "harmful100_safe300"


def _distribution(values):
    values = [float(value) for value in values]
    if not values:
        return {key: float("nan") for key in ("mean", "median", "q25", "q75", "min", "max")}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "q25": quantile(values, 0.25),
        "q75": quantile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def build_horizon_transition_summary(h100_rows, h300_rows):
    by100 = {case_key(row): row for row in h100_rows}
    by300 = {case_key(row): row for row in h300_rows}
    paired_keys = sorted(set(by100) & set(by300))
    paired = []
    for key in paired_keys:
        row100 = by100[key]
        row300 = by300[key]
        harmful100 = bool(int(row100["endpoint_harmful"]))
        harmful300 = bool(int(row300["endpoint_harmful"]))
        paired.append(
            {
                **row100,
                "transition": horizon_transition_key(harmful100, harmful300),
                "harmful100": int(harmful100),
                "harmful300": int(harmful300),
            }
        )

    output = []
    scopes = [("ALL", paired)] + [
        (f"seed{seed}", [row for row in paired if int(row["seed"]) == seed])
        for seed in sorted({int(row["seed"]) for row in paired})
    ]
    transitions = (
        "safe100_safe300",
        "harmful100_harmful300",
        "safe100_harmful300",
        "harmful100_safe300",
    )
    for scope, rows in scopes:
        counts = {name: sum(row["transition"] == name for row in rows) for name in transitions}
        delayed = [row for row in rows if row["transition"] == "safe100_harmful300"]
        h100_safe = counts["safe100_safe300"] + counts["safe100_harmful300"]
        h100_harmful = counts["harmful100_harmful300"] + counts["harmful100_safe300"]
        result = {
                "scope": scope,
                "paired_cases": len(rows),
                **counts,
                "delayed_harmful_rate_all": len(delayed) / len(rows) if rows else float("nan"),
                "delayed_harmful_rate_among_h100_safe": len(delayed) / h100_safe if h100_safe else float("nan"),
                "recovery_rate_among_h100_harmful": counts["harmful100_safe300"] / h100_harmful if h100_harmful else float("nan"),
                "single_raw_median_delayed": statistics.median(float(row["single_raw"]) for row in delayed) if delayed else float("nan"),
                "ema3_raw_median_delayed": statistics.median(float(row["ema3_raw"]) for row in delayed) if delayed else float("nan"),
        }
        for transition in transitions:
            selected = [row for row in rows if row["transition"] == transition]
            for feature in ("single_raw", "ema3_raw"):
                distribution = _distribution(float(row[feature]) for row in selected)
                for statistic in ("median", "q25", "q75"):
                    result[f"{feature}_{statistic}_{transition}"] = distribution[statistic]
        output.append(result)
    return output, paired


def ranking_metrics(rows, score_field):
    scores = [float(row[score_field]) for row in rows]
    labels = [bool(int(row["endpoint_harmful"])) for row in rows]
    prevalence = sum(labels) / len(labels) if labels else float("nan")
    return {
        "cases": len(rows),
        "harmful_prevalence": prevalence,
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
    }


def run_loso(rows, score_field):
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) != 3:
        raise ValueError(f"expected three seeds, found {seeds}")
    folds = []
    predictions = []
    for heldout_seed in seeds:
        train = [row for row in rows if int(row["seed"]) != heldout_seed]
        test = [row for row in rows if int(row["seed"]) == heldout_seed]
        threshold_result = select_balanced_threshold(
            [float(row[score_field]) for row in train],
            [bool(int(row["endpoint_harmful"])) for row in train],
        )
        threshold = threshold_result["threshold"]
        test_metrics = classification_metrics(
            [float(row[score_field]) for row in test],
            [bool(int(row["endpoint_harmful"])) for row in test],
            threshold,
        )
        stage_metrics = {}
        for stage in (8, 9):
            selected = [row for row in test if int(row["stage"]) == stage]
            if selected:
                metrics = classification_metrics(
                    [float(row[score_field]) for row in selected],
                    [bool(int(row["endpoint_harmful"])) for row in selected],
                    threshold,
                )
            else:
                metrics = {"balanced_accuracy": float("nan")}
            stage_metrics[stage] = metrics
        fold = {
            "heldout_seed": heldout_seed,
            "train_seeds": "+".join(str(seed) for seed in seeds if seed != heldout_seed),
            "threshold": threshold,
            "train_BA": threshold_result["metrics"]["balanced_accuracy"],
            "test_cases": len(test),
            "test_BA": test_metrics["balanced_accuracy"],
            "test_sensitivity": test_metrics["sensitivity"],
            "test_specificity": test_metrics["specificity"],
            "test_false_safe": test_metrics["false_safe_rate_all"],
            "test_false_alarm": test_metrics["false_alarm_rate_all"],
            "test_BA_stage8": stage_metrics[8]["balanced_accuracy"],
            "test_BA_stage9": stage_metrics[9]["balanced_accuracy"],
        }
        folds.append(fold)
        for row in test:
            predictions.append(
                {
                    **row,
                    "threshold": threshold,
                    "predicted_harmful": int(float(row[score_field]) > threshold),
                }
            )
    pooled = binary_metrics(
        [bool(int(row["predicted_harmful"])) for row in predictions],
        [bool(int(row["endpoint_harmful"])) for row in predictions],
    )
    return pooled, folds, predictions


def summarize_seed_stage(rows, score_fields, loso_predictions=None):
    loso_predictions = loso_predictions or {}
    output = []
    for seed in sorted({int(row["seed"]) for row in rows}):
        for stage in sorted({int(row["stage"]) for row in rows}):
            selected = [
                row for row in rows
                if int(row["seed"]) == seed and int(row["stage"]) == stage
            ]
            if not selected:
                continue
            for feature in score_fields:
                ranking = ranking_metrics(selected, feature)
                prediction_rows = [
                    row for row in loso_predictions.get(feature, [])
                    if int(row["seed"]) == seed and int(row["stage"]) == stage
                ]
                if prediction_rows:
                    classification = binary_metrics(
                        [bool(int(row["predicted_harmful"])) for row in prediction_rows],
                        [bool(int(row["endpoint_harmful"])) for row in prediction_rows],
                    )
                else:
                    classification = {}
                output.append(
                    {
                        "seed": seed,
                        "stage": stage,
                        "feature": feature,
                        **ranking,
                        "loso_test_ba": classification.get("balanced_accuracy", ""),
                        "loso_test_sensitivity": classification.get("sensitivity", ""),
                        "loso_test_specificity": classification.get("specificity", ""),
                    }
                )
    return output


def summarize_scale_controls(rows):
    features = {
        "single_raw": "actual_step_score",
        "single_cosine": "actual_step_cosine",
        "single_relative": "actual_step_relative",
        "delta_norm": "delta_norm",
        "old_grad_norm": "old_grad_norm",
        "norm_product": "norm_product",
    }
    raw_auc = ranking_metrics(rows, "single_raw")["roc_auc"]
    output = []
    for feature, output_name in features.items():
        metrics = ranking_metrics(rows, feature)
        output.append(
            {
                "feature": output_name,
                **metrics,
                "delta_auc_vs_actual_step_score": metrics["roc_auc"] - raw_auc,
            }
        )
    return output


def score_distribution(rows, features):
    output = []
    for feature in features:
        for label_name, harmful in (("safe", False), ("harmful", True)):
            selected = [
                float(row[feature]) for row in rows
                if bool(int(row["endpoint_harmful"])) == harmful
            ]
            output.append(
                {
                    "feature": feature,
                    "endpoint_label": label_name,
                    "n": len(selected),
                    **_distribution(selected),
                }
            )
    return output
