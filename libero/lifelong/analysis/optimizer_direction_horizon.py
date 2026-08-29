import csv
import math
import statistics
from pathlib import Path

from libero.lifelong.analysis.retention_score_calibration import (
    average_precision,
    binary_metrics,
    classification_metrics,
    quantile,
    roc_auc,
    select_balanced_threshold,
)


KEY_FIELDS = ("seed", "stage", "global_update")


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def row_key(row):
    return tuple(int(row[field]) for field in KEY_FIELDS)


def build_unique_index(rows):
    index = {}
    for row in rows:
        key = row_key(row)
        if key in index:
            raise ValueError(f"duplicate key: {key}")
        index[key] = row
    return index


def enrich_case(case, anchor_index):
    key = row_key(case)
    if key not in anchor_index:
        raise ValueError(f"case key absent from anchors: {key}")
    anchor = anchor_index[key]
    for field in (
        "actual_step_score", "actual_step_cosine", "actual_step_relative"
    ):
        if float(case[field]) != float(anchor[field]):
            raise ValueError(f"case/anchor mismatch for {key}: {field}")
    delta_norm = float(anchor["actual_step_delta_norm"])
    grad_norm = float(anchor["actual_step_old_grad_norm"])
    values = (
        float(case["actual_step_score"]),
        float(case["actual_step_cosine"]),
        float(case["actual_step_relative"]),
        float(case["actual_delta"]),
        delta_norm,
        grad_norm,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"non-finite case values: {key}")
    return {
        **case,
        "delta_norm": delta_norm,
        "old_grad_norm": grad_norm,
        "norm_product": delta_norm * grad_norm,
    }


def ranking_metrics(rows, score_field, label_fn=None):
    label_fn = label_fn or (lambda row: float(row["actual_delta"]) > 0.0)
    scores = [float(row[score_field]) for row in rows]
    labels = [bool(label_fn(row)) for row in rows]
    prevalence = sum(labels) / len(labels) if labels else float("nan")
    return {
        "cases": len(rows),
        "positive_cases": sum(labels),
        "positive_prevalence": prevalence,
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
    }


def run_loso(rows, score_field, label_fn=None):
    label_fn = label_fn or (lambda row: float(row["actual_delta"]) > 0.0)
    seeds = sorted({int(row["seed"]) for row in rows})
    if seeds != [42, 43, 44]:
        raise ValueError(f"expected seeds 42/43/44, found {seeds}")
    fold_rows = []
    predictions = []
    for heldout in seeds:
        train = [row for row in rows if int(row["seed"]) != heldout]
        test = [row for row in rows if int(row["seed"]) == heldout]
        train_labels = [bool(label_fn(row)) for row in train]
        test_labels = [bool(label_fn(row)) for row in test]
        valid = (
            bool(train) and bool(test)
            and len(set(train_labels)) == 2
            and len(set(test_labels)) == 2
        )
        if not valid:
            fold_rows.append(
                {
                    "heldout_seed": heldout,
                    "train_seeds": "+".join(str(seed) for seed in seeds if seed != heldout),
                    "threshold": "NA",
                    "train_BA": "NA",
                    "test_cases": len(test),
                    "test_BA": "NA",
                    "test_sensitivity": "NA",
                    "test_specificity": "NA",
                    "test_false_safe": "NA",
                    "test_false_alarm": "NA",
                    "valid_fold": 0,
                }
            )
            continue
        selected = select_balanced_threshold(
            [float(row[score_field]) for row in train], train_labels
        )
        threshold = selected["threshold"]
        metrics = classification_metrics(
            [float(row[score_field]) for row in test], test_labels, threshold
        )
        fold_rows.append(
            {
                "heldout_seed": heldout,
                "train_seeds": "+".join(str(seed) for seed in seeds if seed != heldout),
                "threshold": threshold,
                "train_BA": selected["metrics"]["balanced_accuracy"],
                "test_cases": len(test),
                "test_BA": metrics["balanced_accuracy"],
                "test_sensitivity": metrics["sensitivity"],
                "test_specificity": metrics["specificity"],
                "test_false_safe": metrics["false_safe_rate_all"],
                "test_false_alarm": metrics["false_alarm_rate_all"],
                "valid_fold": 1,
            }
        )
        for row, label in zip(test, test_labels):
            predictions.append(
                {
                    **row,
                    "heldout_seed": heldout,
                    "threshold": threshold,
                    "predicted_harmful": int(float(row[score_field]) > threshold),
                    "target_positive": int(label),
                }
            )
    pooled = None
    if predictions and len({bool(int(row["target_positive"])) for row in predictions}) == 2:
        pooled = binary_metrics(
            [bool(int(row["predicted_harmful"])) for row in predictions],
            [bool(int(row["target_positive"])) for row in predictions],
        )
    return pooled, fold_rows, predictions


def transition_class(h100, h300):
    harmful100 = float(h100["actual_delta"]) > 0.0
    harmful300 = float(h300["actual_delta"]) > 0.0
    if not harmful100 and not harmful300:
        return "SS"
    if harmful100 and harmful300:
        return "HH"
    if not harmful100 and harmful300:
        return "SH"
    return "HS"


def pair_horizons(h100_rows, h300_rows, anchor_index):
    h100_index = build_unique_index(h100_rows)
    h300_index = build_unique_index(h300_rows)
    output = []
    for key in sorted(set(h100_index) & set(h300_index)):
        h100 = h100_index[key]
        h300 = h300_index[key]
        anchor = anchor_index[key]
        delta_norm = float(anchor["actual_step_delta_norm"])
        grad_norm = float(anchor["actual_step_old_grad_norm"])
        output.append(
            {
                "seed": key[0],
                "stage": key[1],
                "global_update": key[2],
                "actual_delta_h100": h100["actual_delta"],
                "actual_delta_h300": h300["actual_delta"],
                "harmful_h100": int(float(h100["actual_delta"]) > 0.0),
                "harmful_h300": int(float(h300["actual_delta"]) > 0.0),
                "transition": transition_class(h100, h300),
                "actual_step_score": anchor["actual_step_score"],
                "actual_step_cosine": anchor["actual_step_cosine"],
                "actual_step_relative": anchor["actual_step_relative"],
                "delta_norm": delta_norm,
                "old_grad_norm": grad_norm,
                "norm_product": delta_norm * grad_norm,
            }
        )
    return output


def reversal_summary(paired_rows):
    scopes = [("ALL", paired_rows)]
    scopes.extend(
        (f"seed{seed}", [row for row in paired_rows if int(row["seed"]) == seed])
        for seed in (42, 43, 44)
    )
    scopes.extend(
        (f"stage{stage}", [row for row in paired_rows if int(row["stage"]) == stage])
        for stage in (8, 9)
    )
    output = []
    def ratio(numerator, denominator):
        return numerator / denominator if denominator else float("nan")

    for scope, rows in scopes:
        counts = {name: sum(row["transition"] == name for row in rows) for name in ("SS", "HH", "SH", "HS")}
        total = len(rows)
        h100_safe = counts["SS"] + counts["SH"]
        h100_harmful = counts["HH"] + counts["HS"]
        output.append(
            {
                "scope": scope,
                "paired_cases": total,
                **counts,
                "flip_count": counts["SH"] + counts["HS"],
                "flip_rate": ratio(counts["SH"] + counts["HS"], total),
                "delayed_harmful_rate_all": ratio(counts["SH"], total),
                "delayed_harmful_rate_among_h100_safe": ratio(counts["SH"], h100_safe),
                "recovery_rate_all": ratio(counts["HS"], total),
                "recovery_rate_among_h100_harmful": ratio(counts["HS"], h100_harmful),
            }
        )
    return output


def subgroup_rows(paired_rows, task):
    if task == "stable_endpoint":
        allowed, positive = {"SS", "HH"}, "HH"
    elif task == "delayed_harmful":
        allowed, positive = {"SS", "SH"}, "SH"
    elif task == "persistent_harmful":
        allowed, positive = {"HS", "HH"}, "HH"
    else:
        raise ValueError(f"unknown subgroup task: {task}")
    return [
        {**row, "subgroup_positive": int(row["transition"] == positive)}
        for row in paired_rows if row["transition"] in allowed
    ]


def summarize_seed_stage(rows, features, horizon, cosine_predictions):
    output = []
    for seed in (42, 43, 44):
        for stage in (8, 9):
            selected = [
                row for row in rows
                if int(row["seed"]) == seed and int(row["stage"]) == stage
            ]
            heldout = [
                row for row in cosine_predictions
                if int(row["seed"]) == seed and int(row["stage"]) == stage
            ]
            heldout_metrics = binary_metrics(
                [bool(int(row["predicted_harmful"])) for row in heldout],
                [bool(int(row["target_positive"])) for row in heldout],
            )
            for feature in features:
                ranking = ranking_metrics(selected, feature)
                output.append(
                    {
                        "horizon": horizon,
                        "seed": seed,
                        "stage": stage,
                        "feature": feature,
                        "cases": ranking["cases"],
                        "harmful_prevalence": ranking["positive_prevalence"],
                        "roc_auc": ranking["roc_auc"],
                        "average_precision": ranking["average_precision"],
                        "test_BA": heldout_metrics["balanced_accuracy"] if feature == "actual_step_cosine" else "",
                        "test_sensitivity": heldout_metrics["sensitivity"] if feature == "actual_step_cosine" else "",
                        "test_specificity": heldout_metrics["specificity"] if feature == "actual_step_cosine" else "",
                    }
                )
    return output


def distribution_summary(rows, features):
    output = []
    for transition in ("SS", "SH", "HH", "HS"):
        selected_rows = [row for row in rows if row["transition"] == transition]
        for feature in features:
            values = [float(row[feature]) for row in selected_rows]
            output.append(
                {
                    "transition": transition,
                    "feature": feature,
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
