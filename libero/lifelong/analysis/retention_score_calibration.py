import math


CASE_KEY = (
    "method", "seed", "stage", "old_task", "start_epoch", "end_epoch",
    "interval_type",
)


def average_ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def roc_auc(scores, labels):
    scores = [float(value) for value in scores]
    labels = [bool(value) for value in labels]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    positive_rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (
        positive_rank_sum - n_pos * (n_pos + 1) / 2.0
    ) / (n_pos * n_neg)


def average_precision(scores, labels):
    pairs = sorted(
        zip([float(value) for value in scores], [bool(value) for value in labels]),
        key=lambda item: item[0],
        reverse=True,
    )
    total_positive = sum(label for _, label in pairs)
    if total_positive == 0:
        return float("nan")
    grouped = []
    start = 0
    while start < len(pairs):
        score = pairs[start][0]
        end = start + 1
        while end < len(pairs) and pairs[end][0] == score:
            end += 1
        labels_at_score = [label for _, label in pairs[start:end]]
        grouped.append((score, len(labels_at_score), sum(labels_at_score)))
        start = end
    seen = 0
    true_positive = 0
    previous_recall = 0.0
    result = 0.0
    for _, group_size, group_positive in grouped:
        seen += group_size
        true_positive += group_positive
        recall = true_positive / total_positive
        precision = true_positive / seen
        result += (recall - previous_recall) * precision
        previous_recall = recall
    return result


def classify_with_threshold(scores, threshold):
    return [float(score) > float(threshold) for score in scores]


def binary_metrics(predictions, labels):
    predictions = [bool(value) for value in predictions]
    labels = [bool(value) for value in labels]
    if len(predictions) != len(labels) or not labels:
        raise ValueError("invalid prediction/label lengths")
    tp = sum(prediction and label for prediction, label in zip(predictions, labels))
    tn = sum(
        (not prediction) and (not label)
        for prediction, label in zip(predictions, labels)
    )
    fp = sum(
        prediction and (not label)
        for prediction, label in zip(predictions, labels)
    )
    fn = sum(
        (not prediction) and label
        for prediction, label in zip(predictions, labels)
    )
    harmful = tp + fn
    safe = tn + fp
    sensitivity = tp / harmful if harmful else float("nan")
    specificity = tn / safe if safe else float("nan")
    balanced = (
        0.5 * (sensitivity + specificity) if harmful and safe else float("nan")
    )
    return {
        "cases": len(labels),
        "harmful_cases": harmful,
        "safe_cases": safe,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced,
        "false_safe_rate_all": fn / len(labels),
        "false_alarm_rate_all": fp / len(labels),
        "false_safe_rate_among_harmful": fn / harmful if harmful else float("nan"),
        "false_alarm_rate_among_safe": fp / safe if safe else float("nan"),
        "accuracy": (tp + tn) / len(labels),
    }


def classification_metrics(scores, labels, threshold):
    return binary_metrics(classify_with_threshold(scores, threshold), labels)


def candidate_thresholds(scores):
    unique = sorted(set(float(value) for value in scores))
    if not unique:
        raise ValueError("empty scores")
    candidates = [float("-inf")]
    for left, right in zip(unique[:-1], unique[1:]):
        candidates.append((left + right) / 2.0)
    candidates.append(float("inf"))
    return candidates


def select_balanced_threshold(scores, labels):
    best = None
    for threshold in candidate_thresholds(scores):
        metrics = classification_metrics(scores, labels, threshold)
        finite_abs = abs(threshold) if math.isfinite(threshold) else float("inf")
        key = (
            metrics["balanced_accuracy"],
            metrics["sensitivity"],
            metrics["specificity"],
            -finite_abs,
        )
        if best is None or key > best["key"]:
            best = {"threshold": threshold, "metrics": metrics, "key": key}
    return best


def unique_cases(rows):
    by_key = {}
    for row in rows:
        key = tuple(row[field] for field in CASE_KEY)
        if key not in by_key:
            by_key[key] = row
            continue
        for field in ("actual_delta", "pred_all", "material_change"):
            if by_key[key][field] != row[field]:
                raise ValueError(f"inconsistent repeated case {key}: {field}")
    return list(by_key.values())


def loso_predict(rows, score_field):
    predictions = []
    folds = []
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) != 3:
        raise ValueError(f"expected three seeds, found {seeds}")
    for heldout_seed in seeds:
        train = [row for row in rows if int(row["seed"]) != heldout_seed]
        test = [row for row in rows if int(row["seed"]) == heldout_seed]
        train_seeds = sorted({int(row["seed"]) for row in train})
        train_scores = [float(row[score_field]) for row in train]
        train_labels = [float(row["actual_delta"]) > 0.0 for row in train]
        selected = select_balanced_threshold(train_scores, train_labels)
        threshold = selected["threshold"]
        for row in test:
            score = float(row[score_field])
            label = float(row["actual_delta"]) > 0.0
            predictions.append(
                {
                    **row,
                    "heldout_seed": heldout_seed,
                    "threshold": threshold,
                    "predicted_harmful": int(score > threshold),
                    "endpoint_harmful": int(label),
                }
            )
        test_metrics = classification_metrics(
            [float(row[score_field]) for row in test],
            [float(row["actual_delta"]) > 0.0 for row in test],
            threshold,
        )
        folds.append(
            {
                "heldout_seed": heldout_seed,
                "train_seeds": "+".join(str(seed) for seed in train_seeds),
                "threshold": threshold,
                **{f"train_{key}": value for key, value in selected["metrics"].items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )
    return predictions, folds


def pooled_loso_metrics(prediction_rows):
    return binary_metrics(
        [bool(int(row["predicted_harmful"])) for row in prediction_rows],
        [bool(int(row["endpoint_harmful"])) for row in prediction_rows],
    )


def ranking_metrics(rows, score_field):
    scores = [float(row[score_field]) for row in rows]
    labels = [float(row["actual_delta"]) > 0.0 for row in rows]
    prevalence = sum(labels) / len(labels)
    return {
        "cases": len(rows),
        "harmful_prevalence": prevalence,
        "roc_auc": roc_auc(scores, labels),
        "average_precision": average_precision(scores, labels),
        "ap_gain_over_prevalence": average_precision(scores, labels) - prevalence,
    }


def quantile(values, q):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
