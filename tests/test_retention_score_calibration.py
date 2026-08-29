import pytest

from libero.lifelong.analysis.retention_score_calibration import (
    average_precision,
    classification_metrics,
    loso_predict,
    pooled_loso_metrics,
    roc_auc,
    select_balanced_threshold,
    unique_cases,
)


def test_roc_auc_perfect_reverse_and_ties():
    labels = [False, False, True, True]
    assert roc_auc([0.0, 1.0, 2.0, 3.0], labels) == pytest.approx(1.0)
    assert roc_auc([3.0, 2.0, 1.0, 0.0], labels) == pytest.approx(0.0)
    assert roc_auc([1.0, 1.0, 1.0, 1.0], labels) == pytest.approx(0.5)


def test_average_precision_tie_group_is_deterministic():
    scores = [0.9, 0.8, 0.8, 0.1]
    labels = [True, True, False, False]
    assert average_precision(scores, labels) == pytest.approx(5.0 / 6.0)


def test_threshold_selection_can_choose_negative_threshold():
    scores = [-0.1, -0.2, -1.0, -1.2]
    labels = [True, True, False, False]
    zero = classification_metrics(scores, labels, 0.0)
    selected = select_balanced_threshold(scores, labels)
    assert zero["balanced_accuracy"] == pytest.approx(0.5)
    assert selected["threshold"] < 0.0
    assert selected["metrics"]["balanced_accuracy"] == pytest.approx(1.0)


def toy_rows():
    rows = []
    for seed in (42, 43, 44):
        rows.extend(
            [
                {"seed": str(seed), "score": "-0.1", "actual_delta": "1.0"},
                {"seed": str(seed), "score": "-1.0", "actual_delta": "-1.0"},
            ]
        )
    return rows


def test_loso_never_uses_heldout_seed_for_threshold_selection():
    _, folds = loso_predict(toy_rows(), "score")
    assert len(folds) == 3
    for fold in folds:
        train_seeds = {int(value) for value in fold["train_seeds"].split("+")}
        assert fold["heldout_seed"] not in train_seeds
        assert len(train_seeds) == 2
        assert fold["threshold"] == pytest.approx(-0.55)


def test_first_order_case_dedup_and_inconsistency_detection():
    base = {
        "method": "V1", "seed": "42", "stage": "7", "old_task": "6",
        "start_epoch": "70", "end_epoch": "75", "interval_type": "short",
        "actual_delta": "1.0", "pred_all": "-0.2", "material_change": "1",
    }
    rows = [{**base, "rho": str(rho)} for rho in (0.05, 0.10, 0.25, 0.50)]
    assert len(unique_cases(rows)) == 1
    inconsistent = rows + [{**base, "rho": "0.75", "pred_all": "0.1"}]
    with pytest.raises(ValueError, match="inconsistent repeated case"):
        unique_cases(inconsistent)


def test_end_to_end_negative_score_calibration_rescue():
    rows = []
    for seed in (42, 43, 44):
        rows.extend(
            [
                {"seed": str(seed), "score": "-0.1", "actual_delta": "1.0"},
                {"seed": str(seed), "score": "-0.2", "actual_delta": "0.5"},
                {"seed": str(seed), "score": "-1.0", "actual_delta": "-0.5"},
                {"seed": str(seed), "score": "-1.2", "actual_delta": "-1.0"},
            ]
        )
    labels = [float(row["actual_delta"]) > 0.0 for row in rows]
    scores = [float(row["score"]) for row in rows]
    assert roc_auc(scores, labels) == pytest.approx(1.0)
    assert classification_metrics(scores, labels, 0.0)["balanced_accuracy"] == pytest.approx(0.5)
    predictions, _ = loso_predict(rows, "score")
    assert pooled_loso_metrics(predictions)["balanced_accuracy"] == pytest.approx(1.0)
