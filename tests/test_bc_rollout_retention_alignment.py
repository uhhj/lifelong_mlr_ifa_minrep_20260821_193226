import math

import pytest

from libero.lifelong.analysis.bc_rollout_retention_alignment import (
    alignment_metrics,
    build_nbt_surrogate_rows,
    build_unique_index,
    decide_verdict,
    matched_method_contrasts,
    parse_selected_epochs,
    quadrant,
    spearman,
)
from libero.lifelong.analysis.retention_score_calibration import roc_auc


def keyed(method, loss=None, success=None, seed=42, stage=7, epoch=70, task=6):
    row = {
        "method": method,
        "seed": seed,
        "stage": stage,
        "epoch": epoch,
        "task": task,
    }
    if loss is not None:
        row["mean_bc_loss"] = loss
    if success is not None:
        row["success_pct"] = success
    return row


def test_exact_key_join_rejects_missing_and_duplicate_rows():
    bc = [keyed("temporal_v1", loss=2), keyed("zero_guard", loss=1)]
    rollout = [keyed("temporal_v1", success=50)]
    with pytest.raises(ValueError, match="key mismatch"):
        matched_method_contrasts(bc, rollout)
    with pytest.raises(ValueError, match="duplicate alignment key"):
        build_unique_index([bc[0], dict(bc[0])])


def test_contrast_sign_bc_better_rollout_worse_is_bw():
    rows = matched_method_contrasts(
        [keyed("temporal_v1", loss=2), keyed("zero_guard", loss=1)],
        [keyed("temporal_v1", success=50), keyed("zero_guard", success=40)],
    )
    assert len(rows) == 1
    assert rows[0]["relative_bc_advantage"] > 0
    assert rows[0]["rollout_advantage_pp"] < 0
    assert rows[0]["quadrant"] == "BW"
    assert rows[0]["dangerous_mismatch"] == 1


def test_all_quadrants_and_tie():
    assert quadrant(1, 1) == "BB"
    assert quadrant(1, -1) == "BW"
    assert quadrant(-1, 1) == "WB"
    assert quadrant(-1, -1) == "WW"
    assert quadrant(0, 1) == "TIE"
    assert quadrant(1, 0) == "TIE"


def test_spearman_uses_average_ranks_for_rollout_ties():
    assert spearman([1, 2, 3], [0, 0, 1]) == pytest.approx(math.sqrt(3) / 2)


def test_auc_excludes_rollout_ties():
    values = [(0.9, 5), (0.8, 0), (0.2, -5), (0.1, 5)]
    rows = [
        {
            "relative_bc_advantage": score,
            "rollout_advantage_pp": success,
            "quadrant": quadrant(score, success),
            "sign_concordant": "" if success == 0 else int(score * success > 0),
        }
        for score, success in values
    ]
    result = alignment_metrics(rows)
    assert result["rollout_non_tie_rows"] == 3
    assert result["roc_auc_non_tie"] == pytest.approx(
        roc_auc([0.9, 0.2, 0.1], [True, False, True])
    )


def test_selected_epoch_tie_breaks_to_later_epoch(tmp_path):
    path = tmp_path / "eval.log"
    lines = []
    for stage in range(6, 10):
        for epoch in (70, 75, 80, 85, 90, 95):
            success = 0.8 if epoch in (90, 95) else 0.7
            lines.append(
                f"[task {stage} epoch {epoch} and {stage} succ.] {success}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert parse_selected_epochs(path) == {6: 95, 7: 95, 8: 95, 9: 95}


def test_nbt_surrogate_forgetting_signs():
    bc = []
    rollout = []
    losses = {(6, 6): 1, (7, 6): 2, (8, 6): 3, (9, 6): 4}
    successes = {(6, 6): 80, (7, 6): 70, (8, 6): 60, (9, 6): 50}
    for method in ("temporal_v1", "zero_guard"):
        for stage in range(6, 10):
            for task in (6, 7, 8):
                if task > stage:
                    continue
                loss = losses.get((stage, task), 1.0 + stage - task)
                success = successes.get((stage, task), 80.0 - 5 * (stage - task))
                bc.append(keyed(method, loss=loss, stage=stage, epoch=95, task=task))
                rollout.append(
                    keyed(method, success=success, stage=stage, epoch=95, task=task)
                )
    selected = {
        (method, 42): {6: 95, 7: 95, 8: 95, 9: 95}
        for method in ("temporal_v1", "zero_guard")
    }
    rows = build_nbt_surrogate_rows(bc, rollout, selected)
    task6 = next(
        row
        for row in rows
        if row["method"] == "temporal_v1" and row["task"] == 6
    )
    assert task6["bc_forgetting"] == pytest.approx(2.0)
    assert task6["rollout_forgetting_pp"] == pytest.approx(20.0)


def test_verdict_supported_not_supported_and_inconclusive():
    low_seeds = [
        {"sign_concordance_non_tie": 0.5},
        {"sign_concordance_non_tie": 0.6},
        {"sign_concordance_non_tie": 0.7},
    ]
    supported = {
        "spearman": 0.2,
        "roc_auc_non_tie": 0.55,
        "sign_concordance_non_tie": 0.55,
        "dangerous_BW_rate": 0.4,
    }
    assert decide_verdict(supported, low_seeds) == (
        "BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_SUPPORTED"
    )
    high_seeds = [
        {"sign_concordance_non_tie": 0.7},
        {"sign_concordance_non_tie": 0.8},
        {"sign_concordance_non_tie": 0.6},
    ]
    aligned = {
        "spearman": 0.7,
        "roc_auc_non_tie": 0.75,
        "sign_concordance_non_tie": 0.75,
        "dangerous_BW_rate": 0.1,
    }
    assert decide_verdict(aligned, high_seeds) == (
        "BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_NOT_SUPPORTED"
    )
    assert decide_verdict(supported, low_seeds, evidence_ok=False) == (
        "BC_ROLLOUT_RETENTION_SURROGATE_MISMATCH_INCONCLUSIVE"
    )
