from pathlib import Path

import pytest

from libero.lifelong.analysis.optimizer_direction_horizon import (
    build_unique_index,
    enrich_case,
    pair_horizons,
    read_csv,
    reversal_summary,
    row_key,
    run_loso,
    subgroup_rows,
    transition_class,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "online_first_order_margin_validity"


def case(seed, update, harmful=0, score=0.0, stage=8):
    return {
        "seed": str(seed),
        "stage": str(stage),
        "global_update": str(update),
        "actual_delta": "1" if harmful else "-1",
        "actual_step_score": str(score),
        "actual_step_cosine": str(score),
        "actual_step_relative": str(score),
    }


def anchor_for(row):
    return {
        **row,
        "actual_step_delta_norm": "2.0",
        "actual_step_old_grad_norm": "3.0",
    }


def paired_row(seed, update, transition):
    return {
        "seed": seed,
        "stage": 8,
        "global_update": update,
        "transition": transition,
        "actual_step_score": float(update),
        "actual_step_cosine": float(update),
        "actual_step_relative": float(update),
    }


def test_unique_key_and_duplicate_error():
    rows = [case(42, 100), case(42, 200)]
    assert sorted(build_unique_index(rows)) == [(42, 8, 100), (42, 8, 200)]
    with pytest.raises(ValueError, match="duplicate key"):
        build_unique_index(rows + [dict(rows[0])])


def test_pairing_uses_key_intersection_not_row_position():
    h100 = [case(42, update) for update in (100, 200, 300)]
    h300 = [case(42, update) for update in (200, 100)]
    anchors = build_unique_index([anchor_for(row) for row in h100])
    paired = pair_horizons(h100, h300, anchors)
    assert [row["global_update"] for row in paired] == [100, 200]


def test_transition_classes():
    assert transition_class(case(42, 1, 0), case(42, 1, 0)) == "SS"
    assert transition_class(case(42, 1, 1), case(42, 1, 1)) == "HH"
    assert transition_class(case(42, 1, 0), case(42, 1, 1)) == "SH"
    assert transition_class(case(42, 1, 1), case(42, 1, 0)) == "HS"


def test_reversal_metrics_from_known_counts():
    rows = []
    update = 0
    for transition, count in (("SS", 4), ("HH", 3), ("SH", 2), ("HS", 1)):
        for _ in range(count):
            update += 1
            rows.append(paired_row(42, update, transition))
    overall = next(row for row in reversal_summary(rows) if row["scope"] == "ALL")
    assert overall["flip_rate"] == pytest.approx(0.3)
    assert overall["delayed_harmful_rate_among_h100_safe"] == pytest.approx(2 / 6)
    assert overall["recovery_rate_among_h100_harmful"] == pytest.approx(1 / 4)


def test_loso_cosine_threshold_has_no_heldout_leakage():
    rows = []
    for seed in (42, 43, 44):
        rows.extend((case(seed, seed * 10, 0, -1.0), case(seed, seed * 10 + 1, 1, 1.0)))
    _, folds, _ = run_loso(rows, "actual_step_cosine")
    before = next(row["threshold"] for row in folds if row["heldout_seed"] == 42)
    changed = [dict(row) for row in rows]
    for row in changed:
        if int(row["seed"]) == 42:
            row["actual_step_cosine"] = str(float(row["actual_step_cosine"]) * 1000000)
    _, changed_folds, _ = run_loso(changed, "actual_step_cosine")
    after = next(row["threshold"] for row in changed_folds if row["heldout_seed"] == 42)
    assert before == after


def test_subgroup_target_definitions():
    rows = [paired_row(42, index, transition) for index, transition in enumerate(("SS", "SH", "HH", "HS"))]
    delayed = subgroup_rows(rows, "delayed_harmful")
    persistent = subgroup_rows(rows, "persistent_harmful")
    stable = subgroup_rows(rows, "stable_endpoint")
    assert {row["transition"]: row["subgroup_positive"] for row in delayed} == {"SS": 0, "SH": 1}
    assert {row["transition"]: row["subgroup_positive"] for row in persistent} == {"HH": 1, "HS": 0}
    assert {row["transition"]: row["subgroup_positive"] for row in stable} == {"SS": 0, "HH": 1}


def test_committed_case_counts_and_anchor_parity():
    anchors = read_csv(SOURCE / "anchor_raw.csv")
    h100 = read_csv(SOURCE / "cases_h100.csv")
    h300 = read_csv(SOURCE / "cases_h300.csv")
    assert (len(anchors), len(h100), len(h300)) == (267, 261, 249)
    anchor_index = build_unique_index(anchors)
    for row in h100 + h300:
        enriched = enrich_case(row, anchor_index)
        anchor = anchor_index[row_key(row)]
        for field in ("actual_step_score", "actual_step_cosine", "actual_step_relative"):
            assert float(enriched[field]) == float(anchor[field])
