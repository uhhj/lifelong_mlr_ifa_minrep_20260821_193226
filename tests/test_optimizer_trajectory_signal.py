import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from libero.lifelong.analysis.optimizer_trajectory_signal import (
    build_anchor_index,
    build_horizon_transition_summary,
    build_trajectory_cases,
    enrich_case_with_history,
    horizon_transition_key,
    read_csv,
    run_loso,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports" / "online_first_order_margin_validity"


def anchor(update, score, relative=None):
    return {
        "seed": "42",
        "stage": "8",
        "global_update": str(update),
        "actual_step_score": str(score),
        "actual_step_relative": str(score if relative is None else relative),
        "actual_step_cosine": "0.25",
        "actual_step_delta_norm": "2.0",
        "actual_step_old_grad_norm": "3.0",
    }


def case(update, harmful=0, seed=42, stage=8, score=None):
    row = {
        "seed": str(seed),
        "stage": str(stage),
        "global_update": str(update),
        "future_global_update": str(update + 100),
        "actual_delta": "1" if harmful else "-1",
        "relative_actual_delta": "0.1" if harmful else "-0.1",
        "endpoint_harmful": str(harmful),
    }
    if score is not None:
        row["single_raw"] = str(score)
    return row


def test_history_join_uses_update_ids():
    index = build_anchor_index([anchor(100, 0), anchor(200, 2), anchor(300, 4), anchor(400, 8)])
    enriched = enrich_case_with_history(case(300), index)
    assert enriched["score_prev1"] == 2.0
    assert enriched["score_prev2"] == 0.0


def test_ema3_fixed_weights():
    index = build_anchor_index([anchor(100, 0), anchor(200, 2), anchor(300, 4)])
    assert enrich_case_with_history(case(300), index)["ema3_raw"] == 2.5


def test_mean_slope_and_acceleration_formulas():
    index = build_anchor_index([anchor(100, 0), anchor(200, 2), anchor(300, 4)])
    row = enrich_case_with_history(case(300), index)
    assert row["mean2_raw"] == 3.0
    assert row["mean3_raw"] == 2.0
    assert row["slope_raw"] == 2.0
    assert row["acceleration_raw"] == 0.0


def test_horizon_transition_four_classes():
    assert horizon_transition_key(False, False) == "safe100_safe300"
    assert horizon_transition_key(True, True) == "harmful100_harmful300"
    assert horizon_transition_key(False, True) == "safe100_harmful300"
    assert horizon_transition_key(True, False) == "harmful100_safe300"
    rows100 = []
    rows300 = []
    for update, pair in enumerate(((0, 0), (1, 1), (0, 1), (1, 0)), start=1):
        shared = {"single_raw": str(update), "ema3_raw": str(update)}
        rows100.append({**case(update, pair[0]), **shared})
        rows300.append({**case(update, pair[1]), **shared})
    summary, _ = build_horizon_transition_summary(rows100, rows300)
    overall = next(row for row in summary if row["scope"] == "ALL")
    assert [overall[name] for name in (
        "safe100_safe300", "harmful100_harmful300",
        "safe100_harmful300", "harmful100_safe300",
    )] == [1, 1, 1, 1]


def test_loso_threshold_has_no_heldout_leakage():
    rows = []
    for seed in (42, 43, 44):
        rows.extend([
            case(seed * 10, 0, seed=seed, score=-1.0),
            case(seed * 10 + 1, 1, seed=seed, score=1.0),
        ])
    _, folds, _ = run_loso(rows, "single_raw")
    threshold_before = next(row["threshold"] for row in folds if row["heldout_seed"] == 42)
    changed = [dict(row) for row in rows]
    for row in changed:
        if int(row["seed"]) == 42:
            row["single_raw"] = str(float(row["single_raw"]) * 1000000.0)
    _, folds_changed, _ = run_loso(changed, "single_raw")
    threshold_after = next(row["threshold"] for row in folds_changed if row["heldout_seed"] == 42)
    assert threshold_before == threshold_after


def test_single_baseline_exact_parity_on_committed_cases():
    anchors = read_csv(SOURCE / "anchor_raw.csv")
    index = build_anchor_index(anchors)
    for filename in ("cases_h100.csv", "cases_h300.csv"):
        cases = read_csv(SOURCE / filename)
        trajectory = build_trajectory_cases(cases, index)
        source_by_key = {
            (row["seed"], row["stage"], row["global_update"]): row for row in cases
        }
        assert trajectory
        for row in trajectory:
            original = source_by_key[(row["seed"], row["stage"], row["global_update"])]
            assert row["single_raw"] == float(original["actual_step_score"])


def test_diagnosis_does_not_mutate_inputs(tmp_path):
    copied = {}
    for filename in ("anchor_raw.csv", "cases_h100.csv", "cases_h300.csv"):
        destination = tmp_path / filename
        shutil.copyfile(SOURCE / filename, destination)
        copied[filename] = destination.read_text(encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "diagnose_optimizer_trajectory_signal.py"),
            "--anchor", str(tmp_path / "anchor_raw.csv"),
            "--h100", str(tmp_path / "cases_h100.csv"),
            "--h300", str(tmp_path / "cases_h300.csv"),
            "--output-dir", str(tmp_path / "output"),
        ],
        cwd=ROOT,
        check=True,
    )
    for filename, text in copied.items():
        assert (tmp_path / filename).read_text(encoding="utf-8") == text
