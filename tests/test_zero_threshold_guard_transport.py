import csv
from pathlib import Path

import pytest

from libero.lifelong.analysis.candidate_directional_guard import (
    guard_would_fire,
    resolve_guard_threshold,
)
from scripts.diagnose_candidate_directional_guard import build_parser
from scripts.summarize_zero_threshold_guard_transport import (
    build_seed_summary,
    evaluate_transport,
)


def toy_pairs(relative_benefit=0.002):
    rows = []
    for seed in (42, 43, 44):
        for stage in (8, 9):
            rows.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "old_benefit_h100": 0.01 if relative_benefit > 0 else -0.01,
                    "relative_old_benefit_h100": relative_benefit,
                    "relative_current_harm_h100": 0.001,
                    "old_benefit_h300": 0.01,
                    "relative_old_benefit_h300": 0.001,
                    "relative_current_harm_h300": 0.002,
                    "guard_fire_count_h100": 30,
                    "guard_fire_count_h300": 90,
                }
            )
    return rows


def test_zero_mode_does_not_read_threshold_file():
    missing = Path("/definitely/not/exist.csv")
    assert resolve_guard_threshold("zero", 44, threshold_path=missing) == 0.0


def test_legacy_loso_mode_reads_committed_style_csv(tmp_path):
    path = tmp_path / "threshold.csv"
    fields = [
        "scope", "horizon", "task", "feature", "heldout_seed",
        "threshold", "valid_fold",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "scope": "full",
                "horizon": "H100",
                "task": "endpoint_harmful",
                "feature": "actual_step_cosine",
                "heldout_seed": 42,
                "threshold": 0.002288496925602411,
                "valid_fold": 1,
            }
        )
    assert resolve_guard_threshold("loso", 42, path) == pytest.approx(
        0.002288496925602411
    )


def test_seed44_stage8_and_stage9_are_accepted_by_protocol_parser():
    parser = build_parser()
    for stage in (8, 9):
        args = parser.parse_args(
            ["--seed", "44", "--stage", str(stage), "--threshold-mode", "zero"]
        )
        assert args.seed == 44
        assert args.stage == stage
        assert args.threshold_mode == "zero"


def test_zero_threshold_fire_rule_is_strictly_positive():
    assert guard_would_fire(1e-6, 0.0)
    assert not guard_would_fire(0.0, 0.0)
    assert not guard_would_fire(-1e-6, 0.0)


def test_seed_summary_aggregates_two_stages_per_seed():
    rows = toy_pairs()
    rows[4]["relative_old_benefit_h100"] = 0.001
    rows[5]["relative_old_benefit_h100"] = 0.003
    summary = {row["seed"]: row for row in build_seed_summary(rows)}
    assert summary[44]["positive_pairs_h100"] == 2
    assert summary[44]["mean_relative_old_benefit_h100"] == pytest.approx(0.002)
    assert summary[44]["pooled_fire_rate_h100"] == pytest.approx(0.3)


def test_transport_verdict_supported_not_supported_and_inconclusive():
    supported = evaluate_transport(toy_pairs())
    assert supported["verdict"] == "ZERO_THRESHOLD_DIRECTIONAL_GUARD_TRANSPORT_SUPPORTED"

    failed = toy_pairs()
    for row in failed:
        if row["seed"] == 44:
            row["old_benefit_h100"] = -0.01
            row["relative_old_benefit_h100"] = -0.002
    assert evaluate_transport(failed)["verdict"] == (
        "ZERO_THRESHOLD_DIRECTIONAL_GUARD_TRANSPORT_NOT_SUPPORTED"
    )

    inconclusive = evaluate_transport(toy_pairs(relative_benefit=0.0005))
    assert inconclusive["verdict"] == (
        "ZERO_THRESHOLD_DIRECTIONAL_GUARD_TRANSPORT_INCONCLUSIVE"
    )
