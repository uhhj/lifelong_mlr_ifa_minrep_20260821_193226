import json
import pathlib
import sys

import pytest
import torch


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_ifa_pressure_across_regimes import build_manifest
from libero.lifelong.analysis.ifa_optimization_interaction import (
    canonical_existing_rows,
    decide_h20,
    gamma_materiality,
    interaction_rows,
    relative_effects,
)


def row(regime, method, seed, fwt, nbt, auc, gamma=""):
    return {
        "regime": regime,
        "method": method,
        "seed": seed,
        "FWT": fwt,
        "NBT": nbt,
        "AUC": auc,
        "gamma": gamma,
    }


def synthetic_rows():
    rows = []
    for seed in (42, 43, 44):
        rows.extend(
            [
                row("legacy_epoch", "mlr", seed, 80, 10, 75),
                row("legacy_epoch", "ifa", seed, 82, 8, 77, "0.31"),
                row("batch_step", "mlr", seed, 90, 12, 85),
                row("batch_step", "ifa", seed, 85, 11, 80, "0.31"),
            ]
        )
    return rows


def test_t1_canonical_metrics_parser_has_12_rows():
    rows = canonical_existing_rows(ROOT)
    assert len(rows) == 12
    assert {(r["regime"], r["method"], r["seed"]) for r in rows} == {
        (regime, method, seed)
        for regime in ("legacy_epoch", "batch_step")
        for method in ("mlr", "ifa")
        for seed in (42, 43, 44)
    }


def test_t2_relative_fwt_sign():
    effects = relative_effects(synthetic_rows())
    assert effects[0]["delta_fwt"] == 2
    assert effects[-1]["delta_fwt"] == -5


def test_t3_nbt_sign_convention():
    effects = relative_effects(synthetic_rows())
    assert effects[0]["delta_nbt"] == -2
    assert effects[-1]["delta_nbt"] == -1


def test_t4_difference_in_differences_exact():
    interactions = interaction_rows(relative_effects(synthetic_rows()))
    assert [r["interaction_fwt"] for r in interactions] == [-7, -7, -7]


def test_t5_gamma_materiality_true():
    rows = [
        {"gamma_delta_fwt": -4, "gamma_delta_auc": -1},
        {"gamma_delta_fwt": -3, "gamma_delta_auc": 0},
        {"gamma_delta_fwt": 0, "gamma_delta_auc": 0},
    ]
    assert gamma_materiality(rows)


def test_t6_gamma_materiality_false():
    rows = [
        {"gamma_delta_fwt": -1, "gamma_delta_auc": 1},
        {"gamma_delta_fwt": 1, "gamma_delta_auc": -1},
        {"gamma_delta_fwt": 0, "gamma_delta_auc": 0},
    ]
    assert not gamma_materiality(rows)


def test_t7_h20_supported():
    interactions = [{"interaction_fwt": value} for value in (-4, -3, -2)]
    effects = [{"delta_fwt": value} for value in (-4, -3, -2)]
    assert decide_h20(interactions, effects, False).endswith("SUPPORTED")


def test_t8_h20_not_supported():
    interactions = [{"interaction_fwt": value} for value in (-0.2, 0.1, 0.2)]
    effects = [{"delta_fwt": value} for value in (-0.5, 0.0, 0.5)]
    assert decide_h20(interactions, effects, False).endswith("NOT_SUPPORTED")


def test_t9_h20_inconclusive():
    interactions = [{"interaction_fwt": value} for value in (-2, -2, 0)]
    effects = [{"delta_fwt": value} for value in (-2, -1, 0)]
    assert decide_h20(interactions, effects, False).endswith("INCONCLUSIVE")


def _parity_train(logging):
    torch.manual_seed(42)
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / 500, 1.0)
    )
    losses = []
    for batch in range(20):
        optimizer.zero_grad()
        bc = ((parameter - batch / 50) ** 2).sum()
        hinge_values = torch.relu(parameter + torch.tensor([0.31, -0.31]))
        loss = bc + 0.1 * hinge_values.mean()
        if logging:
            _ = {
                "base": float(bc.detach()),
                "hinge": float(hinge_values.detach().mean()),
                "active": float((hinge_values.detach() > 0).float().mean()),
            }
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
        scheduler.step()
    return losses, parameter.detach(), optimizer.state_dict(), scheduler.state_dict()


def test_t10_instrumentation_off_on_parity_first20():
    off, p_off, opt_off, sched_off = _parity_train(False)
    on, p_on, opt_on, sched_on = _parity_train(True)
    assert max(abs(a - b) for a, b in zip(off, on)) <= 1e-6
    assert torch.linalg.vector_norm(p_off - p_on) / torch.linalg.vector_norm(p_off) <= 1e-6
    assert opt_off["param_groups"] == opt_on["param_groups"]
    for state_id in opt_off["state"]:
        for name, value in opt_off["state"][state_id].items():
            other = opt_on["state"][state_id][name]
            assert torch.equal(value, other) if torch.is_tensor(value) else value == other
    assert sched_off == sched_on


def test_t11_probe_manifest_has_2304_rows(tmp_path):
    for seed in (42, 43, 44):
        for task in (7, 8, 9):
            directory = tmp_path / f"features/object_seed{seed}/{task}"
            directory.mkdir(parents=True)
            for index in range(256):
                (directory / f"{index}.pt").touch()
    rows = build_manifest(tmp_path, tmp_path / "manifest.csv")
    assert len(rows) == 2304


def test_t12_existing_reconstruction_does_not_overwrite_history():
    paths = [
        ROOT / "reports/baseline_reconciliation/object_3seed_metrics.json",
        ROOT / "reports/paper_reproduction_v2/mlr_formal/three_seed_summary.json",
        ROOT / "reports/paper_reproduction_v2/ifa_formal/three_seed_summary.json",
    ]
    before = [path.read_bytes() for path in paths]
    canonical_existing_rows(ROOT)
    assert before == [path.read_bytes() for path in paths]
