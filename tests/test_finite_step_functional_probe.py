import csv
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from libero.lifelong.analysis.finite_step_functional_probe import (
    classify_finite_step,
    confusion_metrics,
    load_probe_manifest,
    set_interpolated_backend,
)
from libero.lifelong.analysis.first_order_retention_validity import (
    capture_backend_parameters,
    evaluate_mean_loss,
    load_checkpoint_into_policy,
)
from libero.lifelong.models.brief import BriefPolicy
from scripts.diagnose_first_order_retention_validity import checkpoint_path, run_dir
from scripts.diagnose_per_task_temporal_interference import load_config


class ToyBackend(nn.Module):
    def __init__(self):
        super().__init__()
        self.policy_head = nn.Linear(2, 1, bias=False)
        self.temporal_transformer = nn.Linear(2, 1, bias=False)


def snapshots(policy, head, transformer):
    head_names = [name for name, _ in policy.named_parameters() if name.startswith("policy_head.")]
    transformer_names = [
        name for name, _ in policy.named_parameters()
        if name.startswith("temporal_transformer.")
    ]
    return {
        "head_names": head_names,
        "transformer_names": transformer_names,
        "head": (torch.tensor(head, dtype=torch.float32),),
        "transformer": (torch.tensor(transformer, dtype=torch.float32),),
    }


def test_backend_interpolation_endpoints():
    policy = ToyBackend()
    start = snapshots(policy, [[1.0, 2.0]], [[3.0, 4.0]])
    end = snapshots(policy, [[5.0, 6.0]], [[7.0, 8.0]])
    set_interpolated_backend(policy, start, end, 0.0)
    assert torch.equal(policy.policy_head.weight, start["head"][0])
    assert torch.equal(policy.temporal_transformer.weight, start["transformer"][0])
    set_interpolated_backend(policy, start, end, 1.0)
    assert torch.equal(policy.policy_head.weight, end["head"][0])
    assert torch.equal(policy.temporal_transformer.weight, end["transformer"][0])


def test_backend_interpolation_midpoint():
    policy = ToyBackend()
    start = snapshots(policy, [[0.0, 2.0]], [[4.0, 6.0]])
    end = snapshots(policy, [[2.0, 4.0]], [[8.0, 10.0]])
    set_interpolated_backend(policy, start, end, 0.5)
    assert torch.equal(policy.policy_head.weight, torch.tensor([[1.0, 3.0]]))
    assert torch.equal(policy.temporal_transformer.weight, torch.tensor([[6.0, 8.0]]))


def test_linear_loss_exact_chord():
    start = torch.tensor([0.4, -0.2])
    end = torch.tensor([-0.1, 0.7])
    coefficient = torch.tensor([2.0, -3.0])
    rho = 0.10
    endpoint_delta = float(coefficient @ end - coefficient @ start)
    first_order = float(coefficient @ (end - start))
    probe = start + rho * (end - start)
    probe_delta = float(coefficient @ probe - coefficient @ start)
    assert probe_delta == pytest.approx(rho * endpoint_delta)
    assert probe_delta == pytest.approx(rho * first_order)


def test_nonlinear_false_safe_rescue_classification():
    result = classify_finite_step(0.2, 0.8, first_order_false_safe=True)
    assert not result["probe_false_safe"]
    assert result["rescued_first_order_false_safe"]
    assert result["probe_sign_agreement"]


def test_confusion_metrics():
    result = confusion_metrics([1.0, -1.0, 1.0, -1.0], [1.0, 1.0, -1.0, -1.0])
    assert result["sensitivity"] == pytest.approx(0.5)
    assert result["specificity"] == pytest.approx(0.5)
    assert result["balanced_accuracy"] == pytest.approx(0.5)
    assert result["false_safe_rate"] == pytest.approx(0.25)
    assert result["false_alarm_rate"] == pytest.approx(0.25)
    assert result["sign_agreement_rate"] == pytest.approx(0.5)


def test_real_endpoint_interpolation_parity():
    root = Path.cwd()
    input_dir = root / "reports" / "first_order_retention_validity"
    run = run_dir(root, "V2", 42)
    if not (run / "mlrifa_8_model_ep90.pth").is_file():
        pytest.skip("formal server artifacts are unavailable")
    rows = list(csv.DictReader((input_dir / "checkpoint_transition_raw.csv").open()))
    canonical = next(
        row for row in rows
        if row["method"] == "V2" and row["seed"] == "42"
        and row["stage"] == "8" and row["old_task"] == "7"
        and row["start_epoch"] == "90" and row["end_epoch"] == "95"
    )
    probes = load_probe_manifest(input_dir / "probe_manifest.csv", root)
    cfg = load_config(run / "config.json", root, 42, "cuda")
    policy = BriefPolicy(cfg, None).cuda().eval()
    load_checkpoint_into_policy(policy, checkpoint_path(root, "V2", 42, 8, 90))
    start = capture_backend_parameters(policy)
    load_checkpoint_into_policy(policy, checkpoint_path(root, "V2", 42, 8, 95))
    end = capture_backend_parameters(policy)
    paths = probes[(42, 7)]
    for rho, field in ((0.0, "loss_start"), (1.0, "loss_end")):
        set_interpolated_backend(policy, start, end, rho)
        observed = evaluate_mean_loss(policy, paths, batch_size=32)
        expected = float(canonical[field])
        assert observed == pytest.approx(expected, rel=1e-6, abs=1e-6)
