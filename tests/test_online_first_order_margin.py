import copy
from pathlib import Path

import pytest
import torch

from libero.lifelong.algos.mlr_temporal_robot_igfa import MLRTemporalRobotIGFA
from libero.lifelong.analysis.finite_step_functional_probe import load_probe_manifest
from libero.lifelong.analysis.first_order_retention_validity import (
    load_checkpoint_into_policy,
)
from libero.lifelong.analysis.online_first_order_margin import (
    actual_step_metrics,
    build_future_cases,
    capture_backend_device,
    equal_task_gradient,
    incremental_margin_counts,
    instrumented_observe,
    margin_gate,
    raw_risk_from_dot,
)
from libero.lifelong.analysis.per_task_temporal_interference import load_temporal_bin_map
from libero.lifelong.analysis.retention_score_calibration import loso_predict
from scripts.diagnose_first_order_retention_validity import checkpoint_path, run_dir
from scripts.diagnose_online_first_order_margin import (
    build_loader,
    create_optimizer_scheduler,
)
from scripts.diagnose_per_task_temporal_interference import load_config


def test_risk_sign_matches_v1_gate():
    assert raw_risk_from_dot(-2.5) == pytest.approx(2.5)
    assert raw_risk_from_dot(1.5) == pytest.approx(-1.5)


def test_max_margin_equivalence():
    risks = [-0.4, -0.1, -0.3]
    for threshold in (-0.5, -0.2, 0.0):
        assert margin_gate(risks, threshold) == any(value > threshold for value in risks)


def test_equal_task_gradient_is_not_sample_weighted():
    gradients = {
        6: (torch.tensor([2.0, 4.0]),),
        7: (torch.tensor([6.0, 8.0]),),
    }
    result = equal_task_gradient(gradients)
    assert torch.equal(result[0], torch.tensor([4.0, 6.0]))


def test_actual_step_dot_and_cosine():
    gradient = (torch.tensor([1.0, 2.0]),)
    before = (torch.tensor([0.5, -1.0]),)
    after = (torch.tensor([1.5, 1.0]),)
    result = actual_step_metrics(gradient, before, after)
    assert result["score"] == pytest.approx(5.0)
    assert result["cosine"] == pytest.approx(1.0)


def anchor(update, loss):
    return {
        "seed": 42, "stage": 8, "epoch": 2, "global_update": update,
        "aggregate_old_loss": loss, "native_candidate_bins": 1,
        "native_raw_max": -0.1, "native_raw_mean": -0.2,
        "native_cos_max": -0.1, "native_cos_mean": -0.2,
        "actual_step_score": -0.1, "actual_step_cosine": -0.2,
        "actual_step_relative": -0.01, "native_v1_would_gate": 0,
    }


def test_future_case_join_uses_update_id():
    rows = [anchor(300, 3.0), anchor(100, 1.0), anchor(400, 4.0), anchor(200, 2.0)]
    h100 = build_future_cases(rows, 100)
    h300 = build_future_cases(rows, 300)
    assert {(int(row["global_update"]), int(row["future_global_update"])) for row in h100} == {
        (100, 200), (200, 300), (300, 400)
    }
    assert len(h300) == 1 and int(h300[0]["global_update"]) == 100


def test_loso_negative_margin_rescues_currently_ungated_harmful_cases():
    rows = []
    for seed in (42, 43, 44):
        for score, delta in ((-0.1, 1.0), (-0.2, 0.5), (-1.0, -0.5), (-1.2, -1.0)):
            rows.append(
                {"seed": str(seed), "native_raw_max": score, "actual_delta": delta}
            )
    predictions, folds = loso_predict(rows, "native_raw_max")
    thresholds = {int(fold["heldout_seed"]): fold["threshold"] for fold in folds}
    result = incremental_margin_counts(predictions, thresholds)
    assert result["currently_ungated_harmful"] == 6
    assert result["newly_flagged_harmful"] == 6
    assert result["incremental_harmful_recall"] == pytest.approx(1.0)
    assert result["incremental_false_alarms"] == 0
    assert all(
        float(row["threshold"]) < float(row["native_raw_max"]) <= 0.0
        for row in predictions if int(row["predicted_harmful"])
    )


def test_real_instrumentation_parity():
    root = Path.cwd()
    run = run_dir(root, "V1", 42)
    checkpoint = checkpoint_path(root, "V1", 42, 7, 95) if run.is_dir() else None
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("formal server artifacts are unavailable")
    cfg = load_config(run / "config.json", root, 42, "cuda")
    cfg.temporal_bin_map = str(root / "metadata" / "temporal_bins" / "object_seed42.json")
    cfg.ifa_enabled = False
    cfg.experiment_dir = "/tmp/online_margin_parity"
    sidecar = load_temporal_bin_map(cfg.temporal_bin_map, expected_seed=42)
    loader, _, _ = build_loader(root, 42, 8, sidecar, 10, 0)
    loader.sampler.set_epoch(0)
    batch = next(
        data for data in loader
        if "8" in {str(value) for value in data[1]}
        and any(str(value) != "8" for value in data[1])
    )

    def make_algo():
        algo = MLRTemporalRobotIGFA(n_tasks=10, cfg=cfg).cuda()
        load_checkpoint_into_policy(algo.policy, checkpoint)
        algo._gate_stage = 8
        create_optimizer_scheduler(algo, cfg, loader)
        algo.policy.train()
        return algo

    plain = make_algo()
    instrumented = make_algo()
    manifest = load_probe_manifest(
        root / "reports" / "first_order_retention_validity" / "probe_manifest.csv",
        root,
    )
    label = {task: manifest[(42, task)][:8] for task in (6, 7)}
    score = {task: manifest[(42, task)][:4] for task in (6, 7)}
    torch.manual_seed(20260831)
    torch.cuda.manual_seed_all(20260831)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    plain.observe(copy.deepcopy(batch), current_task_id=8, epoch=0, ifa_info=None)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    instrumented_observe(
        instrumented,
        copy.deepcopy(batch),
        current_task_id=8,
        epoch=0,
        label_probes=label,
        score_probes=score,
        label_batch_size=4,
        score_batch_size=4,
    )
    plain_snapshot = capture_backend_device(plain.policy)
    instrumented_snapshot = capture_backend_device(instrumented.policy)
    differences = [
        left - right
        for left, right in zip(plain_snapshot["values"], instrumented_snapshot["values"])
    ]
    max_abs = max(float(value.abs().max()) for value in differences)
    diff_norm = torch.sqrt(sum(torch.sum(value.double() ** 2) for value in differences))
    reference_norm = torch.sqrt(
        sum(torch.sum(value.double() ** 2) for value in plain_snapshot["values"])
    )
    assert max_abs <= 5e-6
    assert float(diff_norm / (reference_norm + 1e-12)) <= 1e-6
    assert plain.optimizer.param_groups[0]["lr"] == instrumented.optimizer.param_groups[0]["lr"]
    plain_parameters = [parameter for group in plain.optimizer.param_groups for parameter in group["params"]]
    instrumented_parameters = [
        parameter for group in instrumented.optimizer.param_groups for parameter in group["params"]
    ]
    for left, right in zip(plain_parameters, instrumented_parameters):
        left_state = plain.optimizer.state[left]
        right_state = instrumented.optimizer.state[right]
        assert left_state.keys() == right_state.keys()
        for key in left_state:
            if torch.is_tensor(left_state[key]):
                assert torch.allclose(left_state[key], right_state[key], atol=5e-6, rtol=1e-6)
            else:
                assert left_state[key] == right_state[key]
