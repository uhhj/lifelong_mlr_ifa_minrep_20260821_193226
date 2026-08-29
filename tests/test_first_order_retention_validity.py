from pathlib import Path

import pytest
import torch

from libero.lifelong.algos.mlr_per_task_temporal_robot_igfa import (
    per_task_temporal_head_corrections,
)
from libero.lifelong.analysis.first_order_retention_validity import (
    backend_parameter_groups,
    classify_checkpoint_transition,
    load_checkpoint_into_policy,
    parameter_displacement_dot,
    simulate_layer_a_batch,
    weighted_chunk_mean,
)
from libero.lifelong.analysis.per_task_temporal_interference import (
    build_stage_records,
    deterministic_diagnostic_batches,
    gradient_parity,
    load_record_batch,
    load_temporal_bin_map,
)
from libero.lifelong.models.brief import BriefPolicy
from scripts.diagnose_per_task_temporal_interference import load_config


def test_parameter_displacement_dot():
    gradient = (torch.tensor([1.0, -2.0]), torch.tensor([3.0]))
    displacement = (torch.tensor([0.5, 4.0]), torch.tensor([-1.0]))
    assert parameter_displacement_dot(gradient, displacement) == pytest.approx(-10.5)


def test_linear_loss_first_order_prediction_is_exact():
    coefficient = torch.tensor([2.0, -3.0])
    start = torch.tensor([0.4, -0.2])
    end = torch.tensor([-0.1, 0.7])
    prediction = parameter_displacement_dot((coefficient,), (end - start,))
    actual = float(coefficient @ end - coefficient @ start)
    assert prediction == pytest.approx(actual)


def test_transformer_flip_classification():
    result = classify_checkpoint_transition(-1.0, 1.5, -0.2)
    assert result["pred_all"] == pytest.approx(0.5)
    assert result["checkpoint_transformer_flip"]


def test_full_first_order_false_safe_classification():
    result = classify_checkpoint_transition(-0.4, 0.1, 0.2)
    assert result["pred_all"] <= 0.0
    assert result["full_first_order_false_safe"]


def test_probe_weighted_mean_handles_partial_last_minibatch():
    values = [2.0, 8.0, 20.0]
    sizes = [4, 4, 2]
    expanded = [2.0] * 4 + [8.0] * 4 + [20.0] * 2
    assert weighted_chunk_mean(values, sizes) == pytest.approx(sum(expanded) / len(expanded))


def test_real_v2_final_head_gradient_parity():
    root = Path.cwd()
    run = root / "runs" / "object_seed42_mlr_per_task_temporal_igfa" / "run_001"
    checkpoint = run / "mlrifa_7_model_ep95.pth"
    sidecar_path = root / "metadata" / "temporal_bins" / "object_seed42.json"
    if not checkpoint.is_file() or not sidecar_path.is_file():
        pytest.skip("formal server artifacts are unavailable")
    cfg = load_config(run / "config.json", root, 42, "cuda")
    policy = BriefPolicy(cfg, None).cuda().eval()
    load_checkpoint_into_policy(policy, checkpoint)
    sidecar = load_temporal_bin_map(sidecar_path, expected_seed=42)
    records, _ = build_stage_records(
        root / "features" / "object_seed42", 42, 8, sidecar, rate=0.5
    )
    batches = deterministic_diagnostic_batches(records, 10, 50, 20260830)
    batch = next(
        candidate for candidate in batches
        if len({item["task_id"] for item in candidate if item["task_id"] < 8}) >= 2
    )
    _, simulated = simulate_layer_a_batch(policy, batch, 8, "V2")

    policy.zero_grad(set_to_none=True)
    features, actions, task_ids, progress_bins = load_record_batch(batch, "cuda")
    temporal = policy.temporal_encode(features)
    distribution = policy.policy_head(temporal[:, -1, :], train=True)
    losses = policy.policy_head.loss_fn(distribution, actions, reduction="none")
    head_named, _ = backend_parameter_groups(policy)
    head = [parameter for _, parameter in head_named]
    correction, _ = per_task_temporal_head_corrections(
        losses, task_ids, progress_bins, 8, head,
        gate_strength=1.0, n_bins=5,
    )
    losses.mean().backward()
    for parameter, delta in zip(head, correction):
        parameter.grad.add_(delta)
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 100.0)
    observed = tuple(parameter.grad.detach().clone() for parameter in head)
    max_abs, relative_l2 = gradient_parity(simulated["head"], observed)
    assert max_abs <= 5e-6
    assert relative_l2 <= 1e-6
