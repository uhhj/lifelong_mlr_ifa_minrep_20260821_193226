import copy
from pathlib import Path

import pytest
import torch

from libero.lifelong.algos.mlr_temporal_robot_igfa import MLRTemporalRobotIGFA
from libero.lifelong.analysis.candidate_adamw_step import (
    candidate_adamw_deltas,
    candidate_instrumented_observe,
    displacement_metrics,
)
from libero.lifelong.analysis.finite_step_functional_probe import load_probe_manifest
from libero.lifelong.analysis.first_order_retention_validity import (
    load_checkpoint_into_policy,
)
from libero.lifelong.analysis.per_task_temporal_interference import load_temporal_bin_map
from scripts.diagnose_first_order_retention_validity import checkpoint_path, run_dir
from scripts.diagnose_online_first_order_margin import (
    build_loader,
    create_optimizer_scheduler,
)
from scripts.diagnose_per_task_temporal_interference import load_config


def candidate_actual(optimizer, named_parameters):
    named_parameters = list(named_parameters)
    before = {name: parameter.detach().clone() for name, parameter in named_parameters}
    candidate = {
        name: record["delta"]
        for name, record in candidate_adamw_deltas(
            optimizer, named_parameters
        ).items()
    }
    optimizer.step()
    actual = {
        name: parameter.detach() - before[name]
        for name, parameter in named_parameters
        if parameter.grad is not None
    }
    return candidate, actual


def assert_delta_parity(candidate, actual, atol=1e-12):
    assert candidate.keys() == actual.keys()
    for name in candidate:
        assert torch.allclose(candidate[name], actual[name], atol=atol, rtol=1e-10)


def test_fresh_adamw_state_with_weight_decay():
    parameter = torch.nn.Parameter(torch.tensor([1.5, -2.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=0.2)
    parameter.grad = torch.tensor([0.3, -0.4], dtype=torch.float64)
    candidate, actual = candidate_actual(optimizer, [("p", parameter)])
    assert_delta_parity(candidate, actual)


def test_warm_adamw_state_sixth_step():
    parameter = torch.nn.Parameter(torch.tensor([0.4, -0.8], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=3e-4, weight_decay=0.05)
    for index in range(5):
        parameter.grad = torch.tensor([0.1 + index, -0.2 - index], dtype=torch.float64)
        optimizer.step()
    parameter.grad = torch.tensor([-0.7, 0.9], dtype=torch.float64)
    candidate, actual = candidate_actual(optimizer, [("p", parameter)])
    assert_delta_parity(candidate, actual)


def test_multiple_parameter_names_and_shapes():
    first = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([[3.0], [-4.0]], dtype=torch.float64))
    optimizer = torch.optim.AdamW([first, second], lr=1e-3, weight_decay=0.1)
    first.grad = torch.tensor([0.5, -0.25], dtype=torch.float64)
    second.grad = torch.tensor([[0.2], [0.7]], dtype=torch.float64)
    candidate, actual = candidate_actual(
        optimizer, [("first", first), ("second", second)]
    )
    assert set(candidate) == {"first", "second"}
    assert_delta_parity(candidate, actual)


def test_multiple_param_groups_respect_group_hyperparameters():
    first = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([2.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [
            {"params": [first], "lr": 1e-3, "weight_decay": 0.0},
            {"params": [second], "lr": 5e-4, "weight_decay": 0.3},
        ]
    )
    first.grad = torch.tensor([0.4], dtype=torch.float64)
    second.grad = torch.tensor([-0.6], dtype=torch.float64)
    candidate, actual = candidate_actual(
        optimizer, [("first", first), ("second", second)]
    )
    assert_delta_parity(candidate, actual)


def test_amsgrad_warm_state_parity():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW(
        [parameter], lr=1e-3, weight_decay=0.1, amsgrad=True
    )
    for value in (0.3, -0.8, 0.2):
        parameter.grad = torch.tensor([value, -value], dtype=torch.float64)
        optimizer.step()
    parameter.grad = torch.tensor([0.6, 0.1], dtype=torch.float64)
    candidate, actual = candidate_actual(optimizer, [("p", parameter)])
    assert_delta_parity(candidate, actual)


def assert_nested_equal(first, second):
    if torch.is_tensor(first):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            assert_nested_equal(left, right)
    else:
        assert first == second


def test_candidate_prediction_does_not_mutate_parameter_gradient_or_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=0.1)
    parameter.grad = torch.tensor([0.2, -0.4], dtype=torch.float64)
    optimizer.step()
    parameter.grad = torch.tensor([-0.5, 0.7], dtype=torch.float64)
    parameter_before = parameter.detach().clone()
    gradient_before = parameter.grad.detach().clone()
    state_before = copy.deepcopy(optimizer.state_dict())
    candidate_adamw_deltas(optimizer, [("p", parameter)])
    assert torch.equal(parameter, parameter_before)
    assert torch.equal(parameter.grad, gradient_before)
    assert_nested_equal(optimizer.state_dict(), state_before)


@pytest.fixture(scope="module")
def real_instrumentation_result():
    root = Path.cwd()
    run = run_dir(root, "V1", 42)
    checkpoint = checkpoint_path(root, "V1", 42, 7, 95) if run.is_dir() else None
    if checkpoint is None or not checkpoint.is_file():
        pytest.skip("formal server artifacts are unavailable")
    cfg = load_config(run / "config.json", root, 42, "cuda")
    cfg.temporal_bin_map = str(
        root / "metadata" / "temporal_bins" / "object_seed42.json"
    )
    cfg.ifa_enabled = False
    cfg.experiment_dir = "/tmp/candidate_adamw_parity"
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
        # The release scheduler initializes LR at zero; use its configured
        # nonzero AdamW LR so this one-batch parity test has a displacement.
        for group in algo.optimizer.param_groups:
            group["lr"] = float(cfg.train.optimizer.kwargs.lr)
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
    torch.manual_seed(20260901)
    torch.cuda.manual_seed_all(20260901)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    plain.observe(copy.deepcopy(batch), current_task_id=8, epoch=0, ifa_info=None)
    torch.set_rng_state(cpu_rng)
    torch.cuda.set_rng_state_all(cuda_rng)
    result = candidate_instrumented_observe(
        instrumented,
        copy.deepcopy(batch),
        current_task_id=8,
        epoch=0,
        label_probes=label,
        score_probes=score,
        label_batch_size=4,
        score_batch_size=4,
    )
    plain_named = [
        (name, parameter) for name, parameter in plain.policy.named_parameters()
        if parameter.requires_grad
    ]
    instrumented_named = [
        (name, parameter) for name, parameter in instrumented.policy.named_parameters()
        if parameter.requires_grad
    ]
    assert [name for name, _ in plain_named] == [name for name, _ in instrumented_named]
    differences = [
        left.detach() - right.detach()
        for (_, left), (_, right) in zip(plain_named, instrumented_named)
    ]
    max_abs = max(float(value.abs().max()) for value in differences)
    diff_norm = torch.sqrt(sum(torch.sum(value.double() ** 2) for value in differences))
    reference_norm = torch.sqrt(
        sum(torch.sum(parameter.detach().double() ** 2) for _, parameter in plain_named)
    )
    plain_parameters = [
        parameter for group in plain.optimizer.param_groups for parameter in group["params"]
    ]
    instrumented_parameters = [
        parameter for group in instrumented.optimizer.param_groups
        for parameter in group["params"]
    ]
    for left, right in zip(plain_parameters, instrumented_parameters):
        assert_nested_equal(plain.optimizer.state[left], instrumented.optimizer.state[right])
    assert plain.optimizer.param_groups[0]["lr"] == instrumented.optimizer.param_groups[0]["lr"]
    return {
        "max_abs": max_abs,
        "relative_l2": float(diff_norm / (reference_norm + 1e-12)),
        "risk_cosine_abs_error": result["risk_cosine_abs_error"],
        "delta_cosine": result["candidate_actual_delta_cosine"],
    }


def test_real_temporal_v1_step_interceptor_parity(real_instrumentation_result):
    assert real_instrumentation_result["max_abs"] <= 5e-6
    assert real_instrumentation_result["relative_l2"] <= 1e-6


def test_real_mixed_batch_risk_cosine_parity(real_instrumentation_result):
    assert real_instrumentation_result["risk_cosine_abs_error"] <= 1e-3
    assert real_instrumentation_result["delta_cosine"] > 0.999
