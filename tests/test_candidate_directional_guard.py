import copy
import csv
import random

import numpy as np
import pytest
import torch

from libero.lifelong.analysis.candidate_adamw_step import candidate_adamw_deltas
from libero.lifelong.analysis.candidate_directional_guard import (
    capture_branch_state,
    guarded_optimizer_step,
    load_h100_cosine_thresholds,
    observe_with_directional_guard,
    paired_drift_metrics,
    restore_branch_state,
)


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


def make_parameter_optimizer():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3, weight_decay=0.1)
    return parameter, optimizer


def test_threshold_loading_selects_only_preregistered_rows(tmp_path):
    path = tmp_path / "thresholds.csv"
    fields = ["scope", "horizon", "task", "feature", "heldout_seed", "threshold", "valid_fold"]
    rows = [
        ["full", "H100", "endpoint_harmful", "actual_step_cosine", 42, 0.002, 1],
        ["full", "H100", "endpoint_harmful", "actual_step_cosine", 43, 0.004, 1],
        ["full", "H300", "endpoint_harmful", "actual_step_cosine", 42, 99, 1],
        ["subgroup", "H100", "endpoint_harmful", "actual_step_cosine", 43, 99, 1],
        ["full", "H100", "endpoint_harmful", "actual_step_score", 42, 99, 1],
        ["full", "H100", "endpoint_harmful", "actual_step_cosine", 44, 99, 1],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        writer.writerows(rows)
    assert load_h100_cosine_thresholds(path) == {42: 0.002, 43: 0.004}


def test_no_fire_matches_plain_adamw():
    plain, plain_optimizer = make_parameter_optimizer()
    guarded = torch.nn.Parameter(plain.detach().clone())
    guarded_optimizer = torch.optim.AdamW([guarded], lr=1e-3, weight_decay=0.1)
    gradient = torch.tensor([0.3, -0.4], dtype=torch.float64)
    plain.grad = gradient.clone()
    guarded.grad = gradient.clone()
    plain_optimizer.step()
    _, metrics = guarded_optimizer_step(
        guarded_optimizer,
        [("p", guarded)],
        {"p": gradient.clone()},
        threshold=2.0,
        scale=0.5,
        enabled=True,
    )
    assert metrics["guard_fire"] == 0
    assert torch.equal(plain, guarded)
    assert_nested_equal(plain_optimizer.state[plain], guarded_optimizer.state[guarded])


def test_fire_scales_parameter_displacement():
    parameter, optimizer = make_parameter_optimizer()
    parameter.grad = torch.tensor([0.3, -0.4], dtype=torch.float64)
    before = parameter.detach().clone()
    candidate = candidate_adamw_deltas(optimizer, [("p", parameter)])["p"]["delta"]
    _, metrics = guarded_optimizer_step(
        optimizer,
        [("p", parameter)],
        {"p": -parameter.grad.clone()},
        threshold=-2.0,
        scale=0.5,
        enabled=True,
    )
    assert metrics["guard_fire"] == 1
    assert torch.allclose(parameter, before + 0.5 * candidate, atol=1e-14, rtol=0)


def test_fired_guard_commits_full_optimizer_state():
    reference, reference_optimizer = make_parameter_optimizer()
    guarded = torch.nn.Parameter(reference.detach().clone())
    guarded_optimizer = torch.optim.AdamW([guarded], lr=1e-3, weight_decay=0.1)
    gradient = torch.tensor([0.2, 0.7], dtype=torch.float64)
    reference.grad = gradient.clone()
    guarded.grad = gradient.clone()
    reference_optimizer.step()
    guarded_optimizer_step(
        guarded_optimizer,
        [("p", guarded)],
        {"p": -gradient.clone()},
        threshold=-2.0,
        scale=0.5,
        enabled=True,
    )
    assert_nested_equal(
        reference_optimizer.state[reference], guarded_optimizer.state[guarded]
    )
    assert reference_optimizer.param_groups[0]["lr"] == guarded_optimizer.param_groups[0]["lr"]
    assert not torch.equal(reference, guarded)


class TinyAlgo:
    def __init__(self, state):
        self.policy = torch.nn.Linear(2, 1, bias=False).double()
        self.policy.load_state_dict(copy.deepcopy(state))
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=1e-3, weight_decay=0.1
        )

    def observe(self, data, current_task_id, epoch, ifa_info=None):
        del current_task_id, epoch, ifa_info
        inputs, targets = data
        self.optimizer.zero_grad()
        loss = torch.mean((self.policy(inputs) - targets) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 100.0)
        self.optimizer.step()
        return float(loss.detach())


def test_scale_one_instrumented_control_matches_plain_observe():
    base = torch.nn.Linear(2, 1, bias=False).double().state_dict()
    plain = TinyAlgo(base)
    guarded = TinyAlgo(base)
    data = (
        torch.tensor([[1.0, -2.0], [0.5, 0.2]], dtype=torch.float64),
        torch.tensor([[0.4], [-0.7]], dtype=torch.float64),
    )
    plain_loss = plain.observe(copy.deepcopy(data), 8, 10)
    guarded_loss, _ = observe_with_directional_guard(
        guarded,
        copy.deepcopy(data),
        current_task_id=8,
        epoch=10,
        aggregate_old_gradient=(torch.ones_like(guarded.policy.weight),),
        old_gradient_names=["weight"],
        threshold=-2.0,
        scale=1.0,
        enabled=True,
    )
    assert plain_loss == guarded_loss
    assert torch.equal(plain.policy.weight, guarded.policy.weight)
    assert_nested_equal(
        plain.optimizer.state[plain.policy.weight],
        guarded.optimizer.state[guarded.policy.weight],
    )


def test_capture_restore_recovers_model_optimizer_scheduler_and_rng():
    model = torch.nn.Linear(2, 1).double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    algo = type("Algo", (), {"policy": model, "optimizer": optimizer})()
    loss = model(torch.ones(1, 2, dtype=torch.float64)).sum()
    loss.backward()
    optimizer.step()
    scheduler.step()
    state = capture_branch_state(algo, scheduler)
    expected = (random.random(), np.random.random(), torch.rand(3))
    optimizer.zero_grad()
    model(torch.full((1, 2), 2.0, dtype=torch.float64)).sum().backward()
    optimizer.step()
    scheduler.step()
    random.random()
    np.random.random()
    torch.rand(3)
    restore_branch_state(algo, scheduler, state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    assert torch.equal(actual[2], expected[2])
    assert_nested_equal(model.state_dict(), state["policy"])
    assert_nested_equal(optimizer.state_dict(), state["optimizer"])
    assert_nested_equal(scheduler.state_dict(), state["scheduler"])


def test_paired_drift_metric_signs():
    result = paired_drift_metrics(
        2.0,
        4.0,
        {"old": 2.4, "current": 3.0},
        {"old": 2.1, "current": 3.2},
    )
    assert result["control_old_drift"] == pytest.approx(0.4)
    assert result["guard_old_drift"] == pytest.approx(0.1)
    assert result["old_benefit"] == pytest.approx(0.3)
    assert result["relative_old_benefit"] == pytest.approx(0.15)
    assert result["current_harm"] == pytest.approx(0.2)
    assert result["relative_current_harm"] == pytest.approx(0.05)
    assert result["plasticity_preservation"] == pytest.approx(0.8)
