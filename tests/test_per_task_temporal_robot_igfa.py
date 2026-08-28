import itertools

import torch

from libero.lifelong.algos.mlr_per_task_temporal_robot_igfa import (
    gradient_dot,
    per_task_temporal_head_corrections,
    project_gradient_to_task_halfspaces,
)
from libero.lifelong.algos.mlr_static_robot_igfa import signed_replay_projection


def descriptor(task_id, values):
    return {
        "task_id": task_id,
        "gradient": (torch.tensor(values, dtype=torch.float32),),
        "n_samples": 1,
    }


def project(new, old, strength=1.0):
    return project_gradient_to_task_halfspaces(
        (torch.tensor(new, dtype=torch.float32),), old,
        gate_strength=strength,
    )


def test_single_old_equivalence_to_temporal_v1():
    for new, old in (
        ([1.0, 0.0], [1.0, 0.0]),
        ([1.0, 0.0], [-1.0, 1.0]),
        ([1.0, 0.0], [0.0, 1.0]),
    ):
        g_new = (torch.tensor(new),)
        g_old = (torch.tensor(old),)
        expected, _ = signed_replay_projection(g_new, g_old, gate_strength=1.0)
        actual, _ = project(new, [descriptor(6, old)])
        assert torch.allclose(actual[0], expected[0], atol=1e-6, rtol=1e-6)


def test_order_invariance():
    old = [
        descriptor(6, [-1.0, 1.0, 0.0]),
        descriptor(7, [0.0, -1.0, 1.0]),
        descriptor(8, [-1.0, 0.0, -1.0]),
    ]
    outputs = [project([1.0, 0.5, 0.25], list(order))[0][0] for order in itertools.permutations(old)]
    for output in outputs[1:]:
        assert torch.allclose(output, outputs[0], atol=1e-6, rtol=1e-6)


def test_masked_conflict_is_repaired_per_task():
    g_new = (torch.tensor([1.0, 0.0]),)
    old6 = (torch.tensor([-1.0, 1.0]),)
    old7 = (torch.tensor([1.0, 1.0]),)
    aggregate = (0.5 * (old6[0] + old7[0]),)
    v1, v1_metrics = signed_replay_projection(g_new, aggregate)
    assert not v1_metrics["gate_fired"]
    assert gradient_dot(v1, old6).item() < 0.0

    v2, metrics = project_gradient_to_task_halfspaces(
        g_new,
        [
            {"task_id": 6, "gradient": old6, "n_samples": 1},
            {"task_id": 7, "gradient": old7, "n_samples": 1},
        ],
    )
    assert metrics["projection_fired"]
    assert gradient_dot(v2, old6).item() >= -1e-7
    assert gradient_dot(v2, old7).item() >= -1e-7


def test_multi_constraint_feasibility():
    output, metrics = project(
        [1.0, 0.0],
        [descriptor(6, [-1.0, 1.0]), descriptor(7, [-1.0, -1.0])],
    )
    assert metrics["post_violation_count"] == 0
    assert torch.dot(output[0], torch.tensor([-1.0, 1.0])) >= -1e-7
    assert torch.dot(output[0], torch.tensor([-1.0, -1.0])) >= -1e-7


def test_minimum_change_known_2d_case():
    output, _ = project(
        [-1.0, -1.0],
        [descriptor(6, [1.0, 0.0]), descriptor(7, [0.0, 1.0])],
    )
    assert torch.allclose(output[0], torch.zeros(2), atol=1e-6, rtol=0.0)


def test_gate_zero_matches_baseline_head_and_transformer():
    head = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
    transformer = torch.nn.Parameter(torch.tensor([0.5, -0.1]))
    head_coefficients = torch.tensor([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]])
    transformer_coefficients = torch.tensor([[0.2, 0.7], [-0.4, 0.1], [0.3, -0.2]])
    losses = head_coefficients @ head + transformer_coefficients @ transformer
    delta, _ = per_task_temporal_head_corrections(
        losses, torch.tensor([7, 6, 8]), torch.tensor([1, 1, 1]),
        current_task_id=7, head_params=[head], gate_strength=0.0,
    )
    losses.mean().backward()
    assert torch.equal(head.grad + delta[0], head_coefficients.mean(dim=0))
    assert torch.allclose(
        transformer.grad, transformer_coefficients.mean(dim=0),
        atol=5e-6, rtol=1e-6,
    )


def test_stage7_single_old_correction_matches_temporal_v1():
    head = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
    coefficients = torch.tensor([[1.0, -1.0], [-1.0, 0.0]])
    losses = coefficients @ head
    v2_delta, _ = per_task_temporal_head_corrections(
        losses, torch.tensor([7, 6]), torch.tensor([2, 2]),
        current_task_id=7, head_params=[head], gate_strength=1.0,
    )
    g_new = (coefficients[0],)
    g_old = (coefficients[1],)
    projected, _ = signed_replay_projection(g_new, g_old, gate_strength=1.0)
    expected = 0.5 * (projected[0] - g_new[0])
    assert torch.allclose(v2_delta[0], expected, atol=1e-6, rtol=1e-6)


def test_unresolved_and_no_anchor_keep_baseline_behavior():
    head = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
    coefficients = torch.tensor([[2.0, -1.0], [-3.0, 4.0], [1.0, 2.0]])
    losses = coefficients @ head
    delta, results = per_task_temporal_head_corrections(
        losses, torch.tensor([7, 6, 6]), torch.tensor([-1, -1, 3]),
        current_task_id=7, head_params=[head], gate_strength=1.0,
    )
    losses.mean().backward()
    assert results == []
    assert torch.count_nonzero(delta[0]).item() == 0
    assert torch.allclose(
        head.grad, coefficients.mean(dim=0), atol=5e-6, rtol=1e-6,
    )
