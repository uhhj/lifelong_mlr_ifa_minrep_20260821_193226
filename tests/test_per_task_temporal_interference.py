import math
import random

import torch

from libero.lifelong.algos.mlr_static_robot_igfa import signed_replay_projection
from libero.lifelong.analysis.per_task_temporal_interference import (
    classify_task_relations,
    gradient_parity,
    gradient_relation,
    reconstruct_stage_replay_paths,
    weighted_task_gradients,
)


def test_replay_reconstruction_is_deterministic_and_restarts_rng_per_task(tmp_path):
    for task in (6, 7):
        task_dir = tmp_path / str(task)
        task_dir.mkdir()
        for index in range(9):
            (task_dir / f"{index}.pt").touch()
    first = reconstruct_stage_replay_paths(tmp_path, 42, [6, 7], rate=0.5)
    second = reconstruct_stage_replay_paths(tmp_path, 42, [6, 7], rate=0.5)
    expected_indices = random.Random(42).sample(list(range(9)), math.ceil(9 * 0.5))
    assert first == second
    assert [int(path.stem) for path in first[6]] == expected_indices
    assert [int(path.stem) for path in first[7]] == expected_indices


def test_weighted_old_task_gradients_match_direct_mean_gradient():
    parameter = torch.nn.Parameter(torch.tensor([0.3, -0.2]))
    coefficients = {
        6: torch.tensor([[1.0, 2.0], [3.0, -1.0]]),
        7: torch.tensor([[-2.0, 4.0]]),
        8: torch.tensor([[5.0, 0.0], [1.0, 1.0], [-3.0, 2.0]]),
    }
    task_gradients = {}
    task_counts = {}
    losses = []
    for task_id, values in coefficients.items():
        task_loss = values @ parameter
        losses.append(task_loss)
        task_counts[task_id] = len(values)
        task_gradients[task_id] = torch.autograd.grad(
            task_loss.mean(), parameter, retain_graph=True
        )
    direct = torch.autograd.grad(torch.cat(losses).mean(), parameter)
    weighted = weighted_task_gradients(task_gradients, task_counts)
    max_abs, relative_l2 = gradient_parity(weighted, direct)
    assert max_abs <= 5e-6
    assert relative_l2 <= 1e-6


def test_mixed_old_sign_detection():
    result = classify_task_relations([-2.0, 0.5, 1.0], 0.1, [-1.0, 0.2, 0.3])
    assert result["multi_old"]
    assert result["mixed_old_sign"]


def test_masked_conflict_detection_requires_safe_aggregate():
    masked = classify_task_relations([-1.0, 2.0], 0.25, [-1.0, 2.0])
    gated = classify_task_relations([-1.0, 2.0], -0.25, [-0.2, 1.0])
    assert masked["masked_conflict"]
    assert not gated["masked_conflict"]


def test_aggregate_projection_can_leave_residual_individual_conflict():
    new = (torch.tensor([1.0, 0.0]),)
    old6 = (torch.tensor([-2.0, 1.0]),)
    old7 = (torch.tensor([0.0, -3.0]),)
    aggregate = weighted_task_gradients({6: old6, 7: old7}, {6: 1, 7: 1})
    projected, metrics = signed_replay_projection(new, aggregate, gate_strength=1.0)
    pre = [gradient_relation(new, old6)["dot"], gradient_relation(new, old7)["dot"]]
    post = [
        gradient_relation(projected, old6)["dot"],
        gradient_relation(projected, old7)["dot"],
    ]
    result = classify_task_relations(pre, metrics["dot"].item(), post)
    assert metrics["gate_fired"]
    assert result["residual_conflict"]


def test_single_old_task_is_not_a_multi_old_candidate():
    result = classify_task_relations([-1.0], -1.0, [0.0])
    assert not result["multi_old"]
    assert not result["mixed_old_sign"]
    assert not result["masked_conflict"]
    assert not result["residual_conflict"]
