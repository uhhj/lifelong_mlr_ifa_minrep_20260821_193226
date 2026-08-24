import pytest
import torch
import torch.nn as nn

from libero.lifelong.algos.mlr_static_robot_igfa import (
    replace_head_gradients,
    signed_replay_projection,
    weighted_head_gradient,
)


def test_positive_transfer_is_not_gated():
    new = (torch.tensor([1.0, 0.0]),)
    replay = (torch.tensor([1.0, 0.0]),)
    projected, metrics = signed_replay_projection(new, replay)
    assert metrics["dot"].item() > 0
    assert not metrics["gate_fired"]
    assert torch.equal(projected[0], new[0])


def test_negative_conflict_projects_to_anchor_boundary():
    new = (torch.tensor([1.0, -1.0]),)
    replay = (torch.tensor([-1.0, 0.0]),)
    projected, metrics = signed_replay_projection(new, replay)
    assert metrics["dot"].item() < 0
    assert metrics["gate_fired"]
    assert torch.dot(projected[0], replay[0]).item() == pytest.approx(0.0, abs=1e-7)


def test_orthogonal_gradient_is_unchanged():
    new = (torch.tensor([1.0, 0.0]),)
    replay = (torch.tensor([0.0, 1.0]),)
    projected, metrics = signed_replay_projection(new, replay)
    assert metrics["dot"].item() == 0.0
    assert not metrics["gate_fired"]
    assert torch.equal(projected[0], new[0])


def test_weighted_recombination_matches_unequal_mixed_mean():
    new = (torch.tensor([2.0, 4.0]),)
    replay = (torch.tensor([-1.0, 1.0]),)
    combined = weighted_head_gradient(new, replay, n_new=3, n_replay=1)[0]
    expected = 0.75 * new[0] + 0.25 * replay[0]
    assert torch.equal(combined, expected)


def test_gate_replaces_only_head_gradient():
    transformer = nn.Parameter(torch.tensor([2.0]))
    head = nn.Parameter(torch.tensor([3.0, 4.0]))
    baseline_loss = transformer.square().sum() + head.square().sum()
    baseline_loss.backward()
    transformer_baseline = transformer.grad.clone()
    replacement = (torch.tensor([-0.5, 0.25]),)
    replace_head_gradients([head], replacement, loss_scale=1.0)
    assert torch.equal(transformer.grad, transformer_baseline)
    assert torch.equal(head.grad, replacement[0])
