from pathlib import Path

import pytest
import torch

from libero.lifelong.algos.mlr import (
    FileTrajectoryDataset,
    canonical_task_record_id,
)
from libero.lifelong.algos.mlr_temporal_robot_igfa import (
    temporal_head_corrections,
)


def make_losses(head, head_coefficients, transformer=None, transformer_coefficients=None):
    losses = head_coefficients @ head
    if transformer is not None:
        losses = losses + transformer_coefficients @ transformer
    return losses


def corrections(losses, task_ids, bins, head, strength=1.0):
    return temporal_head_corrections(
        losses,
        torch.tensor(task_ids),
        torch.tensor(bins),
        current_task_id=7,
        head_params=[head],
        gate_strength=strength,
        n_bins=5,
    )


def test_canonical_record_id_for_current_and_replay_names(tmp_path):
    assert canonical_task_record_id(tmp_path / "6" / "949.pt") == (6, 949)
    assert canonical_task_record_id(tmp_path / "t6__949.pt") == (6, 949)
    assert canonical_task_record_id(tmp_path / "t6__949__suffix.pt") == (6, 949)
    with pytest.raises(ValueError):
        canonical_task_record_id(tmp_path / "t6__bad.pt")


def test_optional_sidecar_injects_current_and_replay_bins(tmp_path):
    current_dir = tmp_path / "7"
    current_dir.mkdir()
    current = current_dir / "12.pt"
    replay = tmp_path / "t6__949__suffix.pt"
    sample = {"actions": torch.zeros(8, 7), "feature": torch.zeros(8, 4, 512)}
    torch.save(sample, current)
    torch.save(sample, replay)

    baseline, _ = FileTrajectoryDataset([current])[0]
    assert "progress_bin" not in baseline
    dataset = FileTrajectoryDataset(
        [current, replay], temporal_bin_map={"7:12": 3, "6:949": 1}
    )
    assert dataset[0][0]["progress_bin"].item() == 3
    assert dataset[1][0]["progress_bin"].item() == 1


def test_gate_zero_matches_full_mean_for_head_and_transformer():
    head = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
    transformer = torch.nn.Parameter(torch.tensor([0.5, -0.1]))
    head_coefficients = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, -1.0], [-1.0, 0.0]]
    )
    transformer_coefficients = torch.tensor(
        [[0.2, 0.7], [-0.4, 0.1], [0.3, -0.2], [0.8, 0.5]]
    )
    losses = make_losses(
        head, head_coefficients, transformer, transformer_coefficients
    )
    delta, _ = corrections(losses, [7, 6, 7, 6], [0, 0, 1, 1], head, strength=0.0)
    losses.mean().backward()
    temporal_head = head.grad + delta[0]
    assert torch.equal(temporal_head, head_coefficients.mean(dim=0))
    assert torch.equal(transformer.grad, transformer_coefficients.mean(dim=0))


def test_same_bin_isolation_projects_only_negative_local_relation():
    head = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
    coefficients = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, -1.0], [-1.0, 0.0]]
    )
    losses = make_losses(head, coefficients)
    delta, results = corrections(losses, [7, 6, 7, 6], [0, 0, 1, 1], head)
    by_bin = {result["bin"]: result for result in results}
    assert not by_bin[0]["metrics"]["gate_fired"]
    assert by_bin[1]["metrics"]["gate_fired"]
    assert torch.allclose(delta[0], torch.tensor([-0.25, 0.0]))


def test_no_same_bin_anchor_keeps_baseline_gradient():
    head = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
    coefficients = torch.tensor([[1.0, -1.0], [-1.0, 0.0]])
    losses = make_losses(head, coefficients)
    delta, results = corrections(losses, [7, 6], [0, 1], head)
    losses.mean().backward()
    assert results == []
    assert torch.count_nonzero(delta[0]).item() == 0
    assert torch.equal(head.grad, coefficients.mean(dim=0))


def test_unresolved_samples_remain_in_loss_but_not_gate():
    head = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
    coefficients = torch.tensor([[2.0, -1.0], [-3.0, 4.0]])
    losses = make_losses(head, coefficients)
    delta, results = corrections(losses, [7, 6], [-1, -1], head)
    losses.mean().backward()
    assert results == []
    assert torch.count_nonzero(delta[0]).item() == 0
    assert torch.equal(head.grad, coefficients.mean(dim=0))


def test_gate_changes_head_but_leaves_transformer_at_baseline():
    head = torch.nn.Parameter(torch.tensor([0.2, -0.4]))
    transformer = torch.nn.Parameter(torch.tensor([0.5, -0.1]))
    head_coefficients = torch.tensor([[1.0, -1.0], [-1.0, 0.0]])
    transformer_coefficients = torch.tensor([[0.3, -0.2], [0.8, 0.5]])
    losses = make_losses(
        head, head_coefficients, transformer, transformer_coefficients
    )
    delta, results = corrections(losses, [7, 6], [2, 2], head)
    losses.mean().backward()
    baseline_head = head_coefficients.mean(dim=0)
    baseline_transformer = transformer_coefficients.mean(dim=0)
    assert results[0]["metrics"]["gate_fired"]
    assert not torch.equal(head.grad + delta[0], baseline_head)
    assert torch.equal(transformer.grad, baseline_transformer)
