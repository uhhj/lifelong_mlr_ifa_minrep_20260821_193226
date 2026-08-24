from pathlib import Path

import numpy as np
import pytest
import torch

from libero.lifelong.analysis.temporal_interference import (
    action_key,
    annotate_feature_records,
    assign_progress_bin,
    build_raw_temporal_index,
    enumerate_natural_pairs,
    gradient_cosine,
)


@pytest.mark.parametrize(
    "progress, expected",
    [(0.0, 0), (0.199, 0), (0.2, 1), (0.399, 1), (0.4, 2), (0.6, 3), (0.8, 4), (1.0, 4)],
)
def test_progress_binning(progress, expected):
    assert assign_progress_bin(progress) == expected


def test_gradient_cosine():
    first = torch.tensor([1.0, 0.0])
    assert gradient_cosine(first, first)[0] == pytest.approx(1.0)
    assert gradient_cosine(first, -first)[0] == pytest.approx(-1.0)
    assert gradient_cosine(first, torch.tensor([0.0, 1.0]))[0] == pytest.approx(0.0)


class FakeSequenceDataset:
    def __init__(self, actions):
        self.actions = actions
        self._index_to_demo_id = {0: "demo_0", 1: "demo_0"}
        self._demo_id_to_start_indices = {"demo_0": 0}
        self._demo_id_to_demo_length = {"demo_0": 2}

    def __len__(self):
        return len(self.actions)

    def __getitem__(self, index):
        return {"actions": self.actions[index]}


def test_action_alignment(tmp_path):
    actions = [np.zeros((8, 7), dtype=np.float32), np.ones((8, 7), dtype=np.float32)]
    raw_index = build_raw_temporal_index(FakeSequenceDataset(actions))
    task_dir = tmp_path / "6"
    task_dir.mkdir()
    for index, action in enumerate(reversed(actions)):
        torch.save(
            {"actions": torch.from_numpy(action).unsqueeze(0), "feature": torch.zeros(1, 8, 4, 512)},
            task_dir / f"{index}.pt",
        )
    records, summary = annotate_feature_records(tmp_path, raw_index, 6)
    assert summary == {
        "feature_record_count": 2,
        "matched_count": 2,
        "ambiguous_count": 0,
        "unmatched_count": 0,
        "coverage": 1.0,
    }
    assert {record["progress"] for record in records} == {0.0, 1.0}
    assert action_key(actions[0]) in raw_index


def test_natural_transition_enumeration():
    assert enumerate_natural_pairs() == [(6, 7), (6, 8), (7, 8), (6, 9), (7, 9), (8, 9)]
