from collections import defaultdict

import pytest
import torch

import libero.lifelong.algos.mlr_zero_threshold_directional_guard as formal
from libero.lifelong.algos.mlr_temporal_robot_igfa import (
    MLRTemporalRobotIGFA,
)
from libero.lifelong.algos.mlr_zero_threshold_directional_guard import (
    MLRZeroThresholdDirectionalGuard,
    directional_stats_row,
    select_replay_score_probes,
)


def make_buffer(path, task_ids=(6, 7), size=80):
    path.mkdir()
    for task in task_ids:
        for index in range(size):
            (path / f"t{task}__{index}.pt").touch()
    (path / "unrelated.pt").touch()


def test_replay_probe_selection_is_deterministic_old_task_only(tmp_path):
    buffer_dir = tmp_path / "buffer"
    make_buffer(buffer_dir, task_ids=(6, 7, 8))
    first = select_replay_score_probes(buffer_dir, 42, 8)
    second = select_replay_score_probes(buffer_dir, 42, 8)
    assert first == second
    assert set(first) == {6, 7}
    assert all(len(paths) == 64 for paths in first.values())
    assert all(
        path.parent == buffer_dir
        for paths in first.values()
        for path in paths
    )


def test_stage6_has_no_score_probes(tmp_path):
    buffer_dir = tmp_path / "buffer"
    make_buffer(buffer_dir, task_ids=(6,))
    assert select_replay_score_probes(buffer_dir, 42, 6) == {}


def test_stage8_groups_tasks6_and7_and_rejects_short_replay(tmp_path):
    buffer_dir = tmp_path / "buffer"
    make_buffer(buffer_dir, task_ids=(6, 7), size=64)
    assert set(select_replay_score_probes(buffer_dir, 44, 8)) == {6, 7}
    (buffer_dir / "t7__0.pt").unlink()
    with pytest.raises(ValueError, match="old task 7 has only 63"):
        select_replay_score_probes(buffer_dir, 44, 8)


def test_observe_intercepts_one_adamw_step_at_tau0_scale_half(monkeypatch):
    algo = object.__new__(MLRZeroThresholdDirectionalGuard)
    torch.nn.Module.__init__(algo)
    algo.policy = torch.nn.Linear(1, 1, bias=False)
    algo.optimizer = torch.optim.AdamW(algo.policy.parameters(), lr=1e-3)
    algo.guard_score_probe_batch_size = 32
    algo._guard_score_probes = {6: ["probe"]}
    algo._directional_stats = defaultdict(algo._empty_directional_stats)
    calls = {"guard": 0, "step": 0}

    def fake_gradient(policy, paths, batch_size):
        assert policy is algo.policy
        assert paths == ["probe"]
        assert batch_size == 32
        return {
            "names": ["weight"],
            "gradient": (torch.ones_like(algo.policy.weight),),
        }

    def fake_guard(
        optimizer,
        named_parameters,
        old_gradient_by_name,
        threshold,
        scale,
        enabled,
        step_fn,
        step_args,
        step_kwargs,
    ):
        calls["guard"] += 1
        assert optimizer is algo.optimizer
        assert [name for name, _ in named_parameters] == ["weight"]
        assert set(old_gradient_by_name) == {"weight"}
        assert threshold == 0.0
        assert scale == 0.5
        assert enabled is True
        result = step_fn(*step_args, **step_kwargs)
        calls["step"] += 1
        return result, {
            "candidate_step_cosine": 0.25,
            "candidate_step_score": 0.5,
            "old_grad_norm": 1.0,
            "candidate_delta_norm": 0.1,
            "guard_fire": 1,
        }

    def fake_parent_observe(self, data, current_task_id, epoch, ifa_info=None):
        del data, current_task_id, epoch, ifa_info
        self.optimizer.zero_grad()
        loss = (self.policy.weight ** 2).sum()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    monkeypatch.setattr(formal, "mean_backend_gradient_device", fake_gradient)
    monkeypatch.setattr(formal, "guarded_optimizer_step", fake_guard)
    monkeypatch.setattr(
        MLRTemporalRobotIGFA, "observe", fake_parent_observe
    )
    loss = algo.observe(None, current_task_id=7, epoch=0)
    assert loss >= 0
    assert calls == {"guard": 1, "step": 1}
    assert algo._directional_stats[0]["eligible"] == 1
    assert algo._directional_stats[0]["fires"] == 1


def test_directional_stats_row_uses_eligible_and_fire_denominators():
    row = directional_stats_row(
        {
            "batches": 4,
            "eligible": 4,
            "fires": 2,
            "cosine_sum": 1.0,
            "fired_cosine_sum": 0.8,
            "score_sum": 2.0,
            "old_grad_norm_sum": 8.0,
            "delta_norm_sum": 4.0,
        },
        seed=42,
        stage=7,
        epoch=3,
        score_probe_size=64,
        score_probe_batch_size=32,
        threshold=0.0,
        guard_scale=0.5,
    )
    assert row["fire_rate"] == pytest.approx(0.5)
    assert row["mean_candidate_cosine"] == pytest.approx(0.25)
    assert row["mean_fired_cosine"] == pytest.approx(0.4)
    assert row["mean_candidate_score"] == pytest.approx(0.5)
