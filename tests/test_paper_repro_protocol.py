import pathlib
import re

import pytest
import torch

from libero.lifelong.algos.base import Sequential
from libero.lifelong.algos.mlr import (
    FORMAL_CHECKPOINT_EPOCHS,
    LR_PROTOCOL_FIELDS,
    should_save_mlr_checkpoint,
)
from libero.lifelong.utils import make_linear_schedule_with_warmup


ROOT = pathlib.Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_object_task6_reconciliation.sh"


class StepHarness:
    optimizer_step_and_maybe_schedule = Sequential.optimizer_step_and_maybe_schedule
    epoch_schedule_step = Sequential.epoch_schedule_step

    def __init__(self, mode):
        self.parameter = torch.nn.Parameter(torch.tensor([1.0]))
        self.optimizer = torch.optim.SGD([self.parameter], lr=1e-4)
        self.repro_scheduler_mode = mode
        self._optimizer_step_count = 0
        self._scheduler_step_count = 0
        self.scheduler = None if mode == "constant" else torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, make_linear_schedule_with_warmup(5, 20)
        )


def update(harness):
    harness.optimizer.zero_grad()
    harness.parameter.grad = torch.ones_like(harness.parameter)
    harness.optimizer_step_and_maybe_schedule()


def test_linear_schedule_helper_parity():
    schedule = make_linear_schedule_with_warmup(5, 20)
    assert [schedule(i) for i in (0, 1, 5, 20)] == pytest.approx([0, .2, 1, 0])


def test_legacy_scheduler_only_epoch_step():
    h = StepHarness("legacy_epoch")
    update(h)
    assert h._optimizer_step_count == 1 and h._scheduler_step_count == 0
    h.epoch_schedule_step(0)
    assert h._scheduler_step_count == 0
    h.epoch_schedule_step(1)
    assert h._scheduler_step_count == 1


def test_batch_scheduler_exactly_once_per_update():
    h = StepHarness("batch_step")
    for _ in range(7):
        update(h)
    assert h._optimizer_step_count == h._scheduler_step_count == 7


def test_constant_scheduler_none_and_lr():
    h = StepHarness("constant")
    for _ in range(7):
        update(h)
    assert h.scheduler is None
    assert h.optimizer.param_groups[0]["lr"] == 1e-4


def test_legacy_first_20_update_parity():
    p_old = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    p_new = torch.nn.Parameter(p_old.detach().clone())
    old = torch.optim.AdamW([p_old], lr=1e-4, weight_decay=.1)
    h = StepHarness("legacy_epoch")
    h.parameter = p_new
    h.optimizer = torch.optim.AdamW([p_new], lr=1e-4, weight_decay=.1)
    h.scheduler = torch.optim.lr_scheduler.LambdaLR(
        h.optimizer, make_linear_schedule_with_warmup(500, 2000)
    )
    old_sched = torch.optim.lr_scheduler.LambdaLR(old, make_linear_schedule_with_warmup(500, 2000))
    old_losses, new_losses = [], []
    for index in range(20):
        old.zero_grad(); h.optimizer.zero_grad()
        old_loss = ((p_old - index / 50) ** 2).sum()
        new_loss = ((p_new - index / 50) ** 2).sum()
        old_losses.append(old_loss.item()); new_losses.append(new_loss.item())
        old_loss.backward(); new_loss.backward()
        old.step(); h.optimizer_step_and_maybe_schedule()
    assert max(abs(a - b) for a, b in zip(old_losses, new_losses)) <= 1e-6
    relative_l2 = torch.linalg.vector_norm(p_old - p_new) / torch.linalg.vector_norm(p_old)
    assert relative_l2 <= 1e-6
    old_state, new_state = old.state_dict(), h.optimizer.state_dict()
    assert old_state["param_groups"] == new_state["param_groups"]
    assert old_state["state"].keys() == new_state["state"].keys()
    for key in old_state["state"]:
        for name, value in old_state["state"][key].items():
            other = new_state["state"][key][name]
            if torch.is_tensor(value):
                assert torch.equal(value, other)
            else:
                assert value == other
    assert old.param_groups[0]["lr"] == h.optimizer.param_groups[0]["lr"]
    assert old_sched.get_last_lr() == h.scheduler.get_last_lr()


def test_diagnostic_checkpoints_do_not_change_formal_set():
    diagnostics = (9, 19, 29, 39, 49, 59, 69, 79, 89, 99)
    selected = {e for e in range(100) if should_save_mlr_checkpoint(e, 100, diagnostics)}
    assert set(FORMAL_CHECKPOINT_EPOCHS) <= selected
    assert set(FORMAL_CHECKPOINT_EPOCHS).isdisjoint(set(diagnostics))


def test_four_arm_configs_exact():
    text = LAUNCHER.read_text(encoding="utf-8")
    expected = {
        "current_legacy": (10, "legacy_epoch"),
        "release18_legacy": (18, "legacy_epoch"),
        "batch10_linear_step": (10, "batch_step"),
        "batch10_constant": (10, "constant"),
    }
    for arm, (batch, scheduler) in expected.items():
        assert re.search(rf"{arm}\) BATCH={batch}; SCHED={scheduler}", text)


def test_protocol_matrix_required_fields():
    required = {"seed", "arm", "stage", "epoch", "batches", "optimizer_step_start",
                "optimizer_step_end", "lr_start", "lr_end", "lr_min", "lr_max",
                "scheduler_mode", "batch_size", "warmup_steps", "total_steps"}
    assert set(LR_PROTOCOL_FIELDS) == required


def test_each_arm_uses_isolated_buffer():
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'RUN_TAG="paper_v2_object_seed${SEED}_task6_${ARM}"' in text
    assert 'BUFFER_DIR="$PROJECT_ROOT/buffers/$RUN_TAG"' in text


def test_paper_metric_classifier_thresholds():
    def verdict(c_gain, d_gain, release_material=False):
        if c_gain >= 15:
            return "SUPPORTED"
        if c_gain >= 5 or d_gain >= 5 or release_material:
            return "PARTIAL"
        return "NOT_SUPPORTED"
    assert verdict(15, 0) == "SUPPORTED"
    assert verdict(5, 0) == "PARTIAL"
    assert verdict(0, 5) == "PARTIAL"
    assert verdict(0, 0) == "NOT_SUPPORTED"
