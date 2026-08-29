import math
import os
import random
import re
from collections import defaultdict
from pathlib import Path

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from libero.lifelong.algos.mlr_temporal_robot_igfa import (
    MLRTemporalRobotIGFA,
)
from libero.lifelong.analysis.candidate_directional_guard import (
    guarded_optimizer_step,
)
from libero.lifelong.analysis.online_first_order_margin import (
    equal_task_gradient,
    mean_backend_gradient_device,
)


SCORE_PROBE_SEED = 20260831
REPLAY_TASK_RE = re.compile(r"^t(\d+)__")
DIRECTIONAL_FIELDS = [
    "seed",
    "stage",
    "epoch",
    "total_batches",
    "eligible_batches",
    "fire_count",
    "fire_rate",
    "mean_candidate_cosine",
    "mean_fired_cosine",
    "mean_candidate_score",
    "mean_old_grad_norm",
    "mean_candidate_delta_norm",
    "score_probe_size",
    "score_probe_batch_size",
    "threshold",
    "guard_scale",
]


def replay_probe_inventory(buffer_dir, current_stage):
    groups = {
        old_task: [] for old_task in range(6, int(current_stage))
    }
    for path in sorted(Path(buffer_dir).glob("*.pt")):
        match = REPLAY_TASK_RE.match(path.name)
        if match is None:
            continue
        old_task = int(match.group(1))
        if old_task in groups:
            groups[old_task].append(path)
    return groups


def select_replay_score_probes(
    buffer_dir,
    seed,
    current_stage,
    score_size=64,
):
    groups = replay_probe_inventory(buffer_dir, current_stage)
    selected = {}
    for old_task in sorted(groups):
        paths = groups[old_task]
        if len(paths) < int(score_size):
            raise ValueError(
                f"stage {current_stage} old task {old_task} has only "
                f"{len(paths)} replay records; need {score_size}"
            )
        chosen = random.Random(
            SCORE_PROBE_SEED + int(seed) * 100 + int(old_task)
        ).sample(paths, int(score_size))
        selected[old_task] = sorted(chosen, key=lambda path: path.name)
    return selected


def directional_stats_row(
    item,
    *,
    seed,
    stage,
    epoch,
    score_probe_size,
    score_probe_batch_size,
    threshold,
    guard_scale,
):
    eligible = int(item["eligible"])
    fires = int(item["fires"])
    return {
        "seed": int(seed),
        "stage": int(stage),
        "epoch": int(epoch),
        "total_batches": int(item["batches"]),
        "eligible_batches": eligible,
        "fire_count": fires,
        "fire_rate": "" if not eligible else fires / eligible,
        "mean_candidate_cosine": (
            "" if not eligible else item["cosine_sum"] / eligible
        ),
        "mean_fired_cosine": (
            "" if not fires else item["fired_cosine_sum"] / fires
        ),
        "mean_candidate_score": (
            "" if not eligible else item["score_sum"] / eligible
        ),
        "mean_old_grad_norm": (
            "" if not eligible else item["old_grad_norm_sum"] / eligible
        ),
        "mean_candidate_delta_norm": (
            "" if not eligible else item["delta_norm_sum"] / eligible
        ),
        "score_probe_size": int(score_probe_size),
        "score_probe_batch_size": int(score_probe_batch_size),
        "threshold": float(threshold),
        "guard_scale": float(guard_scale),
    }


class MLRZeroThresholdDirectionalGuard(MLRTemporalRobotIGFA):
    """Temporal Robot-IGFA V1 with the preregistered tau=0 AdamW guard."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.guard_threshold = float(
            getattr(cfg, "directional_guard_threshold", 0.0)
        )
        self.guard_scale = float(
            getattr(cfg, "directional_guard_scale", 0.5)
        )
        self.guard_score_probe_size = int(
            getattr(cfg, "directional_guard_score_probe_size", 64)
        )
        self.guard_score_probe_batch_size = int(
            getattr(cfg, "directional_guard_score_probe_batch_size", 32)
        )
        if self.guard_threshold != 0.0:
            raise ValueError("formal zero-threshold guard requires tau=0")
        if self.guard_scale != 0.5:
            raise ValueError("formal zero-threshold guard requires scale=0.5")
        if self.guard_score_probe_size != 64:
            raise ValueError("formal score probe size must be 64")
        self._directional_stage = None
        self._guard_score_probes = {}
        self._directional_stats = defaultdict(self._empty_directional_stats)

    @staticmethod
    def _empty_directional_stats():
        return {
            "batches": 0,
            "eligible": 0,
            "fires": 0,
            "cosine_sum": 0.0,
            "fired_cosine_sum": 0.0,
            "score_sum": 0.0,
            "old_grad_norm_sum": 0.0,
            "delta_norm_sum": 0.0,
        }

    def _flush_directional_stats(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        rows = [
            directional_stats_row(
                self._directional_stats[epoch],
                seed=self.cfg.seed,
                stage=self._directional_stage,
                epoch=epoch,
                score_probe_size=self.guard_score_probe_size,
                score_probe_batch_size=self.guard_score_probe_batch_size,
                threshold=self.guard_threshold,
                guard_scale=self.guard_scale,
            )
            for epoch in range(int(self.cfg.train.n_epochs))
        ]
        path = os.path.join(
            self.experiment_dir, "directional_guard_train_stats.csv"
        )
        self._append_rows(path, DIRECTIONAL_FIELDS, rows)
        print(
            f"[directional-guard] stage={self._directional_stage} "
            f"stats={path}"
        )

    def learn_all_tasks(
        self,
        datasets,
        benchmark,
        result_summary,
        current_task_id,
    ):
        self._directional_stage = int(current_task_id)
        self._directional_stats = defaultdict(self._empty_directional_stats)
        inventory = replay_probe_inventory(
            self.cfg.buffer_dir, current_task_id
        )
        self._guard_score_probes = select_replay_score_probes(
            self.cfg.buffer_dir,
            self.cfg.seed,
            current_task_id,
            self.guard_score_probe_size,
        )
        print(
            "[directional-guard-probes] "
            f"seed={self.cfg.seed} stage={current_task_id} "
            f"old_tasks={sorted(self._guard_score_probes)} "
            f"replay_counts="
            f"{dict((task, len(paths)) for task, paths in inventory.items())} "
            f"selected={self.guard_score_probe_size}/task threshold=0 "
            "scale=0.5 source=replay_buffer"
        )
        result = super().learn_all_tasks(
            datasets,
            benchmark,
            result_summary,
            current_task_id,
        )
        self._flush_directional_stats()
        return result

    def observe(self, data, current_task_id, epoch, ifa_info=None):
        item = self._directional_stats[int(epoch)]
        if not self._guard_score_probes:
            loss = super().observe(
                data,
                current_task_id,
                epoch,
                ifa_info=ifa_info,
            )
            item["batches"] += 1
            return loss

        gradient_policy = (
            self.policy.module if isinstance(self.policy, DDP) else self.policy
        )
        task_gradient_results = {
            int(task_id): mean_backend_gradient_device(
                gradient_policy,
                paths,
                batch_size=self.guard_score_probe_batch_size,
            )
            for task_id, paths in sorted(self._guard_score_probes.items())
        }
        names = next(iter(task_gradient_results.values()))["names"]
        if any(
            result["names"] != names
            for result in task_gradient_results.values()
        ):
            raise ValueError(
                "backend name order differs across old-task probes"
            )
        aggregate_gradient = equal_task_gradient(
            {
                task_id: result["gradient"]
                for task_id, result in task_gradient_results.items()
            }
        )
        old_gradient_by_name = dict(zip(names, aggregate_gradient))
        module = (
            self.policy.module if isinstance(self.policy, DDP) else self.policy
        )
        named_trainable = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]
        captured = {}
        original_step = self.optimizer.step

        def wrapped_step(*args, **kwargs):
            result, metrics = guarded_optimizer_step(
                self.optimizer,
                named_trainable,
                old_gradient_by_name,
                threshold=0.0,
                scale=0.5,
                enabled=True,
                step_fn=original_step,
                step_args=args,
                step_kwargs=kwargs,
            )
            captured.update(metrics)
            return result

        self.optimizer.step = wrapped_step
        try:
            loss = super().observe(
                data,
                current_task_id,
                epoch,
                ifa_info=ifa_info,
            )
        finally:
            self.optimizer.step = original_step
        if not captured:
            raise RuntimeError(
                "directional guard optimizer.step interception failed"
            )
        values = [
            captured["candidate_step_cosine"],
            captured["candidate_step_score"],
            captured["old_grad_norm"],
            captured["candidate_delta_norm"],
        ]
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("non-finite directional guard metrics")
        item["batches"] += 1
        item["eligible"] += 1
        item["fires"] += int(captured["guard_fire"])
        item["cosine_sum"] += float(captured["candidate_step_cosine"])
        item["score_sum"] += float(captured["candidate_step_score"])
        item["old_grad_norm_sum"] += float(captured["old_grad_norm"])
        item["delta_norm_sum"] += float(captured["candidate_delta_norm"])
        if captured["guard_fire"]:
            item["fired_cosine_sum"] += float(
                captured["candidate_step_cosine"]
            )
        return loss
