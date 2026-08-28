import csv
import itertools
import json
import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from libero.lifelong.algos.mlr import MLR, build_dataset_for_task


BIN_FIELDS = [
    "seed", "stage", "epoch", "bin", "candidate_bins",
    "multi_old_candidate_bins", "present_old_constraints",
    "initial_conflicting_constraints", "active_constraints",
    "projection_fire_rate", "mean_present_old_tasks",
    "mean_initial_conflicts", "mean_active_constraints",
    "mean_projection_cost", "mean_worst_pre_cos", "mean_worst_post_cos",
    "post_constraint_violation_count", "zero_old_gradient_count",
    "compatible_became_active_count", "active_count_0", "active_count_1",
    "active_count_2", "active_count_3",
]
OVERALL_FIELDS = [
    "seed", "stage", "epoch", "total_batches", "resolved_current_fraction",
    "resolved_replay_fraction", "batches_with_any_same_bin_anchor",
    "mean_candidate_bins_per_batch", "mean_projected_bins_per_batch",
    "constraint_count_0", "constraint_count_1", "constraint_count_2",
    "constraint_count_3",
]
RELATIVE_TOLERANCE = 1e-6


def gradient_dot(first, second):
    """Dot product with float32 products and float64 scalar accumulation."""
    return sum(
        (torch.sum(a * b, dtype=torch.float64) for a, b in zip(first, second)),
        start=torch.zeros((), dtype=torch.float64, device=first[0].device),
    )


def project_gradient_to_task_halfspaces(
    g_new,
    old_gradients,
    gate_strength=1.0,
    eps=1e-12,
):
    """Minimum-change projection onto all present old-task half-spaces."""
    if not 0.0 <= float(gate_strength) <= 1.0:
        raise ValueError("gate_strength must be in [0, 1]")

    ordered = sorted(old_gradients, key=lambda item: int(item["task_id"]))
    usable = []
    zero_old_gradient_count = 0
    for item in ordered:
        norm2 = gradient_dot(item["gradient"], item["gradient"])
        if float(norm2.detach().item()) <= eps:
            zero_old_gradient_count += 1
        else:
            usable.append({**item, "norm2": norm2})

    if not usable or float(gate_strength) == 0.0:
        return tuple(g_new), {
            "old_tasks": [int(item["task_id"]) for item in usable],
            "initial_conflicting_tasks": [],
            "active_constraint_tasks": [],
            "projection_fired": False,
            "projection_cost": 0.0,
            "worst_pre_cos": float("nan"),
            "worst_post_cos": float("nan"),
            "post_violation_count": 0,
            "zero_old_gradient_count": zero_old_gradient_count,
            "compatible_became_active_count": 0,
        }

    old = [item["gradient"] for item in usable]
    task_ids = [int(item["task_id"]) for item in usable]
    d = torch.stack([gradient_dot(g_new, gradient) for gradient in old])
    gram = torch.stack([
        torch.stack([gradient_dot(first, second) for second in old])
        for first in old
    ])
    scale = max(
        1.0,
        float(torch.max(torch.abs(d)).detach().item()),
        float(torch.max(torch.abs(gram)).detach().item()),
    )
    # The scalar solve is float64, but the correction is applied to float32
    # parameter tensors. One shared scale-relative tolerance covers both steps.
    tol = RELATIVE_TOLERANCE * scale
    initial_conflicting = [
        task_ids[index] for index, value in enumerate(d) if float(value.item()) < 0.0
    ]

    if not initial_conflicting:
        projected = tuple(g_new)
        lambdas = torch.zeros(len(old), dtype=torch.float64, device=d.device)
    else:
        candidates = []
        for size in range(1, len(old) + 1):
            for subset in itertools.combinations(range(len(old)), size):
                indices = torch.tensor(subset, dtype=torch.long, device=d.device)
                gram_ss = gram.index_select(0, indices).index_select(1, indices)
                d_s = d.index_select(0, indices)
                lambda_s = torch.linalg.pinv(gram_ss) @ (-d_s)
                lambda_full = torch.zeros(len(old), dtype=torch.float64, device=d.device)
                lambda_full[indices] = lambda_s
                residual = d + gram @ lambda_full
                if float(torch.min(lambda_full).item()) < -tol:
                    continue
                if float(torch.min(residual).item()) < -tol:
                    continue
                if float(torch.max(torch.abs(residual[indices])).item()) > tol:
                    continue
                cost2 = float((lambda_full @ gram @ lambda_full).detach().item())
                candidates.append((max(cost2, 0.0), tuple(subset), lambda_full))
        if not candidates:
            raise RuntimeError(
                "no feasible active-set candidate for per-task temporal projection"
            )
        _, _, lambdas = min(candidates, key=lambda item: (item[0], item[1]))
        correction = [torch.zeros_like(value) for value in g_new]
        for coefficient, gradient in zip(lambdas, old):
            for index, value in enumerate(gradient):
                correction[index].add_(value, alpha=float(coefficient.item()))
        projected = tuple(
            original + float(gate_strength) * delta
            for original, delta in zip(g_new, correction)
        )

    pre_dots = d
    post_dots = torch.stack([gradient_dot(projected, gradient) for gradient in old])
    post_scale = max(scale, float(torch.max(torch.abs(post_dots)).detach().item()))
    post_tol = RELATIVE_TOLERANCE * post_scale
    post_violation_count = int((post_dots < -post_tol).sum().item())
    if float(gate_strength) == 1.0 and post_violation_count:
        raise RuntimeError(
            f"per-task projection left {post_violation_count} constraint violations"
        )

    new_norm2 = gradient_dot(g_new, g_new)
    projected_delta = tuple(after - before for after, before in zip(projected, g_new))
    correction_norm2 = gradient_dot(projected_delta, projected_delta)
    projection_cost = math.sqrt(max(float(correction_norm2.detach().item()), 0.0)) / (
        math.sqrt(max(float(new_norm2.detach().item()), 0.0)) + eps
    )
    new_norm = math.sqrt(max(float(new_norm2.detach().item()), 0.0))
    projected_norm = math.sqrt(
        max(float(gradient_dot(projected, projected).detach().item()), 0.0)
    )
    pre_cos = []
    post_cos = []
    for index, item in enumerate(usable):
        old_norm = math.sqrt(max(float(item["norm2"].detach().item()), 0.0))
        pre_cos.append(float(pre_dots[index].item()) / (new_norm * old_norm + eps))
        post_cos.append(float(post_dots[index].item()) / (projected_norm * old_norm + eps))
    active = [
        task_ids[index]
        for index, value in enumerate(lambdas)
        if float(value.detach().item()) > tol
    ]
    compatible_became_active = sum(
        task_ids[index] in active and float(d[index].item()) >= 0.0
        for index in range(len(task_ids))
    )
    return projected, {
        "old_tasks": task_ids,
        "initial_conflicting_tasks": initial_conflicting,
        "active_constraint_tasks": active,
        "projection_fired": projection_cost > 0.0,
        "projection_cost": projection_cost,
        "worst_pre_cos": min(pre_cos),
        "worst_post_cos": min(post_cos),
        "post_violation_count": post_violation_count,
        "zero_old_gradient_count": zero_old_gradient_count,
        "compatible_became_active_count": compatible_became_active,
    }


def per_task_temporal_head_corrections(
    loss_vector,
    task_ids,
    progress_bins,
    current_task_id,
    head_params,
    gate_strength=1.0,
    n_bins=5,
):
    """Return full-batch-mean policy-head corrections and per-bin results."""
    current_mask = task_ids == int(current_task_id)
    replay_mask = ~current_mask
    resolved_mask = progress_bins >= 0
    batch_size = len(task_ids)
    total_delta = [torch.zeros_like(parameter) for parameter in head_params]
    results = []

    for bin_id in range(n_bins):
        new_mask = current_mask & resolved_mask & (progress_bins == bin_id)
        replay_bin_mask = replay_mask & resolved_mask & (progress_bins == bin_id)
        n_new = int(new_mask.sum().item())
        n_replay = int(replay_bin_mask.sum().item())
        if n_new == 0 or n_replay == 0:
            continue

        g_new = torch.autograd.grad(
            loss_vector[new_mask].mean(), head_params,
            retain_graph=True, create_graph=False,
        )
        present = sorted(int(value) for value in torch.unique(task_ids[replay_bin_mask]).tolist())
        old_gradients = []
        for old_task in present:
            old_mask = replay_bin_mask & (task_ids == old_task)
            gradient = torch.autograd.grad(
                loss_vector[old_mask].mean(), head_params,
                retain_graph=True, create_graph=False,
            )
            old_gradients.append({
                "task_id": old_task,
                "gradient": gradient,
                "n_samples": int(old_mask.sum().item()),
            })
        projected, metrics = project_gradient_to_task_halfspaces(
            g_new, old_gradients, gate_strength=gate_strength,
        )
        weight_new = n_new / batch_size
        for index, (after, before) in enumerate(zip(projected, g_new)):
            total_delta[index].add_((after - before).detach(), alpha=weight_new)
        results.append({
            "bin": bin_id,
            "n_new": n_new,
            "n_replay": n_replay,
            "old_tasks_present": metrics["old_tasks"],
            "n_old_tasks": len(metrics["old_tasks"]),
            "initial_conflicting_tasks": metrics["initial_conflicting_tasks"],
            "active_constraint_tasks": metrics["active_constraint_tasks"],
            "projection_fired": metrics["projection_fired"],
            "projection_cost": metrics["projection_cost"],
            "worst_pre_cos": metrics["worst_pre_cos"],
            "worst_post_cos": metrics["worst_post_cos"],
            "post_violation_count": metrics["post_violation_count"],
            "zero_old_gradient_count": metrics["zero_old_gradient_count"],
            "compatible_became_active_count": metrics["compatible_became_active_count"],
        })
    return total_delta, results


class MLRPerTaskTemporalRobotIGFA(MLR):
    """MLR with per-old-task same-progress constraints on the policy head."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        with open(cfg.temporal_bin_map, encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload["seed"]) != int(cfg.seed):
            raise ValueError(
                f"temporal map seed {payload['seed']} does not match cfg.seed {cfg.seed}"
            )
        if int(payload["n_bins"]) != 5:
            raise ValueError(f"expected 5 temporal bins, got {payload['n_bins']}")
        self.temporal_bin_map = payload["records"]
        self.n_temporal_bins = 5
        self._gate_stage = None
        self._bin_stats = defaultdict(self._empty_bin_stats)
        self._overall_stats = defaultdict(self._empty_overall_stats)

    @staticmethod
    def _empty_bin_stats():
        return defaultdict(float)

    @staticmethod
    def _empty_overall_stats():
        return defaultdict(float)

    def build_training_dataset(self, current_task_id):
        return build_dataset_for_task(
            self.cfg.feature_dir, current_task_id, self.cfg.buffer_dir,
            self.cfg.seed, start=self.cfg.start_task,
            temporal_bin_map=self.temporal_bin_map,
        )

    def _record_stats(self, epoch, task_ids, progress_bins, results):
        current = task_ids == int(self._gate_stage)
        replay = ~current
        resolved = progress_bins >= 0
        overall = self._overall_stats[int(epoch)]
        overall["batches"] += 1
        overall["current"] += int(current.sum().item())
        overall["replay"] += int(replay.sum().item())
        overall["resolved_current"] += int((current & resolved).sum().item())
        overall["resolved_replay"] += int((replay & resolved).sum().item())
        overall["batches_with_anchor"] += int(bool(results))
        overall["candidate_bins"] += len(results)
        overall["projected_bins"] += sum(int(result["projection_fired"]) for result in results)
        for bin_id in range(self.n_temporal_bins):
            new_mask = current & resolved & (progress_bins == bin_id)
            if not bool(new_mask.any()):
                continue
            old_count = len(torch.unique(task_ids[replay & resolved & (progress_bins == bin_id)]))
            overall[f"constraint_count_{min(old_count, 3)}"] += 1

        for result in results:
            item = self._bin_stats[(int(epoch), int(result["bin"]))]
            n_old = result["n_old_tasks"]
            n_conflict = len(result["initial_conflicting_tasks"])
            n_active = len(result["active_constraint_tasks"])
            item["candidate"] += 1
            item["multi_old"] += int(n_old > 1)
            item["present"] += n_old
            item["conflicts"] += n_conflict
            item["active"] += n_active
            item[f"active_count_{min(n_active, 3)}"] += 1
            item["fired"] += int(result["projection_fired"])
            item["projection_cost"] += result["projection_cost"]
            if math.isfinite(result["worst_pre_cos"]):
                item["worst_pre_cos"] += result["worst_pre_cos"]
            if math.isfinite(result["worst_post_cos"]):
                item["worst_post_cos"] += result["worst_post_cos"]
            item["violations"] += result["post_violation_count"]
            item["zero_old"] += result["zero_old_gradient_count"]
            item["compatible_active"] += result["compatible_became_active_count"]

    @staticmethod
    def _append_rows(path, fields, rows):
        exists = os.path.isfile(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerows(rows)

    def _flush_gate_stats(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        bin_rows = []
        for epoch in range(int(self.cfg.train.n_epochs)):
            for bin_id in range(self.n_temporal_bins):
                item = self._bin_stats[(epoch, bin_id)]
                count = int(item["candidate"])
                bin_rows.append({
                    "seed": int(self.cfg.seed), "stage": int(self._gate_stage),
                    "epoch": epoch, "bin": bin_id,
                    "candidate_bins": count,
                    "multi_old_candidate_bins": int(item["multi_old"]),
                    "present_old_constraints": int(item["present"]),
                    "initial_conflicting_constraints": int(item["conflicts"]),
                    "active_constraints": int(item["active"]),
                    "projection_fire_rate": "" if not count else item["fired"] / count,
                    "mean_present_old_tasks": "" if not count else item["present"] / count,
                    "mean_initial_conflicts": "" if not count else item["conflicts"] / count,
                    "mean_active_constraints": "" if not count else item["active"] / count,
                    "mean_projection_cost": "" if not count else item["projection_cost"] / count,
                    "mean_worst_pre_cos": "" if not count else item["worst_pre_cos"] / count,
                    "mean_worst_post_cos": "" if not count else item["worst_post_cos"] / count,
                    "post_constraint_violation_count": int(item["violations"]),
                    "zero_old_gradient_count": int(item["zero_old"]),
                    "compatible_became_active_count": int(item["compatible_active"]),
                    "active_count_0": int(item["active_count_0"]),
                    "active_count_1": int(item["active_count_1"]),
                    "active_count_2": int(item["active_count_2"]),
                    "active_count_3": int(item["active_count_3"]),
                })
        overall_rows = []
        for epoch in range(int(self.cfg.train.n_epochs)):
            item = self._overall_stats[epoch]
            batches = int(item["batches"])
            overall_rows.append({
                "seed": int(self.cfg.seed), "stage": int(self._gate_stage),
                "epoch": epoch, "total_batches": batches,
                "resolved_current_fraction": item["resolved_current"] / item["current"],
                "resolved_replay_fraction": "" if not item["replay"] else item["resolved_replay"] / item["replay"],
                "batches_with_any_same_bin_anchor": int(item["batches_with_anchor"]),
                "mean_candidate_bins_per_batch": item["candidate_bins"] / batches,
                "mean_projected_bins_per_batch": item["projected_bins"] / batches,
                "constraint_count_0": int(item["constraint_count_0"]),
                "constraint_count_1": int(item["constraint_count_1"]),
                "constraint_count_2": int(item["constraint_count_2"]),
                "constraint_count_3": int(item["constraint_count_3"]),
            })
        run_dir = os.path.join(self.experiment_dir)
        bin_path = os.path.join(run_dir, "per_task_temporal_igfa_gate_stats.csv")
        overall_path = os.path.join(run_dir, "per_task_temporal_igfa_gate_overall.csv")
        self._append_rows(bin_path, BIN_FIELDS, bin_rows)
        self._append_rows(overall_path, OVERALL_FIELDS, overall_rows)
        print(
            f"[per-task-temporal-igfa] stage={self._gate_stage} "
            f"bin_stats={bin_path} overall_stats={overall_path}"
        )

    def learn_all_tasks(self, datasets, benchmark, result_summary, current_task_id):
        self._gate_stage = int(current_task_id)
        self._bin_stats = defaultdict(self._empty_bin_stats)
        self._overall_stats = defaultdict(self._empty_overall_stats)
        result = super().learn_all_tasks(datasets, benchmark, result_summary, current_task_id)
        self._flush_gate_stats()
        return result

    def observe(self, data, current_task_id, epoch, ifa_info=None):
        data[0] = self.map_tensor_to_device(data[0])
        self.optimizer.zero_grad()
        module = self.policy.module if isinstance(self.policy, DDP) else self.policy
        feature = data[0]["feature"].squeeze(1)
        actions = data[0]["actions"].squeeze(1)
        distribution, _ = module(feature)
        loss_vector = module.policy_head.loss_fn(distribution, actions, reduction="none")
        loss_total = loss_vector.mean()
        task_ids = torch.tensor(
            [int(value) for value in data[1]], dtype=torch.long,
            device=loss_vector.device,
        )
        progress_bins = data[0]["progress_bin"].to(loss_vector.device).long().view(-1)
        head_params = [
            parameter for parameter in module.policy_head.parameters()
            if parameter.requires_grad
        ]
        total_delta, results = per_task_temporal_head_corrections(
            loss_vector, task_ids, progress_bins, current_task_id, head_params,
            gate_strength=float(getattr(self.cfg, "robot_igfa_gate_strength", 1.0)),
            n_bins=self.n_temporal_bins,
        )
        (self.loss_scale * loss_total).backward()
        for parameter, delta in zip(head_params, total_delta):
            parameter.grad.add_(self.loss_scale * delta)
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        self._record_stats(epoch, task_ids, progress_bins, results)
        return loss_total.item()
