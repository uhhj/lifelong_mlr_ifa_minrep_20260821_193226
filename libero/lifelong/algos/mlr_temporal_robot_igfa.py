import csv
import json
import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from libero.lifelong.algos.mlr import MLR, build_dataset_for_task
from libero.lifelong.algos.mlr_static_robot_igfa import signed_replay_projection


BIN_FIELDS = [
    "seed", "stage", "epoch", "bin", "candidate_bin_pairs",
    "conflict_bin_pairs", "gate_fire_rate", "mean_cos",
    "mean_conflict_cos", "mean_projection_cost", "mean_new_bin_count",
    "mean_replay_bin_count", "mean_new_head_grad_norm",
    "mean_replay_head_grad_norm", "zero_anchor_count",
]
OVERALL_FIELDS = [
    "seed", "stage", "epoch", "total_batches", "resolved_current_fraction",
    "resolved_replay_fraction", "batches_with_any_same_bin_anchor",
    "same_bin_anchor_batch_fraction", "mean_candidate_bins_per_batch",
    "mean_gated_bins_per_batch",
]


def temporal_head_corrections(
    loss_vector,
    task_ids,
    progress_bins,
    current_task_id,
    head_params,
    gate_strength=1.0,
    n_bins=5,
):
    """Return full-batch-mean policy-head corrections and per-bin metrics."""
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
            loss_vector[new_mask].mean(),
            head_params,
            retain_graph=True,
            create_graph=False,
        )
        g_replay = torch.autograd.grad(
            loss_vector[replay_bin_mask].mean(),
            head_params,
            retain_graph=True,
            create_graph=False,
        )
        gated, metrics = signed_replay_projection(
            g_new, g_replay, gate_strength=gate_strength
        )
        weight_new = n_new / batch_size
        for index, (projected, original) in enumerate(zip(gated, g_new)):
            total_delta[index].add_(
                (projected - original).detach(), alpha=weight_new
            )
        results.append(
            {
                "bin": bin_id,
                "n_new": n_new,
                "n_replay": n_replay,
                "metrics": metrics,
            }
        )
    return total_delta, results


class MLRTemporalRobotIGFA(MLR):
    """MLR with same-progress signed replay projection on the policy head."""

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
        return {
            "candidate": 0,
            "conflict": 0,
            "cos_sum": 0.0,
            "conflict_cos_sum": 0.0,
            "projection_cost_sum": 0.0,
            "new_count_sum": 0,
            "replay_count_sum": 0,
            "new_norm_sum": 0.0,
            "replay_norm_sum": 0.0,
            "zero_anchor": 0,
        }

    @staticmethod
    def _empty_overall_stats():
        return {
            "batches": 0,
            "current": 0,
            "replay": 0,
            "resolved_current": 0,
            "resolved_replay": 0,
            "batches_with_anchor": 0,
            "candidate_bins": 0,
            "gated_bins": 0,
        }

    def build_training_dataset(self, current_task_id):
        return build_dataset_for_task(
            self.cfg.feature_dir,
            current_task_id,
            self.cfg.buffer_dir,
            self.cfg.seed,
            start=self.cfg.start_task,
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
        overall["gated_bins"] += sum(
            int(result["metrics"]["gate_fired"]) for result in results
        )

        for result in results:
            item = self._bin_stats[(int(epoch), int(result["bin"]))]
            metrics = result["metrics"]
            cosine = metrics["cosine"].detach().item()
            item["candidate"] += 1
            item["new_count_sum"] += result["n_new"]
            item["replay_count_sum"] += result["n_replay"]
            if math.isfinite(cosine):
                item["cos_sum"] += cosine
            if metrics["conflict"]:
                item["conflict"] += 1
                if math.isfinite(cosine):
                    item["conflict_cos_sum"] += cosine
            item["projection_cost_sum"] += metrics["projection_cost"].detach().item()
            item["new_norm_sum"] += math.sqrt(metrics["new_norm2"].detach().item())
            item["replay_norm_sum"] += math.sqrt(metrics["replay_norm2"].detach().item())
            item["zero_anchor"] += int(metrics["zero_anchor"])

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
                candidates = item["candidate"]
                conflicts = item["conflict"]
                bin_rows.append(
                    {
                        "seed": int(self.cfg.seed),
                        "stage": int(self._gate_stage),
                        "epoch": epoch,
                        "bin": bin_id,
                        "candidate_bin_pairs": candidates,
                        "conflict_bin_pairs": conflicts,
                        "gate_fire_rate": "" if not candidates else conflicts / candidates,
                        "mean_cos": "" if not candidates else item["cos_sum"] / candidates,
                        "mean_conflict_cos": "" if not conflicts else item["conflict_cos_sum"] / conflicts,
                        "mean_projection_cost": "" if not candidates else item["projection_cost_sum"] / candidates,
                        "mean_new_bin_count": "" if not candidates else item["new_count_sum"] / candidates,
                        "mean_replay_bin_count": "" if not candidates else item["replay_count_sum"] / candidates,
                        "mean_new_head_grad_norm": "" if not candidates else item["new_norm_sum"] / candidates,
                        "mean_replay_head_grad_norm": "" if not candidates else item["replay_norm_sum"] / candidates,
                        "zero_anchor_count": item["zero_anchor"],
                    }
                )
        overall_rows = []
        for epoch in range(int(self.cfg.train.n_epochs)):
            item = self._overall_stats[epoch]
            batches = item["batches"]
            overall_rows.append(
                {
                    "seed": int(self.cfg.seed),
                    "stage": int(self._gate_stage),
                    "epoch": epoch,
                    "total_batches": batches,
                    "resolved_current_fraction": item["resolved_current"] / item["current"],
                    "resolved_replay_fraction": "" if not item["replay"] else item["resolved_replay"] / item["replay"],
                    "batches_with_any_same_bin_anchor": item["batches_with_anchor"],
                    "same_bin_anchor_batch_fraction": item["batches_with_anchor"] / batches,
                    "mean_candidate_bins_per_batch": item["candidate_bins"] / batches,
                    "mean_gated_bins_per_batch": item["gated_bins"] / batches,
                }
            )
        bin_path = os.path.join(self.experiment_dir, "temporal_igfa_gate_stats.csv")
        overall_path = os.path.join(self.experiment_dir, "temporal_igfa_gate_overall.csv")
        self._append_rows(bin_path, BIN_FIELDS, bin_rows)
        self._append_rows(overall_path, OVERALL_FIELDS, overall_rows)
        print(
            f"[temporal-igfa] stage={self._gate_stage} "
            f"bin_stats={bin_path} overall_stats={overall_path}"
        )

    def learn_all_tasks(self, datasets, benchmark, result_summary, current_task_id):
        self._gate_stage = int(current_task_id)
        self._bin_stats = defaultdict(self._empty_bin_stats)
        self._overall_stats = defaultdict(self._empty_overall_stats)
        result = super().learn_all_tasks(
            datasets, benchmark, result_summary, current_task_id
        )
        self._flush_gate_stats()
        return result

    def observe(self, data, current_task_id, epoch, ifa_info=None):
        data[0] = self.map_tensor_to_device(data[0])
        self.optimizer.zero_grad()
        module = self.policy.module if isinstance(self.policy, DDP) else self.policy
        feature = data[0]["feature"].squeeze(1)
        actions = data[0]["actions"].squeeze(1)
        distribution, _ = module(feature)
        loss_vector = module.policy_head.loss_fn(
            distribution, actions, reduction="none"
        )
        loss_total = loss_vector.mean()

        task_ids = torch.tensor(
            [int(value) for value in data[1]],
            dtype=torch.long,
            device=loss_vector.device,
        )
        progress_bins = data[0]["progress_bin"].to(loss_vector.device).long().view(-1)
        head_params = [
            parameter
            for parameter in module.policy_head.parameters()
            if parameter.requires_grad
        ]
        total_delta, results = temporal_head_corrections(
            loss_vector,
            task_ids,
            progress_bins,
            current_task_id,
            head_params,
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
