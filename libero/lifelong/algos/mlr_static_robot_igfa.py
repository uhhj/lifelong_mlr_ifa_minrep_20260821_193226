import csv
import math
import os
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from libero.lifelong.algos.mlr import MLR


GATE_STATS_FIELDS = [
    "seed",
    "stage",
    "epoch",
    "total_batches",
    "paired_batches",
    "conflict_batches",
    "gate_fire_rate",
    "mean_cos",
    "mean_conflict_cos",
    "mean_new_head_grad_norm",
    "mean_replay_head_grad_norm",
    "mean_projection_cost",
    "mean_current_fraction",
    "mean_replay_fraction",
    "zero_anchor_count",
]


def signed_replay_projection(g_new, g_replay, gate_strength=1.0, eps=1e-12):
    dot = sum(torch.sum(new * replay) for new, replay in zip(g_new, g_replay))
    new_norm2 = sum(torch.sum(new * new) for new in g_new)
    replay_norm2 = sum(torch.sum(replay * replay) for replay in g_replay)
    zero_anchor = replay_norm2.detach().item() <= eps
    conflict = dot.detach().item() < 0.0
    gate_fired = conflict and not zero_anchor and gate_strength > 0.0
    if gate_fired:
        coefficient = gate_strength * dot / replay_norm2
        gated = tuple(new - coefficient * replay for new, replay in zip(g_new, g_replay))
    else:
        gated = tuple(g_new)
    denominator = torch.sqrt(new_norm2 * replay_norm2)
    cosine = dot / denominator if denominator.detach().item() > eps else dot.new_tensor(float("nan"))
    projection_norm2 = sum(torch.sum((new - projected) ** 2) for new, projected in zip(g_new, gated))
    projection_cost = torch.sqrt(projection_norm2) / (torch.sqrt(new_norm2) + eps)
    return gated, {
        "dot": dot,
        "new_norm2": new_norm2,
        "replay_norm2": replay_norm2,
        "cosine": cosine,
        "projection_cost": projection_cost,
        "conflict": conflict,
        "gate_fired": gate_fired,
        "zero_anchor": zero_anchor,
    }


def weighted_head_gradient(g_new, g_replay, n_new, n_replay):
    total = n_new + n_replay
    if total <= 0:
        raise ValueError("empty gradient recombination")
    weight_new = n_new / total
    weight_replay = n_replay / total
    return tuple(
        weight_new * new + weight_replay * replay
        for new, replay in zip(g_new, g_replay)
    )


def replace_head_gradients(head_params, gradients, loss_scale):
    for parameter, gradient in zip(head_params, gradients):
        parameter.grad = loss_scale * gradient.detach()


class MLRStaticRobotIGFA(MLR):
    """MLR with a signed mixed-replay functional gate on the policy head."""

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self._gate_stats = defaultdict(self._empty_gate_stats)
        self._gate_stage = None

    @staticmethod
    def _empty_gate_stats():
        return {
            "total_batches": 0,
            "paired_batches": 0,
            "conflict_batches": 0,
            "cos_sum": 0.0,
            "conflict_cos_sum": 0.0,
            "new_norm_sum": 0.0,
            "replay_norm_sum": 0.0,
            "projection_cost_sum": 0.0,
            "current_fraction_sum": 0.0,
            "replay_fraction_sum": 0.0,
            "zero_anchor_count": 0,
        }

    def _record_gate_stats(self, epoch, current_fraction, replay_fraction, metrics=None):
        stats = self._gate_stats[int(epoch)]
        stats["total_batches"] += 1
        stats["current_fraction_sum"] += float(current_fraction)
        stats["replay_fraction_sum"] += float(replay_fraction)
        if metrics is None:
            return
        stats["paired_batches"] += 1
        cosine = metrics["cosine"].detach().item()
        if math.isfinite(cosine):
            stats["cos_sum"] += cosine
        if metrics["conflict"]:
            stats["conflict_batches"] += 1
            if math.isfinite(cosine):
                stats["conflict_cos_sum"] += cosine
        stats["new_norm_sum"] += math.sqrt(metrics["new_norm2"].detach().item())
        stats["replay_norm_sum"] += math.sqrt(metrics["replay_norm2"].detach().item())
        stats["projection_cost_sum"] += metrics["projection_cost"].detach().item()
        stats["zero_anchor_count"] += int(metrics["zero_anchor"])

    def _flush_gate_stats(self):
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        path = os.path.join(self.experiment_dir, "static_igfa_gate_stats.csv")
        exists = os.path.isfile(path)
        with open(path, "a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=GATE_STATS_FIELDS)
            if not exists:
                writer.writeheader()
            for epoch in sorted(self._gate_stats):
                item = self._gate_stats[epoch]
                total = item["total_batches"]
                paired = item["paired_batches"]
                conflicts = item["conflict_batches"]
                writer.writerow(
                    {
                        "seed": int(self.cfg.seed),
                        "stage": int(self._gate_stage),
                        "epoch": epoch,
                        "total_batches": total,
                        "paired_batches": paired,
                        "conflict_batches": conflicts,
                        "gate_fire_rate": "" if not paired else conflicts / paired,
                        "mean_cos": "" if not paired else item["cos_sum"] / paired,
                        "mean_conflict_cos": "" if not conflicts else item["conflict_cos_sum"] / conflicts,
                        "mean_new_head_grad_norm": "" if not paired else item["new_norm_sum"] / paired,
                        "mean_replay_head_grad_norm": "" if not paired else item["replay_norm_sum"] / paired,
                        "mean_projection_cost": "" if not paired else item["projection_cost_sum"] / paired,
                        "mean_current_fraction": item["current_fraction_sum"] / total,
                        "mean_replay_fraction": item["replay_fraction_sum"] / total,
                        "zero_anchor_count": item["zero_anchor_count"],
                    }
                )
        print(f"[static-igfa] stage={self._gate_stage} gate_stats={path}")

    def learn_all_tasks(self, datasets, benchmark, result_summary, current_task_id):
        self._gate_stage = int(current_task_id)
        self._gate_stats = defaultdict(self._empty_gate_stats)
        result = super().learn_all_tasks(datasets, benchmark, result_summary, current_task_id)
        self._flush_gate_stats()
        return result

    def observe(self, data, current_task_id, epoch, ifa_info=None):
        data[0] = self.map_tensor_to_device(data[0])
        self.optimizer.zero_grad()
        module = self.policy.module if isinstance(self.policy, DDP) else self.policy
        feature = data[0]["feature"].squeeze(1)
        actions = data[0]["actions"].squeeze(1)
        # Match the release MLR path on one GPU. The GPT-2 token embedding is
        # unused with inputs_embeds, so routing through DDP's reducer fails on
        # the following iteration even though world_size is one.
        distribution, _ = module(feature)
        loss_vector = module.policy_head.loss_fn(distribution, actions, reduction="none")
        loss_total = loss_vector.mean()

        task_ids = torch.tensor(
            [int(value) for value in data[1]], dtype=torch.long, device=loss_vector.device
        )
        current_mask = task_ids == int(current_task_id)
        replay_mask = ~current_mask
        n_current = int(current_mask.sum().item())
        n_replay = int(replay_mask.sum().item())
        batch_size = len(task_ids)
        current_fraction = n_current / batch_size
        replay_fraction = n_replay / batch_size

        if n_current == 0 or n_replay == 0:
            (self.loss_scale * loss_total).backward()
            if self.cfg.train.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
            self.optimizer.step()
            self._record_gate_stats(epoch, current_fraction, replay_fraction)
            return loss_total.item()

        head_params = [parameter for parameter in module.policy_head.parameters() if parameter.requires_grad]
        loss_new = loss_vector[current_mask].mean()
        loss_replay = loss_vector[replay_mask].mean()
        g_new = torch.autograd.grad(loss_new, head_params, retain_graph=True, create_graph=False)
        g_replay = torch.autograd.grad(loss_replay, head_params, retain_graph=True, create_graph=False)
        gate_strength = float(getattr(self.cfg, "robot_igfa_gate_strength", 1.0))
        gated_new, metrics = signed_replay_projection(
            g_new, g_replay, gate_strength=gate_strength
        )

        (self.loss_scale * loss_total).backward()
        combined_head = weighted_head_gradient(gated_new, g_replay, n_current, n_replay)
        replace_head_gradients(head_params, combined_head, self.loss_scale)
        if self.cfg.train.grad_clip is not None:
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        self._record_gate_stats(epoch, current_fraction, replay_fraction, metrics)
        return loss_total.item()
