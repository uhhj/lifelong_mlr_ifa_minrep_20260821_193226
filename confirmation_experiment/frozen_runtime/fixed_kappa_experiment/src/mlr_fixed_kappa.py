"""Drop-in subclass of the local MLR; original data/stage loop is inherited."""
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from libero.lifelong.algos.mlr import MLR
from libero.lifelong.algos.fixed_kappa import (
    backward_mixed_with_reference, mixed_losses, validate_kappa,
)


class MLRFixedKappa(MLR):
    def __init__(self, n_tasks, cfg, **kwargs):
        if getattr(cfg, "ifa_enabled", True):
            raise ValueError("This experiment requires ifa_enabled=false.")
        if getattr(cfg, "ifa_interaction_logging", False):
            raise ValueError("IFA-specific logging is incompatible with this arm.")
        if dist.is_initialized() and dist.get_world_size() != 1:
            raise ValueError("Only one process/GPU is supported: no silent local-only DDP projection.")
        self._fixed_kappa = validate_kappa(cfg.lifelong.kappa)
        super().__init__(n_tasks=n_tasks, cfg=cfg, **kwargs)
        self._rc_epoch_key = None
        self._rc_rows = []
        self._rc_step = 0
        self._rc_params = tuple(p for p in self.policy.parameters() if p.requires_grad)
        names = [n for n, p in self.policy.named_parameters() if p.requires_grad]
        if not names or any(not n.startswith(("temporal_transformer.", "policy_head.")) for n in names):
            raise ValueError("Unexpected trainable parameters; expected BriefPolicy decoder and head only.")
        self._rc_parameter_names = names
        self._rc_stage_peaks = {}
        path = Path(self.experiment_dir) / "fixed_kappa_parameters.json"
        path.write_text(json.dumps({
            "names": names,
            "trainable_parameters": sum(p.numel() for p in self._rc_params),
            "enabled": bool(cfg.lifelong.projection_enabled),
            "kappa": self._fixed_kappa,
            "reference": "old samples in the same original mixed batch",
            "projection_target": "full mixed BC gradient before global clipping",
        }, indent=2) + "\n", encoding="utf-8")

    def _flush_epoch(self):
        if not self._rc_rows:
            return
        rows = self._rc_rows
        stage, epoch = self._rc_epoch_key
        paired = [r for r in rows if r["n_new"] and r["n_old"]]
        refs = [r for r in rows if r["has_reference"]]
        valid = [r for r in refs if not r["tiny_reference"]]
        projected = [r for r in rows if r["projected"]]
        def avg(key, subset=rows):
            vals = [r[key] for r in subset if r.get(key) is not None]
            return sum(vals) / len(vals) if vals else None
        def weighted_loss(key, count):
            total = sum(r[count] for r in rows)
            return sum(r[key] * r[count] for r in rows if r[count]) / total if total else None
        row = {
            "seed": int(self.cfg.seed), "stage": stage, "epoch": epoch,
            "enabled": bool(self.cfg.lifelong.projection_enabled),
            "kappa": self._fixed_kappa,
            "batches": len(rows), "paired_batches": len(paired),
            "reference_batches": len(refs), "valid_reference_batches": len(valid),
            "no_reference_batches": len(rows) - len(refs),
            "tiny_reference_batches": sum(r["tiny_reference"] for r in rows),
            "conflict_batches": sum(r["conflict"] for r in rows),
            "projected_batches": len(projected),
            "projection_rate_all": len(projected) / len(rows),
            "projection_rate_valid_reference": len(projected) / len(valid) if valid else None,
            "conflict_rate_valid_reference": sum(r["conflict"] for r in valid) / len(valid) if valid else None,
            "mean_cosine": avg("cosine", valid),
            "mean_correction_ratio": avg("correction_ratio"),
            "mean_projected_correction_ratio": avg("correction_ratio", projected),
            "mean_dot_before": avg("dot_before", valid),
            "mean_dot_after": avg("dot_after", valid),
            "mean_mixed_grad_norm": avg("mixed_grad_norm"),
            "mean_reference_grad_norm": avg("reference_grad_norm", refs),
            "mean_preclip_grad_norm": avg("preclip_grad_norm"),
            "new_samples": sum(r["n_new"] for r in rows),
            "old_samples": sum(r["n_old"] for r in rows),
            "mean_bc_loss": avg("bc_loss"),
            "mean_new_loss": weighted_loss("new_loss", "n_new"),
            "mean_old_loss": weighted_loss("old_loss", "n_old"),
            "observe_seconds": sum(r["seconds"] for r in rows),
            "lr_first": rows[0]["lr"], "lr_last": rows[-1]["lr"],
            "stage_peak_allocated_bytes": self._rc_stage_peaks.get(stage, 0),
        }
        path = Path(self.experiment_dir) / "fixed_kappa_epochs.csv"
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        self._rc_rows = []

    def start_task(self, task, train_loader):
        super().start_task(task, train_loader)
        # The outer entry point moves/wraps the policy after __init__. Refresh
        # these references after device placement, before gradient collection.
        self._rc_params = tuple(p for p in self.policy.parameters() if p.requires_grad)

    def learn_all_tasks(self, datasets, benchmark, result_summary, current_task_id):
        device = next(self.policy.parameters()).device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        try:
            return super().learn_all_tasks(datasets, benchmark, result_summary, current_task_id)
        finally:
            self._flush_epoch()

    def observe(self, data, current_task_id, epoch, ifa_info=None):
        if validate_kappa(self.cfg.lifelong.kappa) != self._fixed_kappa:
            raise ValueError("kappa must remain fixed throughout the run")
        key = (int(current_task_id), int(epoch))
        if key != self._rc_epoch_key:
            self._flush_epoch()
            self._rc_epoch_key = key
        device = self._rc_params[0].device
        sync = bool(self.cfg.lifelong.synchronize_timing) and device.type == "cuda"
        if sync:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        data[0] = self.map_tensor_to_device(data[0])
        self.optimizer.zero_grad()  # Preserve the target baseline's zero_grad semantics.
        policy = self.policy.module if isinstance(self.policy, DDP) else self.policy
        per_sample, _ = policy.compute_loss(data[0], None, None, reduction="none")
        ids = [int(x) for x in data[1]]
        loss, new_loss, old_loss, n_new, n_old = mixed_losses(per_sample, ids, current_task_id)
        print("loss:{}".format(loss))  # Keep the native baseline's per-step loss logging.
        if not bool(torch.isfinite(per_sample).all()):
            raise FloatingPointError("Nonfinite per-sample loss; update aborted.")
        metrics = backward_mixed_with_reference(
            loss, old_loss, self._rc_params, loss_scale=self.loss_scale,
            enabled=bool(self.cfg.lifelong.projection_enabled),
            reference_norm_floor=float(self.cfg.lifelong.reference_norm_floor),
            kappa=self._fixed_kappa)
        preclip_norm = None
        if self.cfg.train.grad_clip is not None:
            preclip_norm = float(torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip, error_if_nonfinite=True))
        lr = float(self.optimizer.param_groups[0]["lr"])
        if hasattr(self, "optimizer_step_and_maybe_schedule"):
            self.optimizer_step_and_maybe_schedule()
        else:
            self.optimizer.step()  # Older MLR versions step their scheduler in the inherited epoch loop.
        if sync:
            torch.cuda.synchronize(device)
        metrics.update(
            n_new=n_new, n_old=n_old, bc_loss=float(loss.detach()),
            new_loss=float(new_loss.detach()) if new_loss is not None else None,
            old_loss=float(old_loss.detach()) if old_loss is not None else None,
            preclip_grad_norm=preclip_norm, lr=lr,
            seconds=time.perf_counter() - started)
        if not math.isfinite(metrics["bc_loss"]):
            raise FloatingPointError("Nonfinite loss.")
        if device.type == "cuda":
            self._rc_stage_peaks[int(current_task_id)] = int(torch.cuda.max_memory_allocated(device))
        self._rc_rows.append(metrics)
        self._rc_step += 1
        every = int(self.cfg.lifelong.step_log_every)
        if every > 0 and self._rc_step % every == 0:
            with (Path(self.experiment_dir) / "fixed_kappa_steps.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(dict(metrics, stage=key[0], epoch=key[1], step=self._rc_step), allow_nan=False) + "\n")
        return metrics["bc_loss"]
