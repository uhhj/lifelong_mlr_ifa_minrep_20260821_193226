#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from easydict import EasyDict
from torch.utils.data import DataLoader, DistributedSampler

from libero.lifelong.algos.mlr import MLR, build_dataset_for_task
from libero.lifelong.utils import control_seed, safe_device


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "runs" / "paper_v2_smoke_legacy_epoch" / "run_001" / "config.json"
OUT = ROOT / "reports" / "paper_reproduction_v2" / "legacy_parity.json"


def state_equal(left, right):
    if left.keys() != right.keys():
        return False
    if left["param_groups"] != right["param_groups"]:
        return False
    for key in left["state"]:
        for name, value in left["state"][key].items():
            other = right["state"][key][name]
            if torch.is_tensor(value):
                if not torch.equal(value, other):
                    return False
            elif value != other:
                return False
    return True


def main():
    cfg = EasyDict(json.loads(CONFIG.read_text(encoding="utf-8")))
    cfg.repro_scheduler_mode = "legacy_epoch"
    cfg.ifa_enabled = False
    cfg.device = "cuda"
    dataset = build_dataset_for_task(cfg.feature_dir, 6, cfg.buffer_dir, 42, start=6)
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
    sampler.set_epoch(0)
    loader = DataLoader(dataset, batch_size=10, sampler=sampler, num_workers=0)

    control_seed(42)
    release = safe_device(MLR(1, cfg), "cuda")
    control_seed(42)
    instrumented = safe_device(MLR(1, cfg), "cuda")
    release.train(); instrumented.train()
    release.start_task(-1, loader); instrumented.start_task(-1, loader)
    release.optimizer_step_and_maybe_schedule = release.optimizer.step

    old_losses, new_losses = [], []
    for index, batch in enumerate(loader):
        if index == 20:
            break
        cpu_state = torch.get_rng_state()
        cuda_state = torch.cuda.get_rng_state()
        old_losses.append(release.observe(batch, 6, 0, ifa_info=None))
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)
        new_losses.append(instrumented.observe(batch, 6, 0, ifa_info=None))

    old_params = torch.cat([p.detach().flatten() for p in release.policy.parameters()])
    new_params = torch.cat([p.detach().flatten() for p in instrumented.policy.parameters()])
    denom = torch.linalg.vector_norm(old_params).clamp_min(1e-30)
    payload = {
        "starting_sha": "7b937ca247b153f5ddd16385746c7709a1a2d376",
        "seed": 42,
        "task": 6,
        "updates": 20,
        "loss_max_abs": max(abs(a - b) for a, b in zip(old_losses, new_losses)),
        "parameter_relative_l2": float(torch.linalg.vector_norm(old_params - new_params) / denom),
        "optimizer_state_equal": state_equal(release.optimizer.state_dict(), instrumented.optimizer.state_dict()),
        "lr_exact_equal": release.optimizer.param_groups[0]["lr"] == instrumented.optimizer.param_groups[0]["lr"],
    }
    payload["pass"] = (
        payload["loss_max_abs"] <= 1e-6
        and payload["parameter_relative_l2"] <= 1e-6
        and payload["optimizer_state_equal"]
        and payload["lr_exact_equal"]
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not payload["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
