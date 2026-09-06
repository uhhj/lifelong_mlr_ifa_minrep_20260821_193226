#!/usr/bin/env python3
"""Run native vs shadow observe on real GMM policy/cache, without full training.

Use the server's original Python environment and an existing MLR config.json.
Compares parameters, gradients, optimizer/scheduler state and RNG after each step.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, type=Path)
    p.add_argument("--baseline-config", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--feature-root", type=Path)
    p.add_argument("--checkpoint", type=Path)
    args = p.parse_args()
    repo = args.repo.resolve()
    args.baseline_config = args.baseline_config.resolve()
    args.output = args.output.resolve()
    if args.feature_root:
        args.feature_root = args.feature_root.resolve()
    if args.checkpoint:
        args.checkpoint = args.checkpoint.resolve()
    sys.path.insert(0, str(repo))
    os.chdir(str(repo))
    import numpy as np
    import torch
    from easydict import EasyDict
    from torch.utils.data._utils.collate import default_collate
    from libero.lifelong.algos.mlr import MLR
    from libero.lifelong.algos.mlr_replay_constraint import MLRReplayConstraint
    cfg = EasyDict(json.loads(args.baseline_config.read_text(encoding="utf-8")))
    if bool(getattr(cfg, "ifa_enabled", True)):
        raise ValueError("Use an existing MLR (IFA disabled) configuration.")
    if args.checkpoint:
        cfg.pretrain = str(args.checkpoint.resolve())
    features = (args.feature_root or Path(cfg.feature_dir)).resolve()
    cfg.device = args.device
    cfg.ifa_enabled = False
    cfg.ifa_interaction_logging = False
    cfg.training = True
    cfg.lifelong = EasyDict(algo="MLRReplayConstraint", eval_in_train=False,
        projection_enabled=False, reference_norm_floor=1e-12, synchronize_timing=True, step_log_every=0)
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    def rng():
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    def set_rng(state):
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state_all(state[3])
    max_abs = [0.0]
    def check(a, b, path="root"):
        if torch.is_tensor(a):
            if not torch.is_tensor(b) or a.shape != b.shape:
                raise AssertionError("Structure mismatch: " + path)
            if a.numel():
                max_abs[0] = max(max_abs[0], float((a-b).abs().max()))
            if not torch.equal(a, b):
                raise AssertionError("Exact parity failed at " + path + "; max diff=" + str(max_abs[0]))
        elif isinstance(a, np.ndarray):
            if not np.array_equal(a, b):
                raise AssertionError("Numpy RNG differs")
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise AssertionError("State keys differ at " + path)
            for k in a:
                check(a[k], b[k], path + "." + str(k))
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise AssertionError("State length differs")
            for i, (x, y) in enumerate(zip(a, b)):
                check(x, y, path + "." + str(i))
        elif a != b:
            raise AssertionError("Value differs at " + path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rc_parity_", dir=str(args.output.parent)) as tmp:
        native_cfg, shadow_cfg = copy.deepcopy(cfg), copy.deepcopy(cfg)
        native_cfg.experiment_dir = str(Path(tmp) / "native")
        shadow_cfg.experiment_dir = str(Path(tmp) / "shadow")
        Path(native_cfg.experiment_dir).mkdir(); Path(shadow_cfg.experiment_dir).mkdir()
        native = MLR(1, native_cfg).to(args.device)
        shadow = MLRReplayConstraint(1, shadow_cfg).to(args.device)
        shadow.policy.load_state_dict(native.policy.state_dict(), strict=True)
        # Only loader length is used to configure scheduler. Use the historical
        # first-stage step count where available; both arms use exactly the same.
        count = len(list((features / "6").glob("*.pt")))
        loader = range((count + int(cfg.train.batch_size) - 1) // int(cfg.train.batch_size))
        native.start_task(-1, loader); shadow.start_task(-1, loader)
        native.policy.train(); shadow.policy.train()
        for case, ids in enumerate(([7]*10, [6]*5+[7]*5, [6]*10)):
            files = {t: sorted((features / str(t)).glob("*.pt")) for t in set(ids)}
            if any(not v for v in files.values()):
                raise ValueError("Need actual task6 and task7 cached features.")
            batch = default_collate([(torch.load(str(files[t][i % len(files[t])]), map_location="cpu"), str(t)) for i, t in enumerate(ids)])
            state = rng()
            loss_native = native.observe(copy.deepcopy(batch), 7, case, ifa_info=None)
            after = rng()
            set_rng(state)
            loss_shadow = shadow.observe(copy.deepcopy(batch), 7, case, ifa_info=None)
            check(after, rng(), "rng")
            check(loss_native, loss_shadow, "loss")
            check(native.policy.state_dict(), shadow.policy.state_dict(), "parameters")
            check(native.optimizer.state_dict(), shadow.optimizer.state_dict(), "optimizer")
            for a, b in zip(native.policy.parameters(), shadow.policy.parameters()):
                check(a.grad, b.grad, "gradient")
            if native.scheduler is not None:
                check(native.scheduler.state_dict(), shadow.scheduler.state_dict(), "scheduler")
        shadow._flush_epoch()
    args.output.write_text(json.dumps({"status": "PASS", "cases": ["new_only", "mixed", "old_only"],
        "comparison": "native observe vs projection-disabled subclass, exact equality",
        "max_abs_difference": max_abs[0], "torch": torch.__version__, "device": args.device,
        "baseline_config": str(args.baseline_config),
        "limitations": "Direct policy API, no DDP wrapper; real torchrun smoke is a separate check."}, indent=2) + "\n", encoding="utf-8")
    print("PASS: exact real-policy parity. " + str(args.output))


if __name__ == "__main__":
    main()
