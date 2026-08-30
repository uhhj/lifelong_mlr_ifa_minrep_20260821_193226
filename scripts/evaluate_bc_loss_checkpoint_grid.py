#!/usr/bin/env python3
import argparse
import csv
import gc
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from omegaconf import OmegaConf

from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.utils import control_seed, torch_load_model


RUN_SUFFIX = {
    "temporal_v1": "mlr_temporal_igfa",
    "zero_guard": "mlr_zero_directional_guard",
}
FIELDS = [
    "method",
    "seed",
    "stage",
    "epoch",
    "task",
    "is_old_task",
    "is_acquisition_stage",
    "probe_size",
    "mean_bc_loss",
]


def load_config(path, project_root, seed, device):
    with Path(path).open(encoding="utf-8") as handle:
        cfg = OmegaConf.create(json.load(handle))
    cfg.device = str(device)
    cfg.training = True
    cfg.aug = False
    cfg.pretrain = str(
        Path(project_root) / "checkpoints" / "object_multitask_model_ep99.pth"
    )
    cfg.feature_dir = str(Path(project_root) / "features" / f"object_seed{seed}")
    return cfg


def load_probe_manifest(path, project_root):
    """Load the fixed 256-record old-task probes without importing training analyses."""
    path = Path(path).resolve()
    root = Path(project_root).resolve()
    probes = defaultdict(list)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    if len(rows) != 2304:
        raise ValueError(f"expected 2304 probe records, found {len(rows)}")
    for row in rows:
        seed = int(row["seed"])
        old_task = int(row["old_task"])
        record_id = int(row["record_id"])
        recorded = Path(row["path"])
        resolved = (
            recorded
            if recorded.is_file()
            else root
            / "features"
            / f"object_seed{seed}"
            / str(old_task)
            / f"{record_id}.pt"
        )
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        probes[(seed, old_task)].append(resolved)
    if len(probes) != 9:
        raise ValueError(f"expected 9 seed/task probe groups, found {len(probes)}")
    for key, records in probes.items():
        if len(records) != 256:
            raise ValueError(f"probe group {key} has {len(records)} records, expected 256")
    return dict(probes)


def load_checkpoint_into_policy(policy, checkpoint):
    state_dict = torch_load_model(str(checkpoint), map_location="cpu")[0]
    incompatible = policy.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint incompatibility for {checkpoint}: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    policy.eval()


def load_feature_paths(feature_paths, device):
    features = []
    actions = []
    for path in feature_paths:
        saved = torch.load(str(path), map_location="cpu")
        feature = saved["feature"]
        action = saved["actions"]
        while feature.ndim > 3 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        while action.ndim > 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        if tuple(feature.shape) != (8, 4, 512) or tuple(action.shape) != (8, 7):
            raise ValueError(f"unexpected latent record shape: {path}")
        features.append(feature)
        actions.append(action)
    return torch.stack(features).to(device), torch.stack(actions).to(device)


@torch.no_grad()
def evaluate_mean_loss(policy, feature_paths, batch_size=32):
    policy.eval()
    device = next(policy.parameters()).device
    loss_sum = 0.0
    sample_count = 0
    paths = [Path(path) for path in feature_paths]
    for start in range(0, len(paths), int(batch_size)):
        chunk = paths[start : start + int(batch_size)]
        features, actions = load_feature_paths(chunk, device)
        temporal = policy.temporal_encode(features)
        distribution = policy.policy_head(temporal[:, -1, :], train=True)
        losses = policy.policy_head.loss_fn(distribution, actions, reduction="none")
        if not torch.isfinite(losses).all():
            raise FloatingPointError("non-finite old-task probe loss")
        loss_sum += float(losses.detach().double().sum().cpu())
        sample_count += len(chunk)
    if not sample_count:
        raise RuntimeError("empty old-task probe")
    return loss_sum / sample_count


def checkpoint_path(root, method, seed, stage, epoch):
    run = (
        root
        / "runs"
        / f"object_seed{seed}_{RUN_SUFFIX[method]}"
        / "run_001"
    )
    path = run / f"mlrifa_{stage}_model_ep{epoch}.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--probe-manifest",
        type=Path,
        default=Path("reports/first_order_retention_validity/probe_manifest.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/bc_rollout_retention_alignment/bc_loss_checkpoint_grid.csv"
        ),
    )
    parser.add_argument(
        "--methods", nargs="+", choices=sorted(RUN_SUFFIX), default=list(RUN_SUFFIX)
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--stages", nargs="+", type=int, default=[6, 7, 8, 9])
    parser.add_argument("--epochs", nargs="+", type=int, default=[70, 75, 80, 85, 90, 95])
    parser.add_argument("--tasks", nargs="+", type=int, default=[6, 7, 8])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    root = args.root.resolve()
    manifest = (
        args.probe_manifest
        if args.probe_manifest.is_absolute()
        else root / args.probe_manifest
    )
    probes = load_probe_manifest(manifest, root)
    rows = []
    for method in args.methods:
        for seed in args.seeds:
            control_seed(seed)
            run = root / "runs" / f"object_seed{seed}_{RUN_SUFFIX[method]}" / "run_001"
            cfg = load_config(run / "config.json", root, seed, args.device)
            policy = BriefPolicy(cfg, None).to(args.device)
            for stage in args.stages:
                tasks = [task for task in args.tasks if task <= stage]
                for epoch in args.epochs:
                    checkpoint = checkpoint_path(root, method, seed, stage, epoch)
                    load_checkpoint_into_policy(policy, checkpoint)
                    for task in tasks:
                        paths = probes[(seed, task)]
                        loss = evaluate_mean_loss(
                            policy, paths, batch_size=args.batch_size
                        )
                        if not math.isfinite(loss):
                            raise ValueError(
                                f"non-finite BC loss method={method} seed={seed} "
                                f"stage={stage} epoch={epoch} task={task}"
                            )
                        rows.append(
                            {
                                "method": method,
                                "seed": seed,
                                "stage": stage,
                                "epoch": epoch,
                                "task": task,
                                "is_old_task": int(task < stage),
                                "is_acquisition_stage": int(task == stage),
                                "probe_size": len(paths),
                                "mean_bc_loss": loss,
                            }
                        )
                    print(
                        f"[bc-grid] method={method} seed={seed} stage={stage} "
                        f"epoch={epoch} tasks={tasks}",
                        flush=True,
                    )
            del policy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    expected = len(args.methods) * len(args.seeds) * len(args.epochs) * sum(
        sum(task <= stage for task in args.tasks) for stage in args.stages
    )
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} BC rows, produced {len(rows)}")
    if any(int(row["probe_size"]) != 256 for row in rows):
        raise RuntimeError("formal BC grid did not use 256 probes per task")
    write_rows(args.output, rows)
    print(f"[bc-grid] rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
