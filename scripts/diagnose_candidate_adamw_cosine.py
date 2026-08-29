#!/usr/bin/env python3
import argparse
import csv
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from libero.lifelong.algos.mlr_temporal_robot_igfa import MLRTemporalRobotIGFA
from libero.lifelong.analysis.candidate_adamw_step import (
    candidate_instrumented_observe,
)
from libero.lifelong.analysis.finite_step_functional_probe import load_probe_manifest
from libero.lifelong.analysis.first_order_retention_validity import (
    load_checkpoint_into_policy,
)
from libero.lifelong.analysis.per_task_temporal_interference import (
    load_temporal_bin_map,
)
from libero.lifelong.utils import control_seed
from scripts.diagnose_first_order_retention_validity import checkpoint_path, run_dir
from scripts.diagnose_online_first_order_margin import (
    SCORE_PROBE_SEED,
    build_loader,
    create_optimizer_scheduler,
    write_csv,
)
from scripts.diagnose_per_task_temporal_interference import load_config


ANCHOR_FIELDS = [
    "seed", "stage", "epoch", "global_update", "lr", "aggregate_old_loss",
    "candidate_step_score", "candidate_step_cosine", "candidate_delta_norm",
    "actual_step_score", "actual_step_cosine", "actual_delta_norm",
    "candidate_actual_delta_cosine", "relative_l2_error",
    "max_abs_delta_error", "risk_cosine_abs_error", "risk_score_abs_error",
    "risk_sign_agreement",
]
METADATA_FIELDS = [
    "seed", "stage", "optimizer_class", "param_groups", "configured_lr",
    "betas", "eps", "weight_decay", "amsgrad", "maximize", "grad_clip",
]


def update_metadata(path, row):
    path = Path(path)
    existing = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    by_key = {(int(item["seed"]), int(item["stage"])): item for item in existing}
    by_key[(int(row["seed"]), int(row["stage"]))] = row
    write_csv(path, METADATA_FIELDS, [by_key[key] for key in sorted(by_key)])


def optimizer_metadata(optimizer, cfg, seed, stage):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise TypeError("formal candidate diagnosis requires AdamW")
    signatures = {
        (
            tuple(group["betas"]), float(group["eps"]),
            float(group["weight_decay"]), bool(group.get("amsgrad", False)),
            bool(group.get("maximize", False)),
        )
        for group in optimizer.param_groups
    }
    if len(signatures) != 1:
        raise ValueError("formal report expects one shared AdamW signature")
    betas, eps, weight_decay, amsgrad, maximize = next(iter(signatures))
    return {
        "seed": seed,
        "stage": stage,
        "optimizer_class": optimizer.__class__.__name__,
        "param_groups": len(optimizer.param_groups),
        "configured_lr": float(cfg.train.optimizer.kwargs.lr),
        "betas": f"{betas[0]},{betas[1]}",
        "eps": eps,
        "weight_decay": weight_decay,
        "amsgrad": int(amsgrad),
        "maximize": int(maximize),
        "grad_clip": cfg.train.grad_clip,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage", type=int, choices=(8, 9), required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--burnin-epochs", type=int, default=2)
    parser.add_argument("--anchor-interval", type=int, default=100)
    parser.add_argument("--label-probe-size", type=int, default=256)
    parser.add_argument("--score-probe-size", type=int, default=64)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/candidate_adamw_step_cosine/raw"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    control_seed(args.seed)
    run = run_dir(root, "V1", args.seed)
    cfg = load_config(run / "config.json", root, args.seed, args.device)
    cfg.temporal_bin_map = str(
        root / "metadata" / "temporal_bins" / f"object_seed{args.seed}.json"
    )
    cfg.experiment_dir = str(args.output_dir)
    cfg.ifa_enabled = False
    sidecar = load_temporal_bin_map(
        cfg.temporal_bin_map, expected_seed=args.seed, expected_bins=5
    )
    loader, replay_counts, current_count = build_loader(
        root, args.seed, args.stage, sidecar, args.batch_size, args.num_workers
    )
    algo = MLRTemporalRobotIGFA(n_tasks=10, cfg=cfg).to(args.device)
    entry = checkpoint_path(root, "V1", args.seed, args.stage - 1, 95)
    load_checkpoint_into_policy(algo.policy, entry)
    algo._gate_stage = int(args.stage)
    optimizer, scheduler, total_steps = create_optimizer_scheduler(
        algo, cfg, loader
    )
    update_metadata(
        args.output_dir.parent / "optimizer_metadata.csv",
        optimizer_metadata(optimizer, cfg, args.seed, args.stage),
    )

    manifest = load_probe_manifest(
        root / "reports" / "first_order_retention_validity" / "probe_manifest.csv",
        root,
    )
    label_probes = {}
    score_probes = {}
    for old_task in range(6, args.stage):
        paths = list(manifest[(args.seed, old_task)])
        if args.label_probe_size > len(paths):
            raise ValueError("label probe request exceeds committed manifest")
        label_probes[old_task] = paths[: args.label_probe_size]
        if args.score_probe_size > args.label_probe_size:
            raise ValueError("score probe request exceeds label probe")
        score_probes[old_task] = random.Random(
            SCORE_PROBE_SEED + args.seed * 100 + old_task
        ).sample(label_probes[old_task], args.score_probe_size)

    burnin_update = int(args.burnin_epochs) * len(loader)
    anchors = []
    global_update = 0
    print(
        f"[protocol] seed={args.seed} stage={args.stage} entry={entry} "
        f"epochs={args.epochs} loader_batches={len(loader)} burnin={burnin_update} "
        f"anchor_interval={args.anchor_interval} current={current_count} "
        f"replay={replay_counts} total_steps={total_steps}"
    )
    for epoch in range(args.epochs):
        loader.sampler.set_epoch(epoch)
        algo.policy.train()
        for data in loader:
            is_anchor = (
                global_update >= burnin_update
                and (global_update - burnin_update) % int(args.anchor_interval) == 0
            )
            if is_anchor:
                result = candidate_instrumented_observe(
                    algo,
                    data,
                    current_task_id=args.stage,
                    epoch=epoch,
                    label_probes=label_probes,
                    score_probes=score_probes,
                    label_batch_size=args.probe_batch_size,
                    score_batch_size=args.probe_batch_size,
                )
                row = {
                    "seed": args.seed,
                    "stage": args.stage,
                    "epoch": epoch,
                    "global_update": global_update,
                    "lr": optimizer.param_groups[0]["lr"],
                    **{field: result[field] for field in ANCHOR_FIELDS[5:]},
                }
                anchors.append(row)
                print(
                    f"[anchor] seed={args.seed} stage={args.stage} epoch={epoch} "
                    f"update={global_update} lr={optimizer.param_groups[0]['lr']:.8g} "
                    f"delta_cos={result['candidate_actual_delta_cosine']:.9f} "
                    f"rel_l2={result['relative_l2_error']:.6g} "
                    f"risk_err={result['risk_cosine_abs_error']:.6g}"
                )
            else:
                algo.observe(
                    data, current_task_id=args.stage, epoch=epoch, ifa_info=None
                )
            global_update += 1
        if scheduler is not None and epoch > 0:
            scheduler.step()
    anchor_path = args.output_dir / f"seed{args.seed}_stage{args.stage}_anchor.csv"
    write_csv(anchor_path, ANCHOR_FIELDS, anchors)
    print(
        f"[complete] seed={args.seed} stage={args.stage} updates={global_update} "
        f"anchors={len(anchors)} final_lr={optimizer.param_groups[0]['lr']:.8g}"
    )


if __name__ == "__main__":
    main()
