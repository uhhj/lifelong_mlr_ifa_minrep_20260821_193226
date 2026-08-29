#!/usr/bin/env python3
import argparse
import csv
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from libero.lifelong.algos.mlr import FileTrajectoryDataset
from libero.lifelong.algos.mlr_temporal_robot_igfa import MLRTemporalRobotIGFA
from libero.lifelong.analysis.finite_step_functional_probe import load_probe_manifest
from libero.lifelong.analysis.first_order_retention_validity import (
    load_checkpoint_into_policy,
)
from libero.lifelong.analysis.online_first_order_margin import instrumented_observe
from libero.lifelong.analysis.per_task_temporal_interference import (
    load_temporal_bin_map,
    reconstruct_stage_replay_paths,
)
from libero.lifelong.utils import control_seed, make_linear_schedule_with_warmup
from scripts.diagnose_first_order_retention_validity import checkpoint_path, run_dir
from scripts.diagnose_per_task_temporal_interference import load_config


SCORE_PROBE_SEED = 20260831
ANCHOR_FIELDS = [
    "seed", "stage", "epoch", "global_update", "lr", "old_task_count",
    "aggregate_old_loss", "native_candidate_bins", "native_raw_max",
    "native_raw_mean", "native_cos_max", "native_cos_mean",
    "native_v1_would_gate", "actual_step_score", "actual_step_cosine",
    "actual_step_relative", "actual_step_delta_norm", "actual_step_old_grad_norm",
]
TASK_FIELDS = [
    "seed", "stage", "epoch", "global_update", "old_task", "old_label_loss",
    "old_score_probe_loss", "actual_step_task_score", "actual_step_task_cosine",
    "actual_step_task_relative",
]


def numeric_path_key(path):
    path = Path(path)
    try:
        return int(path.stem)
    except ValueError:
        return path.name


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def update_score_manifest(path, seed, score_probes):
    path = Path(path)
    existing = []
    if path.is_file():
        existing = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    by_key = {
        (int(row["seed"]), int(row["old_task"]), int(row["record_id"])): row
        for row in existing
    }
    for old_task, paths in score_probes.items():
        for record in paths:
            record_id = int(Path(record).stem)
            by_key[(int(seed), int(old_task), record_id)] = {
                "seed": int(seed),
                "old_task": int(old_task),
                "record_id": record_id,
                "path": str(record),
            }
    rows = [by_key[key] for key in sorted(by_key)]
    write_csv(path, ["seed", "old_task", "record_id", "path"], rows)


def build_loader(root, seed, stage, sidecar, batch_size, num_workers):
    feature_root = root / "features" / f"object_seed{seed}"
    current = sorted(feature_root.joinpath(str(stage)).glob("*.pt"), key=numeric_path_key)
    replay = reconstruct_stage_replay_paths(
        feature_root, seed, range(6, int(stage)), rate=0.5
    )
    mix_paths = [str(path) for path in current]
    for old_task in sorted(replay):
        mix_paths.extend(str(path) for path in replay[old_task])
    random.Random(int(seed)).shuffle(mix_paths)
    dataset = FileTrajectoryDataset(mix_paths, temporal_bin_map=sidecar)
    sampler = DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=True, seed=0
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        sampler=sampler,
        persistent_workers=False,
    )
    return loader, {task: len(paths) for task, paths in replay.items()}, len(current)


def create_optimizer_scheduler(algo, cfg, loader):
    optimizer = eval(str(cfg.train.optimizer.name), {"torch": torch})(
        algo.policy.parameters(), **cfg.train.optimizer.kwargs
    )
    sched_cfg = cfg.train.scheduler.kwargs
    warmup_steps = sched_cfg.get("warmup_steps", 500)
    total_steps = 100 * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_linear_schedule_with_warmup(warmup_steps, total_steps),
        last_epoch=sched_cfg.get("last_epoch", -1),
    )
    algo.optimizer = optimizer
    algo.scheduler = scheduler
    return optimizer, scheduler, total_steps


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
        default=Path("reports/online_first_order_margin_validity/raw"),
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
    optimizer, scheduler, total_steps = create_optimizer_scheduler(algo, cfg, loader)

    all_probes = load_probe_manifest(
        root / "reports" / "first_order_retention_validity" / "probe_manifest.csv",
        root,
    )
    label_probes = {}
    score_probes = {}
    for old_task in range(6, args.stage):
        label_paths = list(all_probes[(args.seed, old_task)])
        if args.label_probe_size > len(label_paths):
            raise ValueError("label probe request exceeds committed manifest")
        label_paths = label_paths[: args.label_probe_size]
        label_probes[old_task] = label_paths
        if args.score_probe_size > len(label_paths):
            raise ValueError("score probe request exceeds label probe")
        score_probes[old_task] = random.Random(
            SCORE_PROBE_SEED + args.seed * 100 + old_task
        ).sample(label_paths, args.score_probe_size)
    update_score_manifest(
        args.output_dir.parent / "score_probe_manifest.csv", args.seed, score_probes
    )

    burnin_update = int(args.burnin_epochs) * len(loader)
    anchor_rows = []
    task_rows = []
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
                result = instrumented_observe(
                    algo,
                    data,
                    current_task_id=args.stage,
                    epoch=epoch,
                    label_probes=label_probes,
                    score_probes=score_probes,
                    label_batch_size=args.probe_batch_size,
                    score_batch_size=args.probe_batch_size,
                )
                anchor_rows.append(
                    {
                        "seed": args.seed,
                        "stage": args.stage,
                        "epoch": epoch,
                        "global_update": global_update,
                        "lr": optimizer.param_groups[0]["lr"],
                        "old_task_count": len(label_probes),
                        **{key: result[key] for key in ANCHOR_FIELDS[6:]},
                    }
                )
                for old_task, task_result in sorted(result["task_rows"].items()):
                    task_rows.append(
                        {
                            "seed": args.seed,
                            "stage": args.stage,
                            "epoch": epoch,
                            "global_update": global_update,
                            "old_task": old_task,
                            **task_result,
                        }
                    )
                print(
                    f"[anchor] seed={args.seed} stage={args.stage} epoch={epoch} "
                    f"update={global_update} lr={optimizer.param_groups[0]['lr']:.8g} "
                    f"candidates={result['native_candidate_bins']} "
                    f"raw_max={result['native_raw_max']} step={result['actual_step_score']:.6g}"
                )
            else:
                algo.observe(data, current_task_id=args.stage, epoch=epoch, ifa_info=None)
            global_update += 1
        if scheduler is not None and epoch > 0:
            scheduler.step()
    anchor_path = args.output_dir / f"seed{args.seed}_stage{args.stage}_anchor.csv"
    task_path = args.output_dir / f"seed{args.seed}_stage{args.stage}_task.csv"
    write_csv(anchor_path, ANCHOR_FIELDS, anchor_rows)
    write_csv(task_path, TASK_FIELDS, task_rows)
    print(
        f"[complete] seed={args.seed} stage={args.stage} updates={global_update} "
        f"anchors={len(anchor_rows)} task_rows={len(task_rows)} "
        f"final_lr={optimizer.param_groups[0]['lr']:.8g}"
    )


if __name__ == "__main__":
    main()
