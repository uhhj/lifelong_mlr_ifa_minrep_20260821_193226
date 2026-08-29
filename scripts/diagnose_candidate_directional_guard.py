#!/usr/bin/env python3
import argparse
import copy
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from libero.lifelong.algos.mlr_temporal_robot_igfa import MLRTemporalRobotIGFA
from libero.lifelong.analysis.candidate_directional_guard import (
    capture_branch_state,
    directional_guard_step,
    evaluate_guard_probes,
    resolve_guard_threshold,
    restore_branch_state,
    select_current_probe,
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
    numeric_path_key,
    write_csv,
)
from scripts.diagnose_per_task_temporal_interference import load_config


STEP_FIELDS = [
    "seed", "stage", "arm", "epoch", "update", "threshold_mode", "threshold", "guard_scale",
    "training_loss", "candidate_step_score", "candidate_step_cosine",
    "candidate_delta_norm", "old_grad_norm", "would_fire", "guard_fire",
    "applied_scale", "predicted_applied_score",
    "predicted_applied_delta_norm", "lr", "arm_wall_seconds",
]
PROBE_FIELDS = [
    "seed", "stage", "arm", "update", "old_mean_loss", "current_loss",
    "threshold_mode", "threshold", "guard_scale",
]
OLD_TASK_FIELDS = [
    "seed", "stage", "arm", "update", "old_task", "loss", "threshold_mode"
]


def reset_logging_state(algo, stage):
    algo._gate_stage = int(stage)
    algo._bin_stats = defaultdict(algo._empty_bin_stats)
    algo._overall_stats = defaultdict(algo._empty_overall_stats)


def build_probes(root, seed, stage, label_size, score_size, current_size):
    manifest = load_probe_manifest(
        root / "reports" / "first_order_retention_validity" / "probe_manifest.csv",
        root,
    )
    label_probes = {}
    score_probes = {}
    for old_task in range(6, int(stage)):
        paths = list(manifest[(int(seed), old_task)])
        if int(label_size) > len(paths):
            raise ValueError("label probe request exceeds committed manifest")
        label_probes[old_task] = paths[: int(label_size)]
        if int(score_size) > int(label_size):
            raise ValueError("score probe request exceeds label probe")
        score_probes[old_task] = random.Random(
            SCORE_PROBE_SEED + int(seed) * 100 + old_task
        ).sample(label_probes[old_task], int(score_size))
    current_paths = sorted(
        root.joinpath("features", f"object_seed{seed}", str(stage)).glob("*.pt"),
        key=numeric_path_key,
    )
    current_probe = select_current_probe(
        current_paths, seed=seed, stage=stage, size=current_size
    )
    return label_probes, score_probes, current_probe


def record_probe(seed, stage, arm, update, threshold_mode, threshold, scale, result):
    aggregate = {
        "seed": seed,
        "stage": stage,
        "arm": arm,
        "update": update,
        "old_mean_loss": result["old_mean_loss"],
        "current_loss": result["current_loss"],
        "threshold_mode": threshold_mode,
        "threshold": threshold,
        "guard_scale": scale,
    }
    per_task = [
        {
            "seed": seed,
            "stage": stage,
            "arm": arm,
            "update": update,
            "old_task": task,
            "loss": loss,
            "threshold_mode": threshold_mode,
        }
        for task, loss in sorted(result["old_task_losses"].items())
    ]
    return aggregate, per_task


def run_arm(
    algo,
    scheduler,
    branch_state,
    loader,
    seed,
    stage,
    epoch,
    updates,
    threshold,
    threshold_mode,
    scale,
    enabled,
    old_label_probes,
    score_probes,
    current_probe,
    probe_batch_size,
):
    arm = "guard" if enabled else "control"
    restore_branch_state(algo, scheduler, branch_state)
    reset_logging_state(algo, stage)
    loader.sampler.set_epoch(int(epoch))
    iterator = iter(loader)
    algo.policy.train()
    probe_updates = sorted({0, min(100, int(updates)), int(updates)})
    step_rows = []
    probe_rows = []
    task_rows = []
    initial = evaluate_guard_probes(
        algo.policy, old_label_probes, current_probe, batch_size=probe_batch_size
    )
    row, old_rows = record_probe(
        seed, stage, arm, 0, threshold_mode, threshold, scale, initial
    )
    probe_rows.append(row)
    task_rows.extend(old_rows)
    started = time.monotonic()
    for update in range(1, int(updates) + 1):
        try:
            data = next(iterator)
        except StopIteration as error:
            raise RuntimeError(
                f"loader exhausted before {updates} intervention updates"
            ) from error
        training_loss, metrics = directional_guard_step(
            algo,
            data,
            stage=stage,
            epoch=epoch,
            score_probes=score_probes,
            threshold=threshold,
            scale=scale,
            enabled=enabled,
            score_batch_size=probe_batch_size,
        )
        values = [float(training_loss)] + [
            float(metrics[key])
            for key in (
                "candidate_step_score", "candidate_step_cosine",
                "candidate_delta_norm", "old_grad_norm",
                "predicted_applied_score", "predicted_applied_delta_norm",
            )
        ]
        if not all(torch.isfinite(torch.tensor(values))):
            raise FloatingPointError("non-finite directional guard step metric")
        step_rows.append(
            {
                "seed": seed,
                "stage": stage,
                "arm": arm,
                "epoch": epoch,
                "update": update,
                "threshold_mode": threshold_mode,
                "threshold": threshold,
                "guard_scale": scale,
                "training_loss": training_loss,
                **metrics,
                "lr": algo.optimizer.param_groups[0]["lr"],
                "arm_wall_seconds": "",
            }
        )
        if update in probe_updates:
            result = evaluate_guard_probes(
                algo.policy,
                old_label_probes,
                current_probe,
                batch_size=probe_batch_size,
            )
            row, old_rows = record_probe(
                seed, stage, arm, update, threshold_mode, threshold, scale, result
            )
            probe_rows.append(row)
            task_rows.extend(old_rows)
        if update % 25 == 0 or update == int(updates):
            fires = sum(int(item["guard_fire"]) for item in step_rows)
            would = sum(int(item["would_fire"]) for item in step_rows)
            print(
                f"[arm-progress] seed={seed} stage={stage} arm={arm} "
                f"update={update}/{updates} fires={fires} would_fire={would}",
                flush=True,
            )
    elapsed = time.monotonic() - started
    for row in step_rows:
        row["arm_wall_seconds"] = elapsed
    return step_rows, probe_rows, task_rows


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--stage", type=int, choices=(8, 9), required=True)
    parser.add_argument(
        "--threshold-mode", choices=("loso", "zero"), default="loso"
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--intervention-updates", type=int, default=300)
    parser.add_argument("--guard-scale", type=float, default=0.5)
    parser.add_argument("--label-probe-size", type=int, default=256)
    parser.add_argument("--score-probe-size", type=int, default=64)
    parser.add_argument("--current-probe-size", type=int, default=256)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--threshold-path",
        type=Path,
        default=Path("reports/optimizer_direction_horizon/loso_fold_metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/candidate_directional_guard_smoke/raw"),
    )
    return parser


def main():
    args = build_parser().parse_args()
    root = args.project_root.resolve()
    if int(args.intervention_updates) > 300:
        raise ValueError("directional guard smoke does not permit more than 300 updates")
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
    if len(loader) < int(args.intervention_updates):
        raise ValueError(
            f"loader has {len(loader)} batches, fewer than "
            f"{args.intervention_updates} intervention updates"
        )
    algo = MLRTemporalRobotIGFA(n_tasks=10, cfg=cfg).to(args.device)
    entry = checkpoint_path(root, "V1", args.seed, args.stage - 1, 95)
    load_checkpoint_into_policy(algo.policy, entry)
    reset_logging_state(algo, args.stage)
    optimizer, scheduler, total_steps = create_optimizer_scheduler(algo, cfg, loader)
    label_probes, score_probes, current_probe = build_probes(
        root,
        args.seed,
        args.stage,
        args.label_probe_size,
        args.score_probe_size,
        args.current_probe_size,
    )
    threshold = resolve_guard_threshold(
        args.threshold_mode,
        args.seed,
        root / args.threshold_path if args.threshold_mode == "loso" else None,
    )
    print(
        f"[threshold] mode={args.threshold_mode} value={threshold:.17g}",
        flush=True,
    )
    print(
        f"[protocol] seed={args.seed} stage={args.stage} entry={entry} "
        f"warmup_epochs={args.warmup_epochs} loader_batches={len(loader)} "
        f"intervention_updates={args.intervention_updates} guard_scale={args.guard_scale} "
        f"threshold_mode={args.threshold_mode} threshold={threshold:.17g} "
        f"current={current_count} replay={replay_counts} "
        f"label_probe={args.label_probe_size}/task "
        f"score_probe={args.score_probe_size}/task current_probe={args.current_probe_size} "
        f"total_steps={total_steps}",
        flush=True,
    )
    for epoch in range(int(args.warmup_epochs)):
        loader.sampler.set_epoch(epoch)
        algo.policy.train()
        losses = []
        for data in loader:
            losses.append(
                algo.observe(
                    data, current_task_id=args.stage, epoch=epoch, ifa_info=None
                )
            )
        if scheduler is not None and epoch > 0:
            scheduler.step()
        print(
            f"[warmup] seed={args.seed} stage={args.stage} "
            f"epoch={epoch + 1}/{args.warmup_epochs} batches={len(losses)} "
            f"mean_loss={sum(losses) / len(losses):.8g} "
            f"lr={optimizer.param_groups[0]['lr']:.8g}",
            flush=True,
        )
    branch_state = capture_branch_state(algo, scheduler)
    all_steps = []
    all_probes = []
    all_old_tasks = []
    starts = {}
    for enabled in (False, True):
        arm = "guard" if enabled else "control"
        steps, probes, old_tasks = run_arm(
            algo,
            scheduler,
            branch_state,
            loader,
            args.seed,
            args.stage,
            args.warmup_epochs,
            args.intervention_updates,
            threshold,
            args.threshold_mode,
            args.guard_scale,
            enabled,
            label_probes,
            score_probes,
            current_probe,
            args.probe_batch_size,
        )
        starts[arm] = next(row for row in probes if int(row["update"]) == 0)
        all_steps.extend(steps)
        all_probes.extend(probes)
        all_old_tasks.extend(old_tasks)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"seed{args.seed}_stage{args.stage}"
    write_csv(args.output_dir / f"{prefix}_steps.csv", STEP_FIELDS, all_steps)
    write_csv(args.output_dir / f"{prefix}_probe.csv", PROBE_FIELDS, all_probes)
    write_csv(
        args.output_dir / f"{prefix}_old_task_probe.csv",
        OLD_TASK_FIELDS,
        all_old_tasks,
    )
    old_difference = abs(
        float(starts["control"]["old_mean_loss"])
        - float(starts["guard"]["old_mean_loss"])
    )
    current_difference = abs(
        float(starts["control"]["current_loss"])
        - float(starts["guard"]["current_loss"])
    )
    if old_difference > 1e-6 or current_difference > 1e-6:
        raise RuntimeError(
            f"branch start parity failed: old={old_difference} "
            f"current={current_difference}"
        )
    print(
        f"[complete] seed={args.seed} stage={args.stage} "
        f"arms=2 updates_per_arm={args.intervention_updates} "
        f"branch_old_abs={old_difference:.3g} "
        f"branch_current_abs={current_difference:.3g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
