#!/usr/bin/env python3
import argparse
import csv
import gc
import random
from pathlib import Path

import torch

from libero.lifelong.analysis.first_order_retention_validity import (
    backend_displacement,
    capture_backend_parameters,
    classify_checkpoint_transition,
    evaluate_mean_loss,
    load_checkpoint_into_policy,
    mean_loss_and_group_gradients,
    parameter_displacement_dot,
    simulate_layer_a_batch,
    tuple_norm,
)
from libero.lifelong.analysis.per_task_temporal_interference import (
    build_stage_records,
    deterministic_diagnostic_batches,
    load_temporal_bin_map,
)
from libero.lifelong.models.brief import BriefPolicy
from scripts.diagnose_per_task_temporal_interference import load_config


RUN_SUFFIX = {
    "V1": "mlr_temporal_igfa",
    "V2": "mlr_per_task_temporal_igfa",
}
LAYER_A_FIELDS = [
    "method", "seed", "stage", "batch", "bin", "old_task",
    "projected_current_dot", "final_head_dot", "transformer_dot", "full_dot",
    "projected_current_safe", "head_composition_leak", "transformer_flip",
    "full_first_order_unsafe", "projected_current_cosine", "final_head_cosine",
    "transformer_cosine", "full_cosine", "head_grad_norm",
    "transformer_grad_norm", "old_head_grad_norm", "old_transformer_grad_norm",
    "pre_clip_global_norm",
]
CHECKPOINT_FIELDS = [
    "method", "seed", "stage", "old_task", "start_epoch", "end_epoch",
    "interval_type", "loss_start", "loss_end", "actual_delta",
    "relative_actual_delta", "pred_head", "pred_transformer", "pred_all",
    "head_false_safe", "checkpoint_transformer_flip",
    "full_first_order_false_safe", "sign_agreement",
    "absolute_prediction_error", "head_delta_norm", "transformer_delta_norm",
    "material_change",
]


def write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_dir(project_root, method, seed):
    return project_root / "runs" / f"object_seed{seed}_{RUN_SUFFIX[method]}" / "run_001"


def checkpoint_path(project_root, method, seed, stage, epoch):
    path = run_dir(project_root, method, seed) / f"mlrifa_{stage}_model_ep{epoch}.pth"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def build_probe_manifest(project_root, seeds, probe_size, probe_seed):
    manifest = []
    probes = {}
    for seed in seeds:
        for task in (6, 7, 8):
            paths = sorted(
                (project_root / "features" / f"object_seed{seed}" / str(task)).glob("*.pt"),
                key=lambda path: int(path.stem),
            )
            if len(paths) < probe_size:
                raise ValueError(f"task {task} seed {seed} has only {len(paths)} probes")
            selected = random.Random(probe_seed + seed * 100 + task).sample(paths, probe_size)
            probes[(seed, task)] = selected
            for path in selected:
                manifest.append(
                    {
                        "seed": seed,
                        "old_task": task,
                        "record_id": int(path.stem),
                        "path": str(path),
                    }
                )
    return probes, manifest


def run_layer_a(args):
    rows = []
    root = args.project_root.resolve()
    for method in args.methods:
        for seed in args.seeds:
            sidecar = load_temporal_bin_map(
                root / "metadata" / "temporal_bins" / f"object_seed{seed}.json",
                expected_seed=seed,
            )
            features = root / "features" / f"object_seed{seed}"
            for stage in args.stages:
                checkpoint = checkpoint_path(root, method, seed, stage - 1, 95)
                cfg = load_config(run_dir(root, method, seed) / "config.json", root, seed, args.device)
                policy = BriefPolicy(cfg, None).to(args.device)
                load_checkpoint_into_policy(policy, checkpoint)
                records, replay = build_stage_records(features, seed, stage, sidecar, rate=0.5)
                batch_count = args.stage7_batches if stage == 7 else args.batches
                batches = deterministic_diagnostic_batches(
                    records, args.batch_size, batch_count,
                    args.diagnostic_seed + seed * 100 + stage,
                )
                print(
                    f"[layer-a] method={method} seed={seed} stage={stage} "
                    f"batches={len(batches)} replay="
                    f"{{{', '.join(f'{task}:{len(paths)}' for task, paths in replay.items())}}}"
                )
                relation_count = 0
                for batch_index, batch in enumerate(batches):
                    batch_rows, _ = simulate_layer_a_batch(
                        policy, batch, stage, method,
                        gate_strength=1.0, n_bins=5, grad_clip=100.0,
                    )
                    for row in batch_rows:
                        rows.append(
                            {
                                "method": method, "seed": seed, "stage": stage,
                                "batch": batch_index, **row,
                            }
                        )
                    relation_count += len(batch_rows)
                print(f"[layer-a-complete] relations={relation_count}")
                del policy, batches, records
                gc.collect()
                torch.cuda.empty_cache()
    write_csv(args.output_dir / "layer_a_raw.csv", LAYER_A_FIELDS, rows)
    print(f"[complete] layer_a_rows={len(rows)}")


def transition_rows(
    policy,
    root,
    method,
    seed,
    stage,
    start_stage,
    start_epoch,
    end_epoch,
    interval_type,
    probes,
    probe_batch_size,
):
    start_checkpoint = checkpoint_path(root, method, seed, start_stage, start_epoch)
    end_checkpoint = checkpoint_path(root, method, seed, stage, end_epoch)
    load_checkpoint_into_policy(policy, start_checkpoint)
    start_parameters = capture_backend_parameters(policy)
    start = {}
    for old_task in range(6, stage):
        start[old_task] = mean_loss_and_group_gradients(
            policy, probes[(seed, old_task)], batch_size=probe_batch_size
        )
    load_checkpoint_into_policy(policy, end_checkpoint)
    end_parameters = capture_backend_parameters(policy)
    displacement = backend_displacement(start_parameters, end_parameters)
    head_delta_norm = tuple_norm(displacement["head"])
    transformer_delta_norm = tuple_norm(displacement["transformer"])
    rows = []
    for old_task in range(6, stage):
        loss_start = start[old_task]["mean_loss"]
        loss_end = evaluate_mean_loss(
            policy, probes[(seed, old_task)], batch_size=probe_batch_size
        )
        actual_delta = loss_end - loss_start
        pred_head = parameter_displacement_dot(
            start[old_task]["head_gradient"], displacement["head"]
        )
        pred_transformer = parameter_displacement_dot(
            start[old_task]["transformer_gradient"], displacement["transformer"]
        )
        classification = classify_checkpoint_transition(
            pred_head, pred_transformer, actual_delta
        )
        rows.append(
            {
                "method": method, "seed": seed, "stage": stage,
                "old_task": old_task, "start_epoch": start_epoch,
                "end_epoch": end_epoch, "interval_type": interval_type,
                "loss_start": loss_start, "loss_end": loss_end,
                "actual_delta": actual_delta,
                "relative_actual_delta": actual_delta / max(abs(loss_start), 1e-12),
                "pred_head": pred_head, "pred_transformer": pred_transformer,
                **classification,
                "head_delta_norm": head_delta_norm,
                "transformer_delta_norm": transformer_delta_norm,
                "material_change": int(abs(actual_delta / max(abs(loss_start), 1e-12)) >= 0.005),
            }
        )
    return rows


def run_checkpoint(args):
    root = args.project_root.resolve()
    probes, manifest = build_probe_manifest(
        root, args.seeds, args.probe_size, args.probe_seed
    )
    write_csv(args.output_dir / "probe_manifest.csv", ["seed", "old_task", "record_id", "path"], manifest)
    short_rows = []
    long_rows = []
    intervals = list(zip(args.short_epochs[:-1], args.short_epochs[1:]))
    for method in args.methods:
        for seed in args.seeds:
            cfg = load_config(run_dir(root, method, seed) / "config.json", root, seed, args.device)
            policy = BriefPolicy(cfg, None).to(args.device)
            for stage in args.stages:
                for start_epoch, end_epoch in intervals:
                    rows = transition_rows(
                        policy, root, method, seed, stage, stage,
                        start_epoch, end_epoch, "short", probes,
                        args.probe_batch_size,
                    )
                    short_rows.extend(rows)
                    print(
                        f"[checkpoint-short] method={method} seed={seed} stage={stage} "
                        f"epochs={start_epoch}->{end_epoch} cases={len(rows)}"
                    )
                if not args.skip_long:
                    rows = transition_rows(
                        policy, root, method, seed, stage, stage - 1,
                        95, 95, "long_horizon", probes, args.probe_batch_size,
                    )
                    long_rows.extend(rows)
                    print(
                        f"[checkpoint-long] method={method} seed={seed} stage={stage} "
                        f"cases={len(rows)}"
                    )
            del policy
            gc.collect()
            torch.cuda.empty_cache()
    write_csv(args.output_dir / "checkpoint_transition_raw.csv", CHECKPOINT_FIELDS, short_rows)
    if not args.skip_long:
        write_csv(args.output_dir / "long_horizon_raw.csv", CHECKPOINT_FIELDS, long_rows)
    print(
        f"[complete] probe_records={len(manifest)} short_cases={len(short_rows)} "
        f"long_cases={len(long_rows)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("layer_a", "checkpoint"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=Path("reports/first_order_retention_validity"))
    parser.add_argument("--methods", nargs="+", choices=("V1", "V2"), default=["V1", "V2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--stages", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--batches", type=int, default=150)
    parser.add_argument("--stage7-batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--diagnostic-seed", type=int, default=20260830)
    parser.add_argument("--probe-size", type=int, default=256)
    parser.add_argument("--probe-seed", type=int, default=20260830)
    parser.add_argument("--probe-batch-size", type=int, default=32)
    parser.add_argument("--short-epochs", nargs="+", type=int, default=[70, 75, 80, 85, 90, 95])
    parser.add_argument("--skip-long", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "layer_a":
        run_layer_a(args)
    else:
        run_checkpoint(args)


if __name__ == "__main__":
    main()
