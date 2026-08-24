#!/usr/bin/env python3
import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from libero.libero.benchmark import get_benchmark
from libero.lifelong.analysis.temporal_interference import (
    action_descriptor,
    action_relation,
    aggregate_bin_gradients,
    annotate_feature_records,
    build_raw_temporal_index,
    enumerate_natural_pairs,
    grouped_gradient_stats,
    mean_gradient,
    resolve_stage_checkpoint,
    select_records_by_bin,
)
from libero.lifelong.datasets import get_dataset
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.utils import control_seed, torch_load_model


GROUP_COLUMNS = {
    "all_trainable": ("cos_all", "norm_old_all", "norm_new_all"),
    "temporal_transformer": (
        "cos_transformer",
        "norm_old_transformer",
        "norm_new_transformer",
    ),
    "policy_head": ("cos_head", "norm_old_head", "norm_new_head"),
}
RAW_FIELDS = [
    "method", "seed", "checkpoint_stage", "checkpoint_epoch", "checkpoint_path",
    "old_task", "new_task", "old_bin", "new_bin", "n_old", "n_new",
    "cos_all", "cos_transformer", "cos_head", "norm_old_all", "norm_new_all",
    "norm_old_transformer", "norm_new_transformer", "norm_old_head", "norm_new_head",
    "action_similarity", "action_distance",
]
TASK_FIELDS = [
    "method", "seed", "checkpoint_stage", "checkpoint_epoch", "checkpoint_path",
    "old_task", "new_task", "n_old", "n_new", "cos_all", "cos_transformer",
    "cos_head", "norm_old_all", "norm_new_all", "norm_old_transformer",
    "norm_new_transformer", "norm_old_head", "norm_new_head", "action_similarity",
    "action_distance",
]
ALIGN_FIELDS = [
    "method", "seed", "task", "feature_record_count", "matched_count",
    "ambiguous_count", "unmatched_count", "coverage",
]


def write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_rows(path, fieldnames, rows, key_fields):
    existing = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8") as handle:
            existing = list(csv.DictReader(handle))
    replacements = {tuple(str(row[field]) for field in key_fields) for row in rows}
    kept = [
        row for row in existing
        if tuple(str(row[field]) for field in key_fields) not in replacements
    ]
    write_rows(path, fieldnames, kept + rows)


def stats_columns(stats):
    row = {}
    for group, columns in GROUP_COLUMNS.items():
        cosine, old_norm, new_norm = columns
        row[cosine] = stats[group]["cosine"]
        row[old_norm] = stats[group]["old_norm"]
        row[new_norm] = stats[group]["new_norm"]
    return row


def parse_pairs(values):
    if not values:
        return set(enumerate_natural_pairs())
    pairs = set()
    for value in values:
        old_task, new_task = value.split("-", 1)
        pairs.add((int(old_task), int(new_task)))
    return pairs


def load_config(path):
    with path.open(encoding="utf-8") as handle:
        return OmegaConf.create(json.load(handle))


def build_datasets(cfg):
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    datasets = {}
    shape_meta = None
    for task in range(6, 10):
        path = Path(cfg.folder) / benchmark.get_task_demonstration(task)
        dataset, task_shape = get_dataset(
            dataset_path=str(path),
            obs_modality=cfg.data.obs.modality,
            seq_len=cfg.data.seq_len,
            frame_stack=cfg.data.frame_stack,
            demo_limit=cfg.data.demo_limit,
            train=True,
        )
        datasets[task] = dataset
        shape_meta = task_shape
        print(f"[dataset] task={task} name={benchmark.get_task(task).language} path={path} records={len(dataset)}")
    return datasets, shape_meta


def load_policy(cfg, shape_meta, checkpoint, device):
    policy = BriefPolicy(cfg, shape_meta).to(device)
    state_dict = torch_load_model(str(checkpoint), map_location="cpu")[0]
    incompatible = policy.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint incompatibility: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    policy.eval()
    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("mlr", "mlr_ifa"), default="mlr")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--new-tasks", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--pairs", nargs="*", help="optional OLD-NEW pairs")
    parser.add_argument("--samples-per-bin", type=int, default=32)
    parser.add_argument("--gradient-batch-size", type=int, default=8)
    parser.add_argument("--checkpoint-epoch", type=int, default=95)
    parser.add_argument("--sample-seed", type=int, default=20260823)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--features-root", type=Path, default=Path("features"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/temporal_interference_diagnosis"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--append", action="store_true")
    args = parser.parse_args()

    requested_pairs = parse_pairs(args.pairs)
    output = args.output_dir
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    alignment_rows = []
    task_rows = []

    for seed in args.seeds:
        control_seed(seed)
        run_dir = args.runs_root / f"object_seed{seed}_{args.method}" / "run_001"
        cfg = load_config(run_dir / "config.json")
        datasets, shape_meta = build_datasets(cfg)
        selected = {}
        descriptors = {}
        for task, dataset in datasets.items():
            raw_index = build_raw_temporal_index(dataset)
            records, summary = annotate_feature_records(
                args.features_root / f"object_seed{seed}", raw_index, task
            )
            summary.update({"method": args.method, "seed": seed, "task": task})
            alignment_rows.append(summary)
            selected[task] = select_records_by_bin(
                records,
                args.samples_per_bin,
                args.sample_seed + seed * 1000 + task * 10,
            )
            descriptors[task] = {
                bin_id: action_descriptor(bin_records)
                for bin_id, bin_records in selected[task].items()
            }
            print(
                f"[alignment] method={args.method} seed={seed} task={task} "
                f"matched={summary['matched_count']}/{summary['feature_record_count']} "
                f"coverage={summary['coverage']:.6f}"
            )

        for new_task in args.new_tasks:
            stage = new_task - 1
            stage_pairs = sorted(
                pair for pair in requested_pairs if pair[1] == new_task
            )
            if not stage_pairs:
                continue
            checkpoint = resolve_stage_checkpoint(run_dir, stage, args.checkpoint_epoch)
            print(
                f"[checkpoint] method={args.method} seed={seed} stage={stage} "
                f"epoch={args.checkpoint_epoch} path={checkpoint} pairs={stage_pairs}"
            )
            policy = load_policy(cfg, shape_meta, checkpoint, args.device)
            needed_tasks = sorted({task for pair in stage_pairs for task in pair})
            gradients = {}
            task_gradients = {}
            unused_names = set()
            for task in needed_tasks:
                gradients[task] = {}
                counts = {}
                for bin_id in range(5):
                    bin_records = selected[task][bin_id]
                    gradient, unused = mean_gradient(
                        policy, bin_records, args.gradient_batch_size, args.device
                    )
                    gradients[task][bin_id] = gradient
                    counts[bin_id] = len(bin_records)
                    unused_names.update(unused)
                    print(
                        f"[gradient] method={args.method} seed={seed} stage={stage} "
                        f"task={task} bin={bin_id} samples={len(bin_records)}"
                    )
                task_gradients[task] = aggregate_bin_gradients(gradients[task], counts)
            if unused_names:
                print(f"[gradient] unused_trainable_parameters={sorted(unused_names)}")

            for old_task, pair_new_task in stage_pairs:
                rows = []
                for old_bin in range(5):
                    for new_bin in range(5):
                        stats = grouped_gradient_stats(
                            gradients[old_task][old_bin], gradients[pair_new_task][new_bin]
                        )
                        action_similarity, action_distance = action_relation(
                            descriptors[old_task][old_bin], descriptors[pair_new_task][new_bin]
                        )
                        rows.append(
                            {
                                "method": args.method,
                                "seed": seed,
                                "checkpoint_stage": stage,
                                "checkpoint_epoch": args.checkpoint_epoch,
                                "checkpoint_path": str(checkpoint),
                                "old_task": old_task,
                                "new_task": pair_new_task,
                                "old_bin": old_bin,
                                "new_bin": new_bin,
                                "n_old": len(selected[old_task][old_bin]),
                                "n_new": len(selected[pair_new_task][new_bin]),
                                **stats_columns(stats),
                                "action_similarity": action_similarity,
                                "action_distance": action_distance,
                            }
                        )
                raw_path = raw_dir / (
                    f"{args.method}_seed{seed}_stage{stage}_task{old_task}_vs_{pair_new_task}_cosines.csv"
                )
                write_rows(raw_path, RAW_FIELDS, rows)

                task_stats = grouped_gradient_stats(
                    task_gradients[old_task], task_gradients[pair_new_task]
                )
                old_all = [item for values in selected[old_task].values() for item in values]
                new_all = [item for values in selected[pair_new_task].values() for item in values]
                action_similarity, action_distance = action_relation(
                    action_descriptor(old_all), action_descriptor(new_all)
                )
                task_rows.append(
                    {
                        "method": args.method,
                        "seed": seed,
                        "checkpoint_stage": stage,
                        "checkpoint_epoch": args.checkpoint_epoch,
                        "checkpoint_path": str(checkpoint),
                        "old_task": old_task,
                        "new_task": pair_new_task,
                        "n_old": len(old_all),
                        "n_new": len(new_all),
                        **stats_columns(task_stats),
                        "action_similarity": action_similarity,
                        "action_distance": action_distance,
                    }
                )
                print(f"[pair] wrote={raw_path} task_level_cos={task_stats['all_trainable']['cosine']:.6f}")

            del policy, gradients, task_gradients
            gc.collect()
            torch.cuda.empty_cache()

        del datasets
        gc.collect()

    merge_rows(
        output / "alignment_summary.csv", ALIGN_FIELDS, alignment_rows,
        ("method", "seed", "task"),
    )
    merge_rows(
        output / "task_level_pairs.csv", TASK_FIELDS, task_rows,
        ("method", "seed", "checkpoint_stage", "old_task", "new_task"),
    )
    print(f"[complete] output={output} pairs={len(task_rows)}")


if __name__ == "__main__":
    main()
