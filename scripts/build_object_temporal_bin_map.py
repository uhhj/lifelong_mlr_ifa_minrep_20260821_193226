#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from libero.libero.benchmark import get_benchmark
from libero.lifelong.analysis.temporal_interference import (
    annotate_feature_records,
    build_raw_temporal_index,
)
from libero.lifelong.datasets import get_dataset


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        return OmegaConf.create(json.load(handle))


def build_sequence_dataset(cfg, benchmark, task_id):
    dataset_path = Path(cfg.folder) / benchmark.get_task_demonstration(task_id)
    dataset, _ = get_dataset(
        dataset_path=str(dataset_path),
        obs_modality=cfg.data.obs.modality,
        seq_len=cfg.data.seq_len,
        frame_stack=cfg.data.frame_stack,
        demo_limit=cfg.data.demo_limit,
        train=True,
    )
    return dataset, dataset_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bins", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if int(cfg.seed) != args.seed:
        raise ValueError(f"config seed {cfg.seed} does not match requested seed {args.seed}")
    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    records = {}
    alignment = {}

    for task_id in range(6, 10):
        dataset, dataset_path = build_sequence_dataset(cfg, benchmark, task_id)
        raw_index = build_raw_temporal_index(dataset)
        matched, summary = annotate_feature_records(
            args.feature_root, raw_index, task_id, n_bins=args.n_bins
        )
        if summary["unmatched_count"] != 0:
            raise RuntimeError(
                f"task {task_id} has {summary['unmatched_count']} unmatched feature records"
            )

        feature_paths = sorted((args.feature_root / str(task_id)).glob("*.pt"))
        for path in feature_paths:
            try:
                record_id = int(path.stem)
            except ValueError as error:
                raise ValueError(f"non-canonical formal feature filename: {path}") from error
            records[f"{task_id}:{record_id}"] = -1
        for record in matched:
            record_id = int(Path(record["path"]).stem)
            records[f"{task_id}:{record_id}"] = int(record["bin_id"])

        if len(feature_paths) != summary["feature_record_count"]:
            raise RuntimeError(f"task {task_id} feature accounting mismatch")
        if summary["matched_count"] + summary["ambiguous_count"] != len(feature_paths):
            raise RuntimeError(f"task {task_id} resolution accounting mismatch")
        alignment[str(task_id)] = {
            **summary,
            "dataset_path": str(dataset_path),
        }
        print(
            f"[temporal-map] seed={args.seed} task={task_id} "
            f"matched={summary['matched_count']}/{summary['feature_record_count']} "
            f"ambiguous={summary['ambiguous_count']} unmatched=0 "
            f"coverage={summary['coverage']:.6f}"
        )

    payload = {
        "seed": args.seed,
        "n_bins": args.n_bins,
        "alignment": alignment,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[temporal-map] output={args.output} records={len(records)}")


if __name__ == "__main__":
    main()
