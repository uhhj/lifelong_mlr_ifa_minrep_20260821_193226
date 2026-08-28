#!/usr/bin/env python3
import argparse
import csv
import gc
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from libero.lifelong.analysis.per_task_temporal_interference import (
    build_stage_records,
    deterministic_diagnostic_batches,
    diagnose_batch,
    load_temporal_bin_map,
)
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.utils import control_seed, torch_load_model


RAW_FIELDS = [
    "analysis_phase",
    "seed",
    "checkpoint_stage",
    "checkpoint_epoch",
    "checkpoint_path",
    "new_task",
    "batch_index",
    "bin",
    "old_task",
    "n_new_bin",
    "n_old_task_bin",
    "n_replay_bin_total",
    "n_old_tasks_present",
    "dot_individual_pre",
    "cos_individual_pre",
    "dot_individual_post",
    "cos_individual_post",
    "dot_aggregate_pre",
    "cos_aggregate_pre",
    "aggregate_gate_fired",
    "individual_conflict_pre",
    "individual_conflict_post",
    "multi_old_candidate",
    "mixed_old_sign",
    "masked_conflict",
    "residual_conflict",
]


def load_config(path, project_root, seed, device):
    with Path(path).open(encoding="utf-8") as handle:
        cfg = OmegaConf.create(json.load(handle))
    cfg.device = str(device)
    cfg.training = True
    cfg.aug = False
    cfg.pretrain = str(project_root / "checkpoints" / "object_multitask_model_ep99.pth")
    cfg.feature_dir = str(project_root / "features" / f"object_seed{seed}")
    return cfg


def resolve_checkpoint(run_dir, stage, epoch):
    path = Path(run_dir) / f"mlrifa_{stage}_model_ep{epoch}.pth"
    if not path.is_file():
        raise FileNotFoundError(f"missing exact diagnostic checkpoint: {path}")
    return path


def load_policy(cfg, checkpoint, device):
    policy = BriefPolicy(cfg, None).to(device)
    state_dict = torch_load_model(str(checkpoint), map_location="cpu")[0]
    incompatible = policy.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint incompatibility: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    policy.eval()
    return policy


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--new-tasks", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--batches-per-stage", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--diagnostic-seed", type=int, default=20260828)
    parser.add_argument("--checkpoint-epoch", type=int, default=95)
    parser.add_argument(
        "--checkpoint-mode",
        choices=("entry", "late"),
        default="entry",
        help="entry uses previous-stage checkpoint; late uses current-stage checkpoint",
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/per_task_temporal_interference_diagnosis/raw"),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        control_seed(seed)
        run_dir = (
            project_root
            / "runs"
            / f"object_seed{seed}_mlr_temporal_igfa"
            / "run_001"
        )
        cfg = load_config(run_dir / "config.json", project_root, seed, args.device)
        sidecar = load_temporal_bin_map(
            project_root / "metadata" / "temporal_bins" / f"object_seed{seed}.json",
            expected_seed=seed,
        )
        feature_root = project_root / "features" / f"object_seed{seed}"
        for new_task in args.new_tasks:
            checkpoint_stage = (
                int(new_task) - 1
                if args.checkpoint_mode == "entry"
                else int(new_task)
            )
            checkpoint = resolve_checkpoint(
                run_dir, checkpoint_stage, args.checkpoint_epoch
            )
            records, replay_paths = build_stage_records(
                feature_root,
                seed,
                int(new_task),
                sidecar,
                rate=0.5,
            )
            sample_seed = args.diagnostic_seed + seed * 100 + int(new_task)
            batches = deterministic_diagnostic_batches(
                records,
                args.batch_size,
                args.batches_per_stage,
                sample_seed,
            )
            print(
                f"[protocol] phase={args.checkpoint_mode} seed={seed} "
                f"new_task={new_task} checkpoint_stage={checkpoint_stage} "
                f"checkpoint_epoch={args.checkpoint_epoch} batches={len(batches)} "
                f"batch_size={args.batch_size} records={len(records)} "
                f"replay_counts={{{', '.join(f'{k}: {len(v)}' for k, v in replay_paths.items())}}}"
            )
            policy = load_policy(cfg, checkpoint, args.device)
            rows = []
            candidate_count = 0
            multi_old_count = 0
            max_parity_abs = 0.0
            max_parity_relative_l2 = 0.0
            for batch_index, batch in enumerate(batches):
                results, parity = diagnose_batch(policy, batch, int(new_task))
                max_parity_abs = max(max_parity_abs, parity["max_parity_abs"])
                max_parity_relative_l2 = max(
                    max_parity_relative_l2,
                    parity["max_parity_relative_l2"],
                )
                for result in results:
                    candidate_count += 1
                    multi_old_count += int(result["multi_old"])
                    for old_task in result["present_old_tasks"]:
                        pre = result["pre_relations"][old_task]
                        post = result["post_relations"][old_task]
                        aggregate = result["aggregate_relation"]
                        rows.append(
                            {
                                "analysis_phase": args.checkpoint_mode,
                                "seed": seed,
                                "checkpoint_stage": checkpoint_stage,
                                "checkpoint_epoch": args.checkpoint_epoch,
                                "checkpoint_path": str(checkpoint),
                                "new_task": new_task,
                                "batch_index": batch_index,
                                "bin": result["bin"],
                                "old_task": old_task,
                                "n_new_bin": result["n_new_bin"],
                                "n_old_task_bin": result["old_counts"][old_task],
                                "n_replay_bin_total": result["n_replay_bin_total"],
                                "n_old_tasks_present": len(result["present_old_tasks"]),
                                "dot_individual_pre": pre["dot"],
                                "cos_individual_pre": pre["cosine"],
                                "dot_individual_post": post["dot"],
                                "cos_individual_post": post["cosine"],
                                "dot_aggregate_pre": aggregate["dot"],
                                "cos_aggregate_pre": aggregate["cosine"],
                                "aggregate_gate_fired": int(result["aggregate_gate_fired"]),
                                "individual_conflict_pre": int(pre["dot"] < 0.0),
                                "individual_conflict_post": int(post["dot"] < 0.0),
                                "multi_old_candidate": int(result["multi_old"]),
                                "mixed_old_sign": int(result["mixed_old_sign"]),
                                "masked_conflict": int(result["masked_conflict"]),
                                "residual_conflict": int(result["residual_conflict"]),
                            }
                        )
                if (batch_index + 1) % 50 == 0:
                    print(
                        f"[progress] phase={args.checkpoint_mode} seed={seed} "
                        f"new_task={new_task} batches={batch_index + 1}/{len(batches)} "
                        f"candidates={candidate_count} multi_old={multi_old_count}"
                    )
            raw_path = args.output_dir / (
                f"{args.checkpoint_mode}_seed{seed}_new{new_task}_"
                f"stage{checkpoint_stage}_ep{args.checkpoint_epoch}.csv"
            )
            write_rows(raw_path, rows)
            print(
                f"[complete] raw={raw_path} rows={len(rows)} "
                f"candidate_bins={candidate_count} multi_old={multi_old_count} "
                f"parity_max_abs={max_parity_abs:.9g} "
                f"parity_relative_l2={max_parity_relative_l2:.9g}"
            )
            del policy, batches, records, rows
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
