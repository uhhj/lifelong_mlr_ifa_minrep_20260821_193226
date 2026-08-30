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

from libero.libero.benchmark import get_benchmark
from libero.lifelong.analysis.policy_distribution_retention import (
    aggregate_phase_metric,
    deterministic_gmm_forward_kl,
    diagonal_gaussian_forward_kl,
    diagonal_gaussian_symmetric_kl,
    diagonal_gaussian_w2_sq,
    extract_policy_head_stats,
    quantile,
    stable_gmm_log_prob,
)
from libero.lifelong.analysis.temporal_interference import (
    annotate_feature_records,
    build_raw_temporal_index,
    select_records_by_bin,
)
from libero.lifelong.datasets import get_dataset
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.utils import control_seed, torch_load_model


RUN_SUFFIX = {
    "temporal_v1": "mlr_temporal_igfa",
    "zero_guard": "mlr_zero_directional_guard",
}
EPOCHS = (70, 75, 80, 85, 90, 95)
TASKS = (6, 7, 8)


def load_config(path, root, seed, device):
    with Path(path).open(encoding="utf-8") as handle:
        cfg = OmegaConf.create(json.load(handle))
    cfg.device = str(device)
    cfg.training = True
    cfg.aug = False
    cfg.pretrain = str(root / "checkpoints" / "object_multitask_model_ep99.pth")
    cfg.feature_dir = str(root / "features" / f"object_seed{seed}")
    cfg.folder = str(root / "data")
    return cfg


def checkpoint_path(root, method, seed, stage, epoch):
    path = (
        root / "runs" / f"object_seed{seed}_{RUN_SUFFIX[method]}" / "run_001"
        / f"mlrifa_{stage}_model_ep{epoch}.pth"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_checkpoint(policy, path):
    state_dict = torch_load_model(str(path), map_location="cpu")[0]
    incompatible = policy.load_state_dict(state_dict, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint incompatibility {path}: missing={incompatible.missing_keys} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    policy.eval()


def load_selected_epochs(path):
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            result[(row["method"], int(row["seed"]), int(row["stage"]))] = int(
                row["selected_epoch"]
            )
    if len(result) != 24:
        raise ValueError(f"expected 24 selected epochs, found {len(result)}")
    return result


def build_phase_manifest(root, output, seeds, tasks, samples_per_bin):
    rows = []
    for seed in seeds:
        config_path = (
            root / "runs" / f"object_seed{seed}_mlr_temporal_igfa" / "run_001"
            / "config.json"
        )
        cfg = load_config(config_path, root, seed, "cpu")
        benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
        for task in tasks:
            dataset_path = Path(cfg.folder) / benchmark.get_task_demonstration(task)
            dataset, _ = get_dataset(
                dataset_path=str(dataset_path),
                obs_modality=cfg.data.obs.modality,
                seq_len=cfg.data.seq_len,
                frame_stack=cfg.data.frame_stack,
                demo_limit=cfg.data.demo_limit,
                train=True,
            )
            raw_index = build_raw_temporal_index(dataset)
            records, summary = annotate_feature_records(
                root / "features" / f"object_seed{seed}", raw_index, task, n_bins=5
            )
            selected = select_records_by_bin(
                records,
                samples_per_bin,
                20260823 + seed * 1000 + task * 10,
                n_bins=5,
            )
            for bin_id in range(5):
                for record in selected[bin_id]:
                    rows.append({
                        "seed": seed,
                        "task": task,
                        "bin": bin_id,
                        "record_id": int(Path(record["path"]).stem),
                        "path": record["path"],
                        "progress": record["progress"],
                        "demo_id": record["demo_id"],
                        "rank_in_demo": record["rank_in_demo"],
                        "demo_sequence_count": record["demo_sequence_count"],
                    })
            print(
                f"[phase-manifest] seed={seed} task={task} "
                f"matched={summary['matched_count']} selected={5 * samples_per_bin}",
                flush=True,
            )
    expected = len(seeds) * len(tasks) * 5 * samples_per_bin
    if len(rows) != expected:
        raise RuntimeError(f"phase manifest expected {expected}, found {len(rows)}")
    write_rows(output, rows)
    return rows


def read_global_manifest(path, root, seeds, tasks, limit=None):
    grouped = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed, task = int(row["seed"]), int(row["old_task"])
            if seed not in seeds or task not in tasks:
                continue
            recorded = Path(row["path"])
            record_id = int(row["record_id"])
            resolved = recorded if recorded.is_file() else (
                root / "features" / f"object_seed{seed}" / str(task)
                / f"{record_id}.pt"
            )
            if not resolved.is_file():
                raise FileNotFoundError(resolved)
            grouped[(seed, task)].append({
                "path": str(resolved), "record_id": record_id,
            })
    for key in grouped:
        grouped[key].sort(key=lambda row: row["record_id"])
        if limit:
            grouped[key] = grouped[key][: int(limit)]
    expected = int(limit) if limit else 256
    for key in [(seed, task) for seed in seeds for task in tasks]:
        if len(grouped[key]) != expected:
            raise ValueError(f"global probe {key} has {len(grouped[key])}, expected {expected}")
    return dict(grouped)


def read_phase_manifest(path, seeds, tasks, samples_per_bin):
    grouped = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            seed, task = int(row["seed"]), int(row["task"])
            if seed in seeds and task in tasks:
                grouped[(seed, task)].append({
                    **row,
                    "seed": seed,
                    "task": task,
                    "bin": int(row["bin"]),
                    "record_id": int(row["record_id"]),
                })
    for key, records in grouped.items():
        records.sort(key=lambda row: (row["bin"], row["record_id"]))
        counts = [sum(row["bin"] == bin_id for row in records) for bin_id in range(5)]
        if counts != [samples_per_bin] * 5:
            raise ValueError(f"phase probe {key} counts={counts}")
    if len(grouped) != len(seeds) * len(tasks):
        raise ValueError("phase probe group coverage mismatch")
    return dict(grouped)


def load_probe(records):
    features, actions, ids, bins = [], [], [], []
    for record in records:
        saved = torch.load(record["path"], map_location="cpu")
        feature, action = saved["feature"], saved["actions"]
        while feature.ndim > 3 and feature.shape[0] == 1:
            feature = feature.squeeze(0)
        while action.ndim > 2 and action.shape[0] == 1:
            action = action.squeeze(0)
        if tuple(feature.shape) != (8, 4, 512) or tuple(action.shape) != (8, 7):
            raise ValueError(f"unexpected feature/action shape: {record['path']}")
        features.append(feature)
        actions.append(action)
        ids.append(int(record["record_id"]))
        bins.append(int(record.get("bin", -1)))
    return {
        "features": torch.stack(features),
        "actions": torch.stack(actions),
        "record_ids": ids,
        "bins": bins,
    }


@torch.no_grad()
def extract_stats(policy, probe, device, batch_size):
    chunks = defaultdict(list)
    for start in range(0, len(probe["features"]), batch_size):
        features = probe["features"][start : start + batch_size].to(device)
        stats = extract_policy_head_stats(policy, features)
        for key, value in stats.items():
            chunks[key].append(value.cpu())
    return {key: torch.cat(values) for key, values in chunks.items()}


def record_seeds(seed, task, record_ids):
    return [
        (20260830 + seed * 1000003 + task * 1009 + int(record_id))
        % (2**31 - 1)
        for record_id in record_ids
    ]


def derive_values(anchor, current, probe, seed, task, mc_samples):
    kl = diagonal_gaussian_forward_kl(
        anchor["rollout_mu"], anchor["rollout_std"],
        current["rollout_mu"], current["rollout_std"],
    ).cpu()
    w2 = diagonal_gaussian_w2_sq(
        anchor["rollout_mu"], anchor["rollout_std"],
        current["rollout_mu"], current["rollout_std"],
    ).cpu()
    symkl = diagonal_gaussian_symmetric_kl(
        anchor["rollout_mu"], anchor["rollout_std"],
        current["rollout_mu"], current["rollout_std"],
    ).cpu()
    mean_l2 = torch.linalg.vector_norm(
        anchor["rollout_mu"].double() - current["rollout_mu"].double(), dim=-1
    ).cpu()
    bc = -stable_gmm_log_prob(
        probe["actions"][:, -1, :], current["means"], current["stds"],
        current["logits"],
    ).cpu()
    result = {
        "kl": kl,
        "w2": w2,
        "symkl": symkl,
        "mean_l2": mean_l2,
        "bc": bc,
        "mode_switch": (anchor["rollout_mode"] != current["rollout_mode"]).double(),
    }
    if mc_samples:
        result["full_gmm"] = deterministic_gmm_forward_kl(
            anchor, current, sample_count=mc_samples,
            generator_seed=record_seeds(seed, task, probe["record_ids"]),
        )
    return result


def phase_fields(prefix, values, bins):
    aggregate = aggregate_phase_metric(values.tolist(), bins)
    bin_prefix = "bc_phase" if prefix == "bc" else prefix
    return {
        f"{prefix}_phase_mean": aggregate["mean"],
        **{f"{bin_prefix}_bin{index}": aggregate[f"bin{index}"] for index in range(5)},
        f"{prefix}_phase_worst_bin": aggregate["worst_bin"],
        f"{prefix}_phase_late_bin": aggregate["late_bin"],
    }


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--methods", nargs="+", default=list(RUN_SUFFIX), choices=RUN_SUFFIX)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--tasks", nargs="+", type=int, default=list(TASKS))
    parser.add_argument("--stages", nargs="+", type=int, default=[7, 8, 9])
    parser.add_argument("--epochs", nargs="+", type=int, default=list(EPOCHS))
    parser.add_argument("--samples-per-bin", type=int, default=32)
    parser.add_argument("--global-limit", type=int)
    parser.add_argument("--mc-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--build-phase-manifest", action="store_true")
    parser.add_argument(
        "--global-manifest", type=Path,
        default=Path("reports/first_order_retention_validity/probe_manifest.csv"),
    )
    parser.add_argument(
        "--phase-manifest", type=Path,
        default=Path("reports/success_critical_retention_signal/phase_probe_manifest.csv"),
    )
    parser.add_argument(
        "--selected-epochs", type=Path,
        default=Path("reports/bc_rollout_retention_alignment/selected_epochs.csv"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("reports/success_critical_retention_signal/signal_checkpoint_grid.csv"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    phase_path = resolve(args.phase_manifest)
    if args.build_phase_manifest:
        build_phase_manifest(
            root, phase_path, args.seeds, args.tasks, args.samples_per_bin
        )
    global_groups = read_global_manifest(
        resolve(args.global_manifest), root, args.seeds, args.tasks, args.global_limit
    )
    phase_groups = read_phase_manifest(
        phase_path, args.seeds, args.tasks, args.samples_per_bin
    )
    selected = load_selected_epochs(resolve(args.selected_epochs))
    rows = []
    for method in args.methods:
        for seed in args.seeds:
            control_seed(seed)
            run = root / "runs" / f"object_seed{seed}_{RUN_SUFFIX[method]}" / "run_001"
            cfg = load_config(run / "config.json", root, seed, args.device)
            policy = BriefPolicy(cfg, None).to(args.device)
            probes = {}
            anchors = {}
            for task in args.tasks:
                probes[task] = {
                    "global": load_probe(global_groups[(seed, task)]),
                    "phase": load_probe(phase_groups[(seed, task)]),
                }
                anchor_epoch = selected[(method, seed, task)]
                anchor_checkpoint = checkpoint_path(
                    root, method, seed, task, anchor_epoch
                )
                load_checkpoint(policy, anchor_checkpoint)
                anchors[task] = {
                    "epoch": anchor_epoch,
                    "checkpoint": anchor_checkpoint,
                    "global": extract_stats(
                        policy, probes[task]["global"], args.device, args.batch_size
                    ),
                    "phase": extract_stats(
                        policy, probes[task]["phase"], args.device, args.batch_size
                    ),
                }
                print(
                    f"[anchor] method={method} seed={seed} task={task} "
                    f"epoch={anchor_epoch}", flush=True,
                )
            for stage in args.stages:
                old_tasks = [task for task in args.tasks if task < stage]
                for epoch in args.epochs:
                    checkpoint = checkpoint_path(root, method, seed, stage, epoch)
                    load_checkpoint(policy, checkpoint)
                    for task in old_tasks:
                        current_global = extract_stats(
                            policy, probes[task]["global"], args.device, args.batch_size
                        )
                        current_phase = extract_stats(
                            policy, probes[task]["phase"], args.device, args.batch_size
                        )
                        global_values = derive_values(
                            anchors[task]["global"], current_global,
                            probes[task]["global"], seed, task, args.mc_samples,
                        )
                        phase_values = derive_values(
                            anchors[task]["phase"], current_phase,
                            probes[task]["phase"], seed, task, 0,
                        )
                        row = {
                            "method": method, "seed": seed, "task": task,
                            "anchor_stage": task,
                            "anchor_epoch": anchors[task]["epoch"],
                            "anchor_checkpoint": str(anchors[task]["checkpoint"]),
                            "stage": stage, "epoch": epoch,
                            "checkpoint": str(checkpoint),
                            "global_probe_count": len(probes[task]["global"]["record_ids"]),
                            "phase_probe_count": len(probes[task]["phase"]["record_ids"]),
                            "bc_global_mean": float(global_values["bc"].mean()),
                            "bc_global_q90": quantile(global_values["bc"], 0.9),
                            "rollout_kl_global_mean": float(global_values["kl"].mean()),
                            "rollout_kl_global_q90": quantile(global_values["kl"], 0.9),
                            "rollout_w2_global_mean": float(global_values["w2"].mean()),
                            "rollout_w2_global_q90": quantile(global_values["w2"], 0.9),
                            "rollout_symkl_global_mean": float(global_values["symkl"].mean()),
                            "rollout_mean_l2_global_mean": float(global_values["mean_l2"].mean()),
                            "rollout_mode_switch_rate": float(global_values["mode_switch"].mean()),
                            "full_gmm_fwdkl_global_mean": float(global_values["full_gmm"].mean()),
                            "full_gmm_fwdkl_global_q90": quantile(global_values["full_gmm"], 0.9),
                        }
                        row.update(phase_fields("bc", phase_values["bc"], probes[task]["phase"]["bins"]))
                        row.update(phase_fields("rollout_kl", phase_values["kl"], probes[task]["phase"]["bins"]))
                        row.update(phase_fields("rollout_w2", phase_values["w2"], probes[task]["phase"]["bins"]))
                        if not all(
                            math.isfinite(float(value))
                            for key, value in row.items()
                            if key not in ("method", "anchor_checkpoint", "checkpoint")
                        ):
                            raise FloatingPointError(
                                f"non-finite signal method={method} seed={seed} "
                                f"stage={stage} epoch={epoch} task={task}"
                            )
                        rows.append(row)
                    print(
                        f"[signal-grid] method={method} seed={seed} stage={stage} "
                        f"epoch={epoch} old_tasks={old_tasks} rows={len(rows)}",
                        flush=True,
                    )
            del policy, probes, anchors
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    expected = len(args.methods) * len(args.seeds) * len(args.epochs) * sum(
        sum(task < stage for task in args.tasks) for stage in args.stages
    )
    if len(rows) != expected:
        raise RuntimeError(f"signal grid expected {expected}, found {len(rows)}")
    write_rows(resolve(args.output), rows)
    print(f"[signal-grid] complete rows={len(rows)} output={resolve(args.output)}")


if __name__ == "__main__":
    main()
