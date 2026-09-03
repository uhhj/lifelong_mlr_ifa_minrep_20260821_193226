#!/usr/bin/env python3
"""Read-only IFA pressure probes on frozen MLR checkpoints."""

import argparse
import csv
import json
import random
import re
import statistics
from collections import OrderedDict
from pathlib import Path

import torch

from libero.lifelong.analysis.ifa_optimization_interaction import write_csv
from libero.lifelong.models import get_policy_class


SEEDS = (42, 43, 44)
STAGES = (7, 8, 9)
EPOCHS = (70, 75, 80, 85, 90, 95)
PROBE_SIZE = 256


def parse_pairs(report_path):
    pairs = {}
    pattern = re.compile(r"^\|\s*(42|43|44)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$")
    for line in Path(report_path).read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        seed = int(match.group(1))
        for stage, cell in zip(STAGES, match.groups()[1:]):
            old = re.search(r"task(\d+)", cell)
            pairs[(seed, stage)] = int(old.group(1)) if old else None
    if len(pairs) != 9:
        raise ValueError(f"could not recover all nine seed-stage pair selections from {report_path}")
    return pairs


def build_manifest(root, output):
    rows = []
    for seed in SEEDS:
        for task in STAGES:
            paths = sorted((root / f"features/object_seed{seed}/{task}").glob("*.pt"))
            if len(paths) < PROBE_SIZE:
                raise ValueError(f"task probe is too small: seed={seed} task={task} count={len(paths)}")
            rng = random.Random(20260903 + seed * 1000 + task * 10)
            chosen = rng.sample(paths, PROBE_SIZE)
            for index, path in enumerate(chosen):
                rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "probe_index": index,
                        "sample_seed": 20260903 + seed * 1000 + task * 10,
                        "feature_path": str(path.relative_to(root)),
                    }
                )
    if len(rows) != 2304:
        raise AssertionError(f"expected 2304 manifest rows, got {len(rows)}")
    write_csv(output, rows)
    return rows


def checkpoint_dir(root, regime, seed):
    if regime == "legacy_epoch":
        return root / f"runs/object_seed{seed}_mlr/run_001"
    if seed == 42:
        return root / "runs/object_seed42_mlr_paper_candidate/run_001"
    return root / f"runs/object_seed{seed}_mlr_paper_v2/run_001"


def load_policy(checkpoint_path, device, root):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    cfg = checkpoint["cfg"]
    cfg.device = str(device)
    cfg.pretrain = str(root / "checkpoints/object_multitask_model_ep99.pth")
    policy = get_policy_class(cfg.policy.policy_type)(cfg, cfg.shape_meta)
    state = checkpoint["state_dict"]
    if state and next(iter(state)).startswith("module."):
        state = OrderedDict((name[7:], value) for name, value in state.items())
    policy.load_state_dict(state, strict=False)
    # BriefPolicy.forward returns training outputs only in train mode; this also
    # matches the mode in which the released IFA term is evaluated.
    return policy.to(device).train()


def load_samples(root, manifest_rows):
    return [torch.load(root / row["feature_path"], map_location="cpu") for row in manifest_rows]


def normalized(vector):
    return vector / (vector.norm(p=2, dim=-1, keepdim=True) + 1e-8)


def probe_checkpoint(policy, samples, old_reference, gamma_values, device, batch_size=32):
    bc_sum, count = 0.0, 0
    theta_pos_values, theta_neg_values = [], []
    current_reference = samples[0]["feature"][0, 0, 0, :].to(device).float()
    current_reference = normalized(current_reference.unsqueeze(0)).squeeze(0)
    if old_reference is not None:
        old_reference = normalized(old_reference.to(device).float().unsqueeze(0)).squeeze(0)
        phi = torch.acos((current_reference * old_reference).sum().clamp(-1 + 1e-6, 1 - 1e-6))
    else:
        phi = None
    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch = samples[start : start + batch_size]
            feature = torch.stack([item["feature"] for item in batch]).to(device).float()
            actions = torch.stack([item["actions"] for item in batch]).to(device).float()
            bc_loss, _ = policy.compute_loss({"feature": feature, "actions": actions}, None, None)
            bs = len(batch)
            bc_sum += float(bc_loss.item()) * bs
            count += bs
            if old_reference is not None:
                temporal = policy.temporal_encode(feature.squeeze(1))
                last_feat = normalized(temporal[:, -1, :])
                true_ref = current_reference.unsqueeze(0).expand_as(last_feat)
                false_ref = old_reference.unsqueeze(0).expand_as(last_feat)
                cos_pos = (last_feat * true_ref).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
                cos_neg = (last_feat * false_ref).sum(-1).clamp(-1 + 1e-6, 1 - 1e-6)
                theta_pos_values.append(torch.acos(cos_pos).cpu())
                theta_neg_values.append(torch.acos(cos_neg).cpu())
    bc_mean = bc_sum / count
    if old_reference is None:
        return {"bc_mean": bc_mean, "phi": None}
    theta_pos = torch.cat(theta_pos_values)
    theta_neg = torch.cat(theta_neg_values)
    output = {
        "bc_mean": bc_mean,
        "theta_pos_mean": float(theta_pos.mean()),
        "theta_neg_mean": float(theta_neg.mean()),
        "phi": float(phi.cpu()),
    }
    for gamma in gamma_values:
        raw = theta_pos - theta_neg + gamma * phi.cpu()
        hinge = torch.relu(raw)
        suffix = f"{gamma:.2f}".replace(".", "")
        output[f"hinge{suffix}_mean"] = float(hinge.mean())
        output[f"hinge{suffix}_active_rate"] = float((hinge > 0).float().mean())
        output[f"lambda_hinge{suffix}_over_bc"] = 0.1 * float(hinge.mean()) / max(abs(bc_mean), 1e-12)
        if gamma == 0.31:
            output["margin_raw_mean"] = float(raw.mean())
            output["margin_q50"] = float(torch.quantile(raw, 0.50))
            output["margin_q90"] = float(torch.quantile(raw, 0.90))
    return output


def aggregate(rows, keys):
    groups = {}
    for row in rows:
        if row["pair_recovery_status"] == "no_selected_pair":
            continue
        key = tuple(row[name] for name in keys)
        groups.setdefault(key, []).append(row)
    output = []
    for key, members in sorted(groups.items()):
        item = dict(zip(keys, key))
        for name in ("bc_mean", "hinge030_mean", "hinge031_mean", "hinge030_active_rate", "hinge031_active_rate", "lambda_hinge030_over_bc", "lambda_hinge031_over_bc"):
            item[name] = statistics.mean(float(row[name]) for row in members)
        item["checkpoint_count"] = len(members)
        output.append(item)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "reports/ifa_optimization_interaction"
    manifest_path = out / "probes/current_task_probe_manifest.csv"
    manifest = build_manifest(root, manifest_path)
    manifest_groups = {}
    for row in manifest:
        manifest_groups.setdefault((int(row["seed"]), int(row["task"])), []).append(row)
    pair_source = root / "reports/baseline_reconciliation/protocol_audit.md"
    pairs = parse_pairs(pair_source)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for regime in ("legacy_epoch", "batch_step"):
        for seed in SEEDS:
            for stage in STAGES:
                samples = load_samples(root, manifest_groups[(seed, stage)])
                old_task = pairs[(seed, stage)]
                old_reference = None
                if old_task is not None:
                    reference = torch.load(
                        root / f"features/object_seed{seed}/{old_task}/0.pt", map_location="cpu"
                    )
                    old_reference = reference["feature"][0, 0, 0, :]
                for epoch in EPOCHS:
                    checkpoint = checkpoint_dir(root, regime, seed) / f"mlrifa_{stage}_model_ep{epoch}.pth"
                    if not checkpoint.exists():
                        raise FileNotFoundError(checkpoint)
                    policy = load_policy(checkpoint, device, root)
                    probe_seed = 20260903 + seed * 1000 + stage * 10
                    torch.manual_seed(probe_seed)
                    if device.type == "cuda":
                        torch.cuda.manual_seed_all(probe_seed)
                    values = probe_checkpoint(policy, samples, old_reference, (0.30, 0.31), device)
                    row = {
                        "regime": regime,
                        "seed": seed,
                        "stage": stage,
                        "epoch": epoch,
                        "old_task": "" if old_task is None else old_task,
                        "pair_recovery_status": "no_selected_pair" if old_task is None else "recovered_from_per_stage_report",
                        "pair_source": str(pair_source.relative_to(root)),
                        **values,
                    }
                    if old_task is None:
                        for name in ("hinge030_mean", "hinge031_mean", "hinge030_active_rate", "hinge031_active_rate", "lambda_hinge030_over_bc", "lambda_hinge031_over_bc", "theta_pos_mean", "theta_neg_mean", "margin_raw_mean", "margin_q50", "margin_q90"):
                            row[name] = ""
                    rows.append(row)
                    del policy
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    if len(rows) != 108:
        raise AssertionError(f"expected 108 checkpoint probes, got {len(rows)}")
    write_csv(out / "pressure/ifa_pressure_checkpoint_grid.csv", rows)
    regime_summary = aggregate(rows, ("regime",))
    stage_summary = aggregate(rows, ("regime", "stage"))
    write_csv(out / "pressure/ifa_pressure_regime_summary.csv", regime_summary)
    write_csv(out / "pressure/ifa_pressure_stage_summary.csv", stage_summary)

    seed_rows = []
    for seed in SEEDS:
        item = {"seed": seed}
        for regime, prefix in (("legacy_epoch", "legacy"), ("batch_step", "batch")):
            members = [row for row in rows if row["seed"] == seed and row["regime"] == regime and row["pair_recovery_status"] != "no_selected_pair"]
            item[f"{prefix}_ratio_median"] = statistics.median(float(row["lambda_hinge031_over_bc"]) for row in members)
            item[f"{prefix}_active_rate"] = statistics.mean(float(row["hinge031_active_rate"]) for row in members)
        seed_rows.append(item)
    write_csv(out / "pressure/ifa_pressure_seed_summary.csv", seed_rows)
    print(json.dumps({"checkpoint_rows": len(rows), "manifest_rows": len(manifest), "seed_summary": seed_rows}, indent=2))


if __name__ == "__main__":
    main()
