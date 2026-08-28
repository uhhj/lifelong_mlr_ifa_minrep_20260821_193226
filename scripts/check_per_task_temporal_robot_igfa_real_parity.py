#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch

from libero.lifelong.algos.mlr_per_task_temporal_robot_igfa import (
    per_task_temporal_head_corrections,
)
from libero.lifelong.algos.mlr_temporal_robot_igfa import temporal_head_corrections
from libero.lifelong.analysis.per_task_temporal_interference import (
    build_stage_records,
    deterministic_diagnostic_batches,
    gradient_parity,
    load_record_batch,
    load_temporal_bin_map,
)
from scripts.diagnose_per_task_temporal_interference import load_config, load_policy


def forward_loss(policy, records):
    device = next(policy.parameters()).device
    features, actions, task_ids, progress_bins = load_record_batch(records, device)
    temporal = policy.temporal_encode(features)
    distribution = policy.policy_head(temporal[:, -1, :], train=True)
    losses = policy.policy_head.loss_fn(distribution, actions, reduction="none")
    if not torch.isfinite(losses).all():
        raise FloatingPointError("non-finite real-batch parity loss")
    return losses, task_ids, progress_bins


def gradients_or_zero(loss, parameters):
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/per_task_temporal_robot_igfa_v2/real_parity.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    run = root / "runs" / f"object_seed{args.seed}_mlr_temporal_igfa" / "run_001"
    cfg = load_config(run / "config.json", root, args.seed, args.device)
    policy = load_policy(cfg, run / "mlrifa_6_model_ep95.pth", args.device)
    sidecar = load_temporal_bin_map(
        root / "metadata" / "temporal_bins" / f"object_seed{args.seed}.json",
        expected_seed=args.seed,
    )
    records, _ = build_stage_records(
        root / "features" / f"object_seed{args.seed}", args.seed, 7, sidecar,
        rate=0.5,
    )
    batches = deterministic_diagnostic_batches(records, 10, 100, 20260828)
    selected = None
    for batch in batches:
        tasks = {int(item["task_id"]) for item in batch}
        bins = {(int(item["task_id"]), int(item["progress_bin"])) for item in batch}
        if 7 in tasks and 6 in tasks and any(
            (7, bin_id) in bins and (6, bin_id) in bins for bin_id in range(5)
        ):
            selected = batch
            break
    if selected is None:
        raise RuntimeError("no real stage7 mixed batch with a same-bin anchor")

    head = [parameter for parameter in policy.policy_head.parameters() if parameter.requires_grad]
    transformer = [
        parameter for parameter in policy.temporal_transformer.parameters()
        if parameter.requires_grad
    ]

    losses, task_ids, progress_bins = forward_loss(policy, selected)
    baseline = gradients_or_zero(losses.mean(), head + transformer)
    baseline_head = baseline[:len(head)]
    baseline_transformer = baseline[len(head):]

    losses, task_ids, progress_bins = forward_loss(policy, selected)
    gate_zero_delta, _ = per_task_temporal_head_corrections(
        losses, task_ids, progress_bins, 7, head, gate_strength=0.0, n_bins=5,
    )
    full = gradients_or_zero(losses.mean(), head + transformer)
    gate_zero_head = tuple(
        gradient + delta for gradient, delta in zip(full[:len(head)], gate_zero_delta)
    )
    gate_zero_transformer = full[len(head):]
    head_abs, head_relative = gradient_parity(gate_zero_head, baseline_head)
    transformer_abs, transformer_relative = gradient_parity(
        gate_zero_transformer, baseline_transformer
    )

    losses, task_ids, progress_bins = forward_loss(policy, selected)
    v1_delta, v1_results = temporal_head_corrections(
        losses, task_ids, progress_bins, 7, head, gate_strength=1.0, n_bins=5,
    )
    v2_delta, v2_results = per_task_temporal_head_corrections(
        losses, task_ids, progress_bins, 7, head, gate_strength=1.0, n_bins=5,
    )
    single_abs, single_relative = gradient_parity(v2_delta, v1_delta)
    if not v1_results or not v2_results:
        raise AssertionError("selected real batch did not produce same-bin corrections")

    payload = {
        "seed": args.seed,
        "checkpoint": str(run / "mlrifa_6_model_ep95.pth"),
        "batch_size": len(selected),
        "task_counts": {
            str(task): sum(int(item["task_id"]) == task for item in selected)
            for task in (6, 7)
        },
        "gate_zero": {
            "head_max_abs": head_abs,
            "head_relative_l2": head_relative,
            "transformer_max_abs": transformer_abs,
            "transformer_relative_l2": transformer_relative,
        },
        "single_old_v1_parity": {
            "max_abs": single_abs,
            "relative_l2": single_relative,
            "candidate_bins": len(v2_results),
        },
    }
    for section in ("gate_zero", "single_old_v1_parity"):
        values = payload[section]
        for key, value in values.items():
            if key.endswith("max_abs") and value > 5e-6:
                raise AssertionError(f"{section} {key}={value} exceeds 5e-6")
            if key.endswith("relative_l2") and value > 1e-6:
                raise AssertionError(f"{section} {key}={value} exceeds 1e-6")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
