#!/usr/bin/env python3
"""Build the frozen, bridge, and interaction reports from formal logs."""

import argparse
import csv
import json
import statistics
from pathlib import Path

from compute_object_metrics import load_matrix, metrics

from libero.lifelong.analysis.ifa_optimization_interaction import (
    canonical_existing_rows,
    decide_h20,
    gamma_materiality,
    interaction_rows,
    mean_std,
    pressure_increased,
    relative_effects,
    spearman,
    write_csv,
)


SEEDS = (42, 43, 44)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def existing(root, out):
    rows = canonical_existing_rows(root)
    effects = relative_effects(rows)
    interactions = interaction_rows(effects, "CONFUNDED_BY_GAMMA_030_031")
    write_csv(out / "existing/canonical_metrics.csv", rows)
    write_csv(out / "existing/existing_relative_effects.csv", effects)
    write_csv(out / "existing/existing_seed_interaction.csv", interactions)
    return rows


def bridge_metrics(root, out):
    rows, matrices, selected = [], {}, {}
    bridge_dir = out / "bridge"
    bridge_dir.mkdir(parents=True, exist_ok=True)
    success_rows, epoch_rows = [], []
    for seed in SEEDS:
        log = root / f"runs/object_seed{seed}_ifa_batchstep_gamma031_bridge/eval_seed{seed}.log"
        matrix, epochs = load_matrix(log)
        score = metrics(matrix)
        matrices[seed], selected[seed] = matrix, epochs
        payload = {
            "benchmark": "LIBERO_OBJECT",
            "method": "ifa_batchstep_gamma031_bridge",
            "gamma": 0.31,
            "seed": seed,
            **score,
            "selected_epochs": epochs,
        }
        (bridge_dir / f"seed{seed}_metrics.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
        rows.append({"seed": seed, "FWT": score["fwt"], "NBT": score["nbt"], "AUC": score["auc"]})
        for stage in range(6, 10):
            epoch_rows.append({"seed": seed, "stage": stage, "selected_epoch": epochs[str(stage)]})
            for task in range(6, stage + 1):
                success_rows.append(
                    {"seed": seed, "stage": stage, "task": task, "success": matrix[stage - 6][task - 6]}
                )
    summary = {
        name.lower(): {
            key: round(value, 3)
            for key, value in mean_std([float(row[name]) for row in rows]).items()
        }
        for name in ("FWT", "NBT", "AUC")
    }
    payload = {
        "benchmark": "LIBERO_OBJECT",
        "method": "ifa_batchstep_gamma031_bridge",
        "gamma": 0.31,
        "seeds": list(SEEDS),
        "per_seed": {str(row["seed"]): row for row in rows},
        "summary": summary,
    }
    (bridge_dir / "three_seed_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(bridge_dir / "selected_epochs.csv", epoch_rows)
    write_csv(bridge_dir / "per_task_success.csv", success_rows)
    return rows, matrices


def source_matrices(root):
    legacy = json.loads(
        (root / "reports/baseline_reconciliation/object_3seed_metrics.json").read_text(encoding="utf-8")
    )["per_seed"]
    batch_mlr = json.loads(
        (root / "reports/paper_reproduction_v2/mlr_formal/three_seed_summary.json").read_text(encoding="utf-8")
    )["per_seed"]
    batch_ifa = json.loads(
        (root / "reports/paper_reproduction_v2/ifa_formal/three_seed_summary.json").read_text(encoding="utf-8")
    )["per_seed"]
    return legacy, batch_mlr, batch_ifa


def summarize_bridge(root, out):
    canonical = canonical_existing_rows(root)
    bridge_rows, bridge_matrix = bridge_metrics(root, out)
    _, batch_mlr, batch_ifa = source_matrices(root)
    bridge_by_seed = {int(row["seed"]): row for row in bridge_rows}

    gamma_rows = []
    for seed in SEEDS:
        old = batch_ifa[str(seed)]
        new = bridge_by_seed[seed]
        gamma_rows.append(
            {
                "seed": seed,
                "gamma_delta_fwt": new["FWT"] - float(old["fwt"]),
                "gamma_delta_nbt": new["NBT"] - float(old["nbt"]),
                "gamma_delta_auc": new["AUC"] - float(old["auc"]),
            }
        )
    gamma_is_material = gamma_materiality(gamma_rows)
    write_csv(out / "bridge/gamma_effects.csv", gamma_rows)

    fixed_rows = [row for row in canonical if row["regime"] == "legacy_epoch"]
    fixed_rows.extend(row for row in canonical if row["regime"] == "batch_step" and row["method"] == "mlr")
    for row in bridge_rows:
        fixed_rows.append(
            {
                "regime": "batch_step",
                "method": "ifa",
                "gamma": "0.31",
                "seed": row["seed"],
                "FWT": row["FWT"],
                "NBT": row["NBT"],
                "AUC": row["AUC"],
                "source_path": f"reports/ifa_optimization_interaction/bridge/seed{row['seed']}_metrics.json",
            }
        )
    fixed_effects = relative_effects(fixed_rows)
    fixed_interactions = interaction_rows(fixed_effects)
    write_csv(out / "interaction/fixed_gamma_relative_effects.csv", fixed_effects)

    interaction_summary = []
    for metric in ("fwt", "nbt", "auc"):
        values = [float(row[f"interaction_{metric}"]) for row in fixed_interactions]
        stats = mean_std(values)
        interaction_summary.append(
            {"metric": metric.upper(), "mean": stats["mean"], "sample_std": stats["sample_std"]}
        )
    write_csv(out / "interaction/fixed_gamma_interaction.csv", fixed_interactions + [
        {
            "seed": "mean",
            "interaction_fwt": interaction_summary[0]["mean"],
            "interaction_nbt": interaction_summary[1]["mean"],
            "interaction_auc": interaction_summary[2]["mean"],
            "interpretation": "fixed_gamma_0.31",
        }
    ])

    stage_current, stage_old, per_task = [], [], []
    legacy, _, _ = source_matrices(root)
    for regime in ("legacy_epoch", "batch_step"):
        for seed in SEEDS:
            if regime == "legacy_epoch":
                mlr_matrix = legacy["mlr"][str(seed)]["r"]
                ifa_matrix = legacy["mlr_ifa"][str(seed)]["r"]
            else:
                mlr_matrix = batch_mlr[str(seed)]["r"]
                ifa_matrix = bridge_matrix[seed]
            for stage in range(6, 10):
                current_delta = ifa_matrix[stage - 6][stage - 6] - mlr_matrix[stage - 6][stage - 6]
                stage_current.append(
                    {"regime": regime, "seed": seed, "stage": stage, "delta_current_success": current_delta}
                )
                for task in range(6, stage):
                    delta = ifa_matrix[stage - 6][task - 6] - mlr_matrix[stage - 6][task - 6]
                    stage_old.append(
                        {"regime": regime, "seed": seed, "stage": stage, "task": task, "delta_old_success": delta}
                    )
                    per_task.append(
                        {"regime": regime, "seed": seed, "stage": stage, "task": task, "kind": "old", "delta_success": delta}
                    )
                per_task.append(
                    {"regime": regime, "seed": seed, "stage": stage, "task": stage, "kind": "current", "delta_success": current_delta}
                )
    write_csv(out / "interaction/stage_current_task_effects.csv", stage_current)
    write_csv(out / "interaction/stage_old_task_effects.csv", stage_old)
    write_csv(out / "interaction/per_task_method_effects.csv", per_task)

    batch_effects = [row for row in fixed_effects if row["regime"] == "batch_step"]
    verdict = decide_h20(fixed_interactions, batch_effects, gamma_is_material)
    flags = ["GAMMA_030_031_MATERIAL" if gamma_is_material else "GAMMA_030_031_NOT_MATERIAL"]
    mean_batch_fwt = statistics.mean(float(row["delta_fwt"]) for row in batch_effects)
    mean_batch_nbt = statistics.mean(float(row["delta_nbt"]) for row in batch_effects)
    mean_batch_auc = statistics.mean(float(row["delta_auc"]) for row in batch_effects)
    if mean_batch_fwt <= -2 and sum(float(row["delta_fwt"]) < 0 for row in batch_effects) >= 2:
        flags.append("IFA_PLASTICITY_COST_AMPLIFIED")
    if mean_batch_nbt >= -1 or sum(float(row["delta_nbt"]) < 0 for row in batch_effects) <= 1:
        flags.append("IFA_RETENTION_GAIN_NOT_ESTABLISHED")
    bridge_auc = statistics.mean(row["AUC"] for row in bridge_rows)
    if abs(bridge_auc - 79.4) <= 2 and mean_batch_auc <= -2:
        flags.append("ABSOLUTE_MATCH_RELATIVE_GAIN_MISMATCH_CONFIRMED")
    if max(row["AUC"] for row in bridge_rows) - min(row["AUC"] for row in bridge_rows) >= 10 or statistics.stdev(row["NBT"] for row in bridge_rows) >= 8:
        flags.append("SEED_HETEROGENEITY_MATERIAL")

    pressure_flag = False
    pressure_seed_path = out / "pressure/ifa_pressure_seed_summary.csv"
    if pressure_seed_path.exists():
        pressure_flag = pressure_increased(read_csv(pressure_seed_path))
        if pressure_flag:
            flags.append("IFA_PRESSURE_INCREASED_UNDER_STRONG_OPTIMIZATION")

    pressure_scatter = []
    raw_path = out / "bridge/raw"
    for seed in SEEDS:
        rows = read_csv(raw_path / f"seed{seed}_ifa_epoch_stats.csv")
        for stage in (7, 8, 9):
            late = [row for row in rows if int(row["stage"]) == stage and int(row["epoch"]) >= 90]
            pressure_scatter.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "late_active_fraction": statistics.mean(float(row["ifa_active_fraction"]) for row in late),
                    "late_ifa_to_bc_ratio": statistics.mean(float(row["ifa_to_bc_ratio"]) for row in late),
                    "late_mean_hinge": statistics.mean(float(row["mean_ifa_hinge"]) for row in late),
                    "current_task_rollout_penalty": bridge_matrix[seed][stage - 6][stage - 6] - batch_mlr[str(seed)]["r"][stage - 6][stage - 6],
                }
            )
    penalties = [row["current_task_rollout_penalty"] for row in pressure_scatter]
    correlations = {
        "spearman_active_vs_penalty": spearman([row["late_active_fraction"] for row in pressure_scatter], penalties),
        "spearman_ratio_vs_penalty": spearman([row["late_ifa_to_bc_ratio"] for row in pressure_scatter], penalties),
        "spearman_hinge_vs_penalty": spearman([row["late_mean_hinge"] for row in pressure_scatter], penalties),
    }
    for row in pressure_scatter:
        row.update(correlations)
    write_csv(out / "interaction/ifa_pressure_vs_fwt_penalty.csv", pressure_scatter)

    gamma_means = {key: statistics.mean(float(row[key]) for row in gamma_rows) for key in gamma_rows[0] if key != "seed"}
    interaction_means = {row["metric"]: row["mean"] for row in interaction_summary}
    legacy_effects = [row for row in fixed_effects if row["regime"] == "legacy_epoch"]
    legacy_means = {key: statistics.mean(float(row[key]) for row in legacy_effects) for key in ("delta_fwt", "delta_nbt", "delta_auc")}
    report = f"""# IFA optimization-regime interaction diagnosis

## Identity

- Starting SHA: `865bae54fe6ed0fde17c6e2f4656dc86270e16ec`
- Branch: `research/ifa-optimization-regime-interaction`
- Scope: LIBERO_OBJECT, seeds 42/43/44.

## Frozen existing results

Legacy MLR is 77.500 FWT / 2.778 NBT / 76.284 AUC; legacy IFA gamma0.31 is 78.333 / 6.852 / 74.861. Batch-step MLR is 90.000 / 13.704 / 82.882; batch-step IFA gamma0.30 is 86.667 / 13.611 / 79.757.

## Why the gamma0.31 bridge was required

The existing scheduler comparison was confounded by gamma 0.31 versus 0.30. The bridge changes only gamma within the batch-step IFA protocol.

## Bridge protocol

Batch 10, batch-step linear schedule with 500 warmup steps, AdamW 1e-4, replay rate 0.5, IFA weight 0.1, gamma 0.31, tasks 6-9, 100 epochs per stage, and 20 rollout trials.

## Bridge 3-seed results

- FWT: {statistics.mean(row['FWT'] for row in bridge_rows):.3f} ± {statistics.stdev(row['FWT'] for row in bridge_rows):.3f}
- NBT: {statistics.mean(row['NBT'] for row in bridge_rows):.3f} ± {statistics.stdev(row['NBT'] for row in bridge_rows):.3f}
- AUC: {bridge_auc:.3f} ± {statistics.stdev(row['AUC'] for row in bridge_rows):.3f}

## Gamma0.30 vs 0.31

Gamma0.31 minus gamma0.30 is {gamma_means['gamma_delta_fwt']:.3f} FWT, {gamma_means['gamma_delta_nbt']:.3f} NBT, and {gamma_means['gamma_delta_auc']:.3f} AUC. Flag: **{flags[0]}**.

## Fixed-gamma IFA effects

Legacy IFA minus MLR: {legacy_means['delta_fwt']:.3f} FWT / {legacy_means['delta_nbt']:.3f} NBT / {legacy_means['delta_auc']:.3f} AUC. Batch-step IFA minus MLR: {mean_batch_fwt:.3f} / {mean_batch_nbt:.3f} / {mean_batch_auc:.3f}.

## Difference-in-differences

Batch-step minus legacy relative IFA effect: {interaction_means['FWT']:.3f} FWT / {interaction_means['NBT']:.3f} NBT / {interaction_means['AUC']:.3f} AUC.

## IFA pressure

Offline pressure is reported in `pressure/`; increased-under-strong-optimization flag is {pressure_flag}. Pressure-versus-rollout correlations use only nine seed-stage points and are descriptive.

## Stage/task decomposition

The complete current-task and old-task deltas are in `interaction/stage_current_task_effects.csv` and `interaction/stage_old_task_effects.csv`; no retention tradeoff is inferred without corresponding old-task improvement.

## Observed

The formal bridge isolates gamma 0.31 under the batch-step optimization regime. Flags: {', '.join(flags) if flags else 'NONE'}.

## Inferred

The primary rule-based verdict is **{verdict}**. This verdict concerns optimization-regime dependence in the released implementation.

## Not established

This experiment does not recover the unpublished exact internal code version used to produce the paper table.

A scheduler interaction does not prove IFA is intrinsically ineffective; it establishes optimization-regime dependence in this released implementation.

The offline pressure probe is diagnostic, not a causal substitute for the formal bridge runs.

Only LIBERO_OBJECT and seeds42/43/44 are studied.

## Verdict

**{verdict}**
"""
    (out / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({"verdict": verdict, "flags": flags, "bridge": bridge_rows}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("existing", "bridge"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "reports/ifa_optimization_interaction"
    existing(root, out)
    if args.mode == "bridge":
        summarize_bridge(root, out)


if __name__ == "__main__":
    main()
