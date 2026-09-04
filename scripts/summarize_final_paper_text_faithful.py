#!/usr/bin/env python3
"""Summarize the frozen final paper-text-faithful evaluation protocol."""

import csv
import importlib.metadata
import json
import platform
import re
import statistics
import subprocess
from pathlib import Path

from compute_object_metrics import LINE, load_matrix, metrics


SEEDS = (42, 43, 44)
METHODS = ("mlr", "ifa")
METRICS = ("fwt", "nbt", "auc")
SECONDARY_EPOCHS = (70, 75, 80, 85, 90, 95, 99)
PAPER = {
    "mlr": {"fwt": (83.3, 2.6), "nbt": (12.3, 2.4), "auc": (77.6, 3.0)},
    "ifa": {"fwt": (84.6, 1.9), "nbt": (11.4, 5.6), "auc": (79.4, 1.5)},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fixed_epoch_matrix(path, epoch=99):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = LINE.search(line)
        if match:
            task, checkpoint, stage = map(int, match.group(1, 2, 3))
            if checkpoint == epoch and 6 <= task <= stage <= 9:
                records[(stage, task)] = float(match.group(4)) * 100.0
    expected = {(stage, task) for stage in range(6, 10) for task in range(6, stage + 1)}
    missing = sorted(expected - records.keys())
    if missing:
        raise ValueError(f"missing primary ep{epoch} entries in {path}: {missing}")
    matrix = [[None] * 4 for _ in range(4)]
    for stage, task in expected:
        matrix[stage - 6][task - 6] = records[(stage, task)]
    return matrix, {str(stage): epoch for stage in range(6, 10)}


def sample_summary(per_seed):
    result = {"per_seed": per_seed, "sample_standard_deviation": True}
    for name in METRICS:
        values = [float(per_seed[str(seed)][name]) for seed in SEEDS]
        result[name] = {
            "mean": round(statistics.mean(values), 3),
            "std": round(statistics.stdev(values), 3),
        }
    return result


def read_view(root, method, seed, view):
    run = root / "runs/final_paper_text_faithful" / method / f"seed{seed}"
    if view == "primary":
        matrix, selected = fixed_epoch_matrix(run / "primary_ep99.log")
        label = "PRIMARY_EP99"
    else:
        matrix, selected = load_matrix(run / "secondary_selected.log", epochs=SECONDARY_EPOCHS)
        label = "SECONDARY_RELEASE_COMPATIBLE_SELECTION"
    return {
        "benchmark": "LIBERO_OBJECT",
        "method": method,
        "seed": seed,
        "evaluation_view": label,
        "n_eval": 20,
        "selected_epochs": selected,
        **metrics(matrix),
    }


def environment_snapshot():
    packages = {}
    for name in ("torch", "mujoco", "robosuite", "transformers", "clip"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-distribution-metadata"
    try:
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
        ).strip().splitlines()
    except (OSError, subprocess.CalledProcessError):
        gpu = ["unavailable"]
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages, "gpu": gpu}


def metric_line(summary):
    return " / ".join(
        f"{name.upper()} {summary[name]['mean']:.3f} ± {summary[name]['std']:.3f}"
        for name in METRICS
    )


def main():
    root = Path(__file__).resolve().parents[1]
    report_root = root / "reports/final_paper_text_faithful"
    primary_dir = report_root / "primary_ep99"
    secondary_dir = report_root / "secondary_selected"
    comparison_dir = report_root / "comparison"
    for directory in (primary_dir, secondary_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    all_results = {"primary": {}, "secondary": {}}
    for view, out_dir in (("primary", primary_dir), ("secondary", secondary_dir)):
        for method in METHODS:
            per_seed = {}
            for seed in SEEDS:
                result = read_view(root, method, seed, view)
                per_seed[str(seed)] = result
                if view == "primary":
                    write_json(out_dir / f"{method}_seed{seed}.json", result)
            summary = sample_summary(per_seed)
            summary.update({"benchmark": "LIBERO_OBJECT", "method": method, "evaluation_view": per_seed["42"]["evaluation_view"]})
            write_json(out_dir / f"{method}_three_seed_summary.json", summary)
            all_results[view][method] = summary

    paired_by_view = {}
    for view, out_dir in (("primary", primary_dir), ("secondary", secondary_dir)):
        rows = []
        for seed in SEEDS:
            mlr = all_results[view]["mlr"]["per_seed"][str(seed)]
            ifa = all_results[view]["ifa"]["per_seed"][str(seed)]
            rows.append({
                "view": mlr["evaluation_view"],
                "seed": seed,
                "delta_fwt": round(ifa["fwt"] - mlr["fwt"], 3),
                "delta_nbt": round(ifa["nbt"] - mlr["nbt"], 3),
                "delta_auc": round(ifa["auc"] - mlr["auc"], 3),
            })
        mean_row = {"view": rows[0]["view"], "seed": "mean"}
        for name in METRICS:
            mean_row[f"delta_{name}"] = round(statistics.mean(float(row[f"delta_{name}"]) for row in rows), 3)
        rows.append(mean_row)
        paired_by_view[view] = rows
        write_csv(out_dir / "paired_method_effects.csv", ["view", "seed", "delta_fwt", "delta_nbt", "delta_auc"], rows)

    selected_rows = []
    for method in METHODS:
        for seed in SEEDS:
            selected = all_results["secondary"][method]["per_seed"][str(seed)]["selected_epochs"]
            for stage in range(6, 10):
                selected_rows.append({"method": method, "seed": seed, "stage": stage, "selected_epoch": selected[str(stage)]})
    write_csv(secondary_dir / "selected_epochs.csv", ["method", "seed", "stage", "selected_epoch"], selected_rows)

    matrix_rows = []
    current_rows = []
    old_rows = []
    for method in METHODS:
        for seed in SEEDS:
            result = all_results["primary"][method]["per_seed"][str(seed)]
            matrix = result["r"]
            for stage in range(6, 10):
                for task in range(6, stage + 1):
                    matrix_rows.append({"method": method, "seed": seed, "stage": stage, "task": task, "success": matrix[stage - 6][task - 6]})
                current_rows.append({"method": method, "seed": seed, "stage": stage, "task": stage, "success": matrix[stage - 6][stage - 6]})
                old = [matrix[stage - 6][task - 6] for task in range(6, stage)]
                old_rows.append({
                    "method": method,
                    "seed": seed,
                    "stage": stage,
                    "old_task_count": len(old),
                    "mean_old_success": "" if not old else round(statistics.mean(old), 3),
                })
    write_csv(primary_dir / "per_task_success.csv", ["method", "seed", "stage", "task", "success"], matrix_rows)
    write_csv(primary_dir / "per_stage_current_success.csv", ["method", "seed", "stage", "task", "success"], current_rows)
    write_csv(primary_dir / "per_stage_old_success.csv", ["method", "seed", "stage", "old_task_count", "mean_old_success"], old_rows)

    for view in ("primary", "secondary"):
        rows = []
        for method in METHODS:
            for name in METRICS:
                ours = all_results[view][method][name]
                paper_mean, paper_std = PAPER[method][name]
                rows.append({
                    "view": view,
                    "method": method,
                    "metric": name.upper(),
                    "our_mean": ours["mean"],
                    "our_sample_std": ours["std"],
                    "paper_mean": paper_mean,
                    "paper_reported_std": paper_std,
                    "signed_gap_ours_minus_paper": round(ours["mean"] - paper_mean, 3),
                })
        write_csv(comparison_dir / f"paper_vs_{view}.csv", list(rows[0]), rows)

    sensitivity_rows = []
    material = False
    for method in METHODS:
        for name in METRICS:
            primary_mean = all_results["primary"][method][name]["mean"]
            secondary_mean = all_results["secondary"][method][name]["mean"]
            difference = round(secondary_mean - primary_mean, 3)
            sensitivity_rows.append({
                "method": method,
                "metric": name.upper(),
                "primary_mean": primary_mean,
                "secondary_mean": secondary_mean,
                "secondary_minus_primary": difference,
            })
            if name in ("fwt", "auc") and abs(difference) >= 2.0:
                material = True
    write_csv(comparison_dir / "primary_vs_secondary.csv", list(sensitivity_rows[0]), sensitivity_rows)
    checkpoint_flag = "CHECKPOINT_SELECTION_MATERIAL" if material else "CHECKPOINT_SELECTION_NOT_MATERIAL"

    collapse_items = []
    for method in METHODS:
        for task in range(6, 10):
            values = [
                all_results["primary"][method]["per_seed"][str(seed)]["r"][task - 6][task - 6]
                for seed in SEEDS
            ]
            if sum(value <= 10.0 for value in values) >= 2:
                collapse_items.append({"method": method, "task": task, "values": values})

    primary_delta = paired_by_view["primary"][-1]
    negative_auc_seeds = sum(float(row["delta_auc"]) < 0 for row in paired_by_view["primary"][:-1])
    if (
        float(primary_delta["delta_fwt"]) >= 0
        and float(primary_delta["delta_nbt"]) <= 0
        and float(primary_delta["delta_auc"]) > 0
        and not collapse_items
    ):
        verdict = "PAPER_TEXT_FAITHFUL_METHOD_LEVEL_REPRODUCTION_SUPPORTED"
    elif float(primary_delta["delta_auc"]) < 0 and negative_auc_seeds >= 2:
        verdict = "PAPER_TEXT_FAITHFUL_METHOD_LEVEL_REPRODUCTION_NOT_SUPPORTED"
    else:
        verdict = "PAPER_TEXT_FAITHFUL_METHOD_LEVEL_REPRODUCTION_PARTIAL"

    stage_differences = []
    for stage in range(6, 10):
        mlr_values = [all_results["primary"]["mlr"]["per_seed"][str(seed)]["r"][stage - 6][stage - 6] for seed in SEEDS]
        ifa_values = [all_results["primary"]["ifa"]["per_seed"][str(seed)]["r"][stage - 6][stage - 6] for seed in SEEDS]
        stage_differences.append((stage, statistics.mean(ifa_values) - statistics.mean(mlr_values), statistics.mean(mlr_values), statistics.mean(ifa_values)))
    largest_stage = max(stage_differences, key=lambda item: abs(item[1]))

    env = environment_snapshot()
    primary_signs = {
        name: ["+" if float(row[f"delta_{name}"]) > 0 else "-" if float(row[f"delta_{name}"]) < 0 else "0" for row in paired_by_view["primary"][:-1]]
        for name in METRICS
    }
    secondary_delta = paired_by_view["secondary"][-1]
    primary_seed_rows = "\n".join(
        f"| {method.upper()} | {seed} | {all_results['primary'][method]['per_seed'][str(seed)]['fwt']:.3f} | "
        f"{all_results['primary'][method]['per_seed'][str(seed)]['nbt']:.3f} | "
        f"{all_results['primary'][method]['per_seed'][str(seed)]['auc']:.3f} |"
        for method in METHODS for seed in SEEDS
    )
    secondary_seed_rows = "\n".join(
        f"| {method.upper()} | {seed} | {all_results['secondary'][method]['per_seed'][str(seed)]['fwt']:.3f} | "
        f"{all_results['secondary'][method]['per_seed'][str(seed)]['nbt']:.3f} | "
        f"{all_results['secondary'][method]['per_seed'][str(seed)]['auc']:.3f} | "
        f"{list(all_results['secondary'][method]['per_seed'][str(seed)]['selected_epochs'].values())} |"
        for method in METHODS for seed in SEEDS
    )
    gap_lines = "\n".join(
        f"- {method.upper()} {name.upper()}: {all_results['primary'][method][name]['mean'] - PAPER[method][name][0]:+.3f} pp"
        for method in METHODS for name in METRICS
    )
    report = f"""# Final paper-text-faithful reproduction

## Verdict

`{verdict}`

Flags: `{checkpoint_flag}`; structural collapse: `{bool(collapse_items)}`.

Provenance verdict: `PAPER_EXACT_RECIPE_NOT_PUBLICLY_RECOVERABLE`.

## Paper-explicit protocol

LIBERO-OBJECT; batch 10; 100 epochs per lifelong stage; AdamW at initial LR 1e-4; linear scheduler; replay equivalent to five demonstrations per task; IFA lambda 0.1 and alpha/gamma 0.30; seeds 42/43/44; 20 evaluation trials.

## Paper-unspecified frozen choices

Scheduler advancement after every optimizer step (**PAPER DOES NOT SPECIFY SCHEDULER STEP UNIT**); warmup 500 (**RELEASE_FALLBACK**); AdamW betas [0.9, 0.999], eps 1e-8 and weight decay 0.1 (**RELEASE_FALLBACK**); existing seed-specific augmented feature caches (**FEATURE_AUGMENTATION = RELEASE_FALLBACK; NOT PAPER-SPECIFIED**); release GMM stochastic action sampling (**RELEASE_IMPLEMENTATION_FALLBACK**); ep99 primary checkpoint; release-compatible selected-checkpoint secondary view.

## Pretraining limitation

This is a **LIFELONG-STAGE PAPER-TEXT-FAITHFUL REPRODUCTION WITH RELEASED/FROZEN PRETRAINED INITIALIZATION**. Exact paper-time pretraining code is not public; every run uses the same frozen `object_multitask_model_ep99.pth`. This is not a full end-to-end exact-paper reproduction.

## Environment

- Python: {env['python']}
- GPU: {', '.join(env['gpu'])}
- PyTorch: {env['packages']['torch']}; MuJoCo: {env['packages']['mujoco']}; robosuite: {env['packages']['robosuite']}; transformers: {env['packages']['transformers']}; CLIP: {env['packages']['clip']}

## Primary ep99 results

- MLR: {metric_line(all_results['primary']['mlr'])}
- MLR+IFA alpha0.30: {metric_line(all_results['primary']['ifa'])}
- Mean IFA - MLR: ΔFWT {float(primary_delta['delta_fwt']):+.3f}; ΔNBT {float(primary_delta['delta_nbt']):+.3f}; ΔAUC {float(primary_delta['delta_auc']):+.3f} pp.
- Per-seed signs (42/43/44): FWT {primary_signs['fwt']}; NBT {primary_signs['nbt']}; AUC {primary_signs['auc']}.

| Method | Seed | FWT | NBT | AUC |
| --- | ---: | ---: | ---: | ---: |
{primary_seed_rows}

## Secondary checkpoint sensitivity

- MLR: {metric_line(all_results['secondary']['mlr'])}
- MLR+IFA: {metric_line(all_results['secondary']['ifa'])}
- Mean IFA - MLR: ΔFWT {float(secondary_delta['delta_fwt']):+.3f}; ΔNBT {float(secondary_delta['delta_nbt']):+.3f}; ΔAUC {float(secondary_delta['delta_auc']):+.3f} pp.
- Flag: `{checkpoint_flag}`. The flag uses the frozen rule `|secondary-primary FWT| >= 2pp OR |secondary-primary AUC| >= 2pp` for either method.
- Selected epochs are in `secondary_selected/selected_epochs.csv`.

| Method | Seed | FWT | NBT | AUC | Selected epochs for stages 6/7/8/9 |
| --- | ---: | ---: | ---: | ---: | --- |
{secondary_seed_rows}

## Paper comparison

Paper MLR: FWT 83.3±2.6 / NBT 12.3±2.4 / AUC 77.6±3.0. Paper IFA: FWT 84.6±1.9 / NBT 11.4±5.6 / AUC 79.4±1.5. Exact signed gaps and local sample standard deviations are in `comparison/paper_vs_primary.csv` and `comparison/paper_vs_secondary.csv`; performance above the paper mean is not treated as failure.

Primary signed gaps (ours minus paper):

{gap_lines}

## Per-stage interpretation

Largest primary current-task method difference is stage/task {largest_stage[0]}: MLR mean {largest_stage[2]:.3f}, IFA mean {largest_stage[3]:.3f}, IFA-MLR {largest_stage[1]:+.3f} pp. Structural collapse rule: same current task at or below 10 pp in at least two of three seeds for either method. Detected items: `{json.dumps(collapse_items)}`.

## Final interpretation

What reproduced: the frozen paper-explicit lifelong configuration, paired three-seed MLR/IFA execution, primary final-state evaluation, secondary checkpoint sensitivity, and method-level comparison. What remains unrecoverable: the exact paper-time pretraining implementation and all paper-unspecified choices listed above. These results are the final paper-text-faithful local baseline; no further batch/gamma/scheduler/augmentation/checkpoint search will be used to chase Table 1.

Branch: `research/final-paper-text-faithful-reproduction`. Starting SHA: `1f4e45d604be7f2b61b356e70ac4fa7c80557fe0`. The exact final commit SHA is recorded in the final Git handoff because a commit cannot contain its own hash.

## Exact successful commands

```bash
bash scripts/run_final_paper_text_faithful.sh protocol
bash scripts/run_final_paper_text_faithful.sh test
bash scripts/run_final_paper_text_faithful.sh smoke
bash scripts/run_final_paper_text_faithful.sh formal
```
"""
    (report_root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
    write_json(report_root / "SUMMARY.json", {
        "verdict": verdict,
        "flags": [checkpoint_flag],
        "provenance_verdict": "PAPER_EXACT_RECIPE_NOT_PUBLICLY_RECOVERABLE",
        "primary": all_results["primary"],
        "secondary": all_results["secondary"],
        "primary_paired_effect": primary_delta,
        "secondary_paired_effect": secondary_delta,
        "structural_collapse": collapse_items,
    })
    print(verdict)
    print(checkpoint_flag)


if __name__ == "__main__":
    main()
