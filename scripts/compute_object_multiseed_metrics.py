#!/usr/bin/env python3
import argparse
import csv
import json
import re
import statistics
from pathlib import Path


LINE = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
TASKS = tuple(range(6, 10))
EPOCHS = (70, 75, 80, 85, 90, 95)
METHODS = ("mlr", "mlr_ifa")
PAPER = {
    "mlr": {"fwt": (83.3, 2.6), "nbt": (12.3, 2.4), "auc": (77.6, 3.0)},
    "mlr_ifa": {"fwt": (84.6, 1.9), "nbt": (11.4, 5.6), "auc": (79.4, 1.5)},
}


def parse_log(path):
    records = {}
    for text in Path(path).read_text(encoding="utf-8").splitlines():
        match = LINE.search(text)
        if match:
            task, epoch, stage = map(int, match.group(1, 2, 3))
            records[(stage, epoch, task)] = float(match.group(4)) * 100.0
    selected = {}
    matrix = []
    for stage in TASKS:
        for epoch in EPOCHS:
            if (stage, epoch, stage) not in records:
                raise ValueError(f"missing current-task result {(stage, epoch, stage)} in {path}")
        selected[stage] = max(
            EPOCHS, key=lambda epoch: (records[(stage, epoch, stage)], epoch)
        )
        row = []
        for task in TASKS:
            if task > stage:
                row.append(None)
                continue
            key = (stage, selected[stage], task)
            if key not in records:
                raise ValueError(f"missing selected-checkpoint result {key} in {path}")
            row.append(records[key])
        matrix.append(row)
    return selected, matrix


def metrics(matrix):
    diagonal = [matrix[i][i] for i in range(4)]
    forgetting = []
    auc_tasks = []
    for task_index in range(4):
        trajectory = [matrix[stage][task_index] for stage in range(task_index, 4)]
        auc_tasks.append(statistics.mean(trajectory))
        if task_index < 3:
            forgetting.append(diagonal[task_index] - statistics.mean(trajectory[1:]))
    return {
        "fwt": statistics.mean(diagonal),
        "nbt": statistics.mean(forgetting),
        "auc": statistics.mean(auc_tasks),
    }


def rounded(value):
    return round(value, 3)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/baseline_reconciliation"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args()
    out = args.output_dir
    matrices_dir = out / "matrices"
    out.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)

    results = {method: {} for method in METHODS}
    stage_rows = []
    fwt_rows = []
    forgetting_rows = []

    for seed in args.seeds:
        for method in METHODS:
            log = args.runs_root / f"object_seed{seed}_{method}" / f"eval_seed{seed}.log"
            selected, matrix = parse_log(log)
            score = metrics(matrix)
            results[method][seed] = {
                **{name: rounded(value) for name, value in score.items()},
                "selected_epochs": {str(k): v for k, v in selected.items()},
                "r": [[None if x is None else rounded(x) for x in row] for row in matrix],
            }

            matrix_rows = []
            for stage_index, stage in enumerate(TASKS):
                row = {"stage": stage}
                for task_index, task in enumerate(TASKS):
                    value = matrix[stage_index][task_index]
                    row[f"task{task}"] = "" if value is None else rounded(value)
                    if value is not None:
                        stage_rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "stage": stage,
                                "task": task,
                                "selected_epoch": selected[stage],
                                "success": rounded(value),
                            }
                        )
                matrix_rows.append(row)
            write_csv(
                matrices_dir / f"seed{seed}_{method}.csv",
                ["stage", "task6", "task7", "task8", "task9"],
                matrix_rows,
            )

            for task_index, task in enumerate(TASKS):
                diagonal = matrix[task_index][task_index]
                fwt_rows.append(
                    {"seed": seed, "method": method, "task": task, "fwt": rounded(diagonal)}
                )
                later = [matrix[stage][task_index] for stage in range(task_index + 1, 4)]
                forgetting_rows.append(
                    {
                        "seed": seed,
                        "method": method,
                        "task": task,
                        "diagonal_success": rounded(diagonal),
                        "mean_later_success": "" if not later else rounded(statistics.mean(later)),
                        "forgetting": "" if not later else rounded(diagonal - statistics.mean(later)),
                    }
                )

    write_csv(out / "per_task_fwt.csv", ["seed", "method", "task", "fwt"], fwt_rows)
    write_csv(
        out / "per_task_forgetting.csv",
        ["seed", "method", "task", "diagonal_success", "mean_later_success", "forgetting"],
        forgetting_rows,
    )
    write_csv(
        out / "per_stage_success.csv",
        ["seed", "method", "stage", "task", "selected_epoch", "success"],
        stage_rows,
    )

    summary = {}
    for method in METHODS:
        summary[method] = {}
        for metric in ("fwt", "nbt", "auc"):
            values = [results[method][seed][metric] for seed in args.seeds]
            mean = statistics.mean(values)
            std = statistics.stdev(values)
            paper_mean, paper_std = PAPER[method][metric]
            gap = mean - paper_mean
            summary[method][metric] = {
                "mean": rounded(mean),
                "sample_std": rounded(std),
                "paper_mean": paper_mean,
                "paper_reported_std": paper_std,
                "our_mean_minus_paper_mean": rounded(gap),
                "standardized_descriptive_gap": rounded(gap / paper_std),
            }

    deltas = {}
    for seed in args.seeds:
        deltas[str(seed)] = {
            metric: rounded(results["mlr_ifa"][seed][metric] - results["mlr"][seed][metric])
            for metric in ("fwt", "nbt", "auc")
        }
    deltas["mean"] = {
        metric: rounded(statistics.mean([deltas[str(seed)][metric] for seed in args.seeds]))
        for metric in ("fwt", "nbt", "auc")
    }

    pair_effects = {
        "task7_to_6": {
            str(seed): rounded(
                results["mlr_ifa"][seed]["r"][1][0] - results["mlr"][seed]["r"][1][0]
            )
            for seed in args.seeds
        },
        "task8_to_7": {
            str(seed): rounded(
                results["mlr_ifa"][seed]["r"][2][1] - results["mlr"][seed]["r"][2][1]
            )
            for seed in args.seeds
        },
        "task9_no_pair_diagonal": {
            str(seed): rounded(
                results["mlr_ifa"][seed]["r"][3][3] - results["mlr"][seed]["r"][3][3]
            )
            for seed in args.seeds
        },
    }
    for effect in pair_effects.values():
        effect["mean"] = rounded(statistics.mean([effect[str(seed)] for seed in args.seeds]))

    payload = {
        "benchmark": "LIBERO_OBJECT",
        "seeds": args.seeds,
        "standard_deviation": "sample",
        "per_seed": results,
        "three_seed_summary": summary,
        "mlr_ifa_minus_mlr": deltas,
        "pair_stage_success_differences": pair_effects,
    }
    (out / "object_3seed_metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# LIBERO-OBJECT three-seed baseline reconciliation",
        "",
        "Values are percentages. Standard deviations are sample standard deviations over seeds 42, 43, and 44.",
        "",
        "## Per-seed metrics",
        "",
        "| Method | Seed | FWT | NBT | AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        for seed in args.seeds:
            score = results[method][seed]
            lines.append(f"| {method} | {seed} | {score['fwt']:.3f} | {score['nbt']:.3f} | {score['auc']:.3f} |")
    lines += [
        "",
        "## Three-seed summary and paper comparison",
        "",
        "| Method | Metric | Ours mean ± sample std | Paper | Gap | Standardized descriptive gap |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        for metric in ("fwt", "nbt", "auc"):
            item = summary[method][metric]
            lines.append(
                f"| {method} | {metric.upper()} | {item['mean']:.3f} ± {item['sample_std']:.3f} "
                f"| {item['paper_mean']:.1f} ± {item['paper_reported_std']:.1f} "
                f"| {item['our_mean_minus_paper_mean']:.3f} "
                f"| {item['standardized_descriptive_gap']:.3f} |"
            )
    lines += ["", "## MLR+IFA minus MLR", "", "| Seed | ΔFWT | ΔNBT | ΔAUC |", "| ---: | ---: | ---: | ---: |"]
    for seed in [str(seed) for seed in args.seeds] + ["mean"]:
        item = deltas[seed]
        lines.append(f"| {seed} | {item['fwt']:.3f} | {item['nbt']:.3f} | {item['auc']:.3f} |")
    (out / "object_3seed_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
