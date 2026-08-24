#!/usr/bin/env python3
import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


GROUPS = {
    "all_trainable": "cos_all",
    "temporal_transformer": "cos_transformer",
    "policy_head": "cos_head",
}
SUMMARY_FIELDS = [
    "method", "seed", "checkpoint_stage", "old_task", "new_task", "group",
    "task_level_cosine", "diag_bin0", "diag_bin1", "diag_bin2", "diag_bin3",
    "diag_bin4", "diag_mean", "diag_min", "diag_max", "diag_range",
    "positive_bin_fraction", "negative_bin_fraction", "near_zero_bin_fraction",
    "mixed_sign",
]
THREE_FIELDS = [
    "method", "old_task", "new_task", "group", "task_level_mean",
    "task_level_sample_std", "diag_mean", "diag_mean_sample_std", "diag_range_mean",
    "mixed_sign_seed_count", "large_range_seed_count", "replicated",
]


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value):
    return str(value).lower() == "true"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw_files = sorted(args.input_dir.glob("*_cosines.csv"))
    if not raw_files:
        raise FileNotFoundError(f"no cosine CSVs in {args.input_dir}")
    raw = [row for path in raw_files for row in read_csv(path)]
    task_rows = read_csv(args.output_dir / "task_level_pairs.csv")
    task_lookup = {
        (row["method"], row["seed"], row["old_task"], row["new_task"]): row
        for row in task_rows
    }
    pair_rows = []
    matrix_lookup = {}
    keys = sorted({
        (row["method"], row["seed"], row["checkpoint_stage"], row["old_task"], row["new_task"])
        for row in raw
    })
    for method, seed, stage, old_task, new_task in keys:
        subset = [row for row in raw if (
            row["method"], row["seed"], row["checkpoint_stage"], row["old_task"], row["new_task"]
        ) == (method, seed, stage, old_task, new_task)]
        task = task_lookup[(method, seed, old_task, new_task)]
        for group, column in GROUPS.items():
            matrix = np.full((5, 5), np.nan)
            for row in subset:
                matrix[int(row["old_bin"]), int(row["new_bin"])] = float(row[column])
            if np.isnan(matrix).any():
                raise ValueError(f"incomplete matrix: {(method, seed, old_task, new_task, group)}")
            matrix_lookup[(method, int(seed), int(old_task), int(new_task), group)] = matrix
            diag = np.diag(matrix)
            positive = diag >= 0.05
            negative = diag <= -0.05
            task_column = {
                "all_trainable": "cos_all",
                "temporal_transformer": "cos_transformer",
                "policy_head": "cos_head",
            }[group]
            pair_rows.append({
                "method": method, "seed": int(seed), "checkpoint_stage": int(stage),
                "old_task": int(old_task), "new_task": int(new_task), "group": group,
                "task_level_cosine": float(task[task_column]),
                **{f"diag_bin{i}": float(diag[i]) for i in range(5)},
                "diag_mean": float(diag.mean()), "diag_min": float(diag.min()),
                "diag_max": float(diag.max()), "diag_range": float(diag.max() - diag.min()),
                "positive_bin_fraction": float(positive.mean()),
                "negative_bin_fraction": float(negative.mean()),
                "near_zero_bin_fraction": float((~positive & ~negative).mean()),
                "mixed_sign": bool(positive.any() and negative.any()),
            })
    write_csv(args.output_dir / "temporal_pair_summary.csv", SUMMARY_FIELDS, pair_rows)

    aggregated = []
    aggregate_keys = sorted({
        (row["method"], row["old_task"], row["new_task"], row["group"])
        for row in pair_rows
    })
    for method, old_task, new_task, group in aggregate_keys:
        rows = [row for row in pair_rows if (
            row["method"], row["old_task"], row["new_task"], row["group"]
        ) == (method, old_task, new_task, group)]
        task_values = [float(row["task_level_cosine"]) for row in rows]
        diag_means = [float(row["diag_mean"]) for row in rows]
        ranges = [float(row["diag_range"]) for row in rows]
        mixed_count = sum(as_bool(row["mixed_sign"]) for row in rows)
        large_count = sum(value >= 0.20 for value in ranges)
        aggregated.append({
            "method": method, "old_task": old_task, "new_task": new_task, "group": group,
            "task_level_mean": statistics.mean(task_values),
            "task_level_sample_std": statistics.stdev(task_values) if len(task_values) > 1 else "",
            "diag_mean": statistics.mean(diag_means),
            "diag_mean_sample_std": statistics.stdev(diag_means) if len(diag_means) > 1 else "",
            "diag_range_mean": statistics.mean(ranges),
            "mixed_sign_seed_count": mixed_count, "large_range_seed_count": large_count,
            "replicated": len(rows) >= 3 and mixed_count >= 2 and large_count >= 2,
        })
    write_csv(args.output_dir / "temporal_3seed_summary.csv", THREE_FIELDS, aggregated)

    figures = args.output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    methods = sorted({row["method"] for row in pair_rows})
    for method in methods:
        seeds = sorted({int(row["seed"]) for row in pair_rows if row["method"] == method})
        for seed in seeds:
            pair_keys = sorted({
                (int(row["old_task"]), int(row["new_task"])) for row in pair_rows
                if row["method"] == method and int(row["seed"]) == seed
            })
            if not pair_keys:
                continue
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
            image = None
            for axis, pair in zip(axes.flat, pair_keys):
                matrix = matrix_lookup[(method, seed, pair[0], pair[1], "all_trainable")]
                image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
                axis.set_title(f"task {pair[0]} vs {pair[1]}")
                axis.set_xlabel("new-task bin")
                axis.set_ylabel("old-task bin")
                axis.set_xticks(range(5)); axis.set_yticks(range(5))
            for axis in axes.flat[len(pair_keys):]:
                axis.axis("off")
            fig.colorbar(image, ax=axes, shrink=0.8, label="gradient cosine")
            fig.suptitle(f"{method.upper()} seed {seed}: all-trainable 5x5 cosine")
            fig.savefig(figures / f"{method}_seed{seed}_heatmaps.png", dpi=160)
            plt.close(fig)

        natural_pairs = sorted({
            (int(row["old_task"]), int(row["new_task"])) for row in pair_rows
            if row["method"] == method
        })
        for old_task, new_task in natural_pairs:
            fig, axis = plt.subplots(figsize=(6, 4))
            for seed in seeds:
                key = (method, seed, old_task, new_task, "all_trainable")
                if key in matrix_lookup:
                    axis.plot(range(5), np.diag(matrix_lookup[key]), marker="o", label=f"seed {seed}")
            axis.axhline(0, color="black", linewidth=0.8)
            axis.axhline(0.05, color="gray", linewidth=0.6, linestyle="--")
            axis.axhline(-0.05, color="gray", linewidth=0.6, linestyle="--")
            axis.set(xlabel="normalized progress bin", ylabel="gradient cosine",
                     title=f"{method.upper()} task {old_task} vs {new_task}: diagonal")
            axis.set_xticks(range(5)); axis.legend()
            fig.tight_layout()
            fig.savefig(figures / f"{method}_task{old_task}_vs_{new_task}_diagonal.png", dpi=160)
            plt.close(fig)

        module_rows = [row for row in aggregated if row["method"] == method]
        labels = sorted({f"{row['old_task']}-{row['new_task']}" for row in module_rows})
        x = np.arange(len(labels)); width = 0.25
        fig, axis = plt.subplots(figsize=(9, 4))
        for offset, group in enumerate(GROUPS):
            lookup = {
                f"{row['old_task']}-{row['new_task']}": float(row["diag_mean"])
                for row in module_rows if row["group"] == group
            }
            axis.bar(x + (offset - 1) * width, [lookup[label] for label in labels], width, label=group)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xticks(x, labels); axis.set_ylabel("3-seed mean diagonal cosine")
        axis.set_title(f"{method.upper()}: transformer versus head")
        axis.legend(); fig.tight_layout()
        fig.savefig(figures / f"{method}_module_comparison.png", dpi=160)
        plt.close(fig)
    print(f"[complete] raw_files={len(raw_files)} pair_rows={len(pair_rows)} output={args.output_dir}")


if __name__ == "__main__":
    main()
