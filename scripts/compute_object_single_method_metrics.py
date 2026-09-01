#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from compute_object_metrics import load_matrix, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    matrix, selected = load_matrix(args.log)
    score = metrics(matrix)
    task6_success = matrix[0][0]
    promotion = (
        score["fwt"] >= 80.0
        and score["auc"] >= 75.0
        and task6_success >= 55.0
    )
    payload = {
        "benchmark": "LIBERO_OBJECT",
        "method": "MLR paper candidate",
        "seed": 42,
        "protocol": {
            "batch_size": 10,
            "scheduler_mode": "batch_step",
            "warmup_steps": 500,
            "epochs_per_stage": 100,
            "ifa_enabled": False,
            "evaluation_trials": 20,
        },
        **score,
        "selected_epochs": selected,
        "historical_task6_success": 40.0,
        "task6_success": task6_success,
        "promotion_gate": {
            "fwt_at_least_80": score["fwt"] >= 80.0,
            "auc_at_least_75": score["auc"] >= 75.0,
            "task6_gain_at_least_15pp": task6_success >= 55.0,
            "pass": promotion,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "success_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage", "task6", "task7", "task8", "task9"])
        for index, row in enumerate(matrix):
            writer.writerow([index + 6] + ["" if value is None else value for value in row])
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
