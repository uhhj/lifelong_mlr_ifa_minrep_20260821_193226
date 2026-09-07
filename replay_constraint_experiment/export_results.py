"""Read completed runs; export compact evidence without training or evaluation."""
import csv
import importlib.metadata
import json
import math
import platform
import statistics
from pathlib import Path

from summarize import read_fixed_matrix, score_matrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/replay_constraint"


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    data = read(OUT / "aggregate_v1_results.json")
    evidence, matrices, per_task, epochs = [], [], [], []
    fields = ["seed", "benchmark_name", "start_task", "end_task", "past_data_rate", "repro_scheduler_mode"]
    for seed in (42, 43, 44):
        base = ROOT / "runs/final_paper_text_faithful/mlr" / ("seed%d" % seed)
        run = ROOT / "runs/replay_constraint/aggregate_v1/formal/constraint" / ("seed%d" % seed)
        cfg = read(run / "run_001/config.json")
        old = read(base / "run_001/config.json")
        for key in fields:
            assert cfg[key] == old[key], (seed, key)
        assert cfg["train"] == old["train"], (seed, "train config")
        lr, old_lr = [csv_rows(p / "run_001/lr_protocol.csv") for p in (run, base)]
        assert len(lr) == len(old_lr) == 400
        for a, b in zip(lr, old_lr):
            assert {k: v for k, v in a.items() if k != "arm"} == {k: v for k, v in b.items() if k != "arm"}
        manifest = read(run / "experiment.json")
        assert manifest["smoke"] is False and manifest["evaluation_trials"] == 20
        assert (run / "PRIMARY_EVAL_COMPLETE").is_file()
        validation = read(run / "protocol_validation.json")
        assert validation["status"] == "PASS"
        evidence.append({"seed": seed, "manifest": manifest, "validation": validation,
                         "train_config_matches_mlr": True, "lr_rows_exact_except_arm": 400,
                         "train_execution": read(run / "train_execution.json"),
                         "eval_execution": read(run / "eval_execution.json")})
        erows = csv_rows(run / "run_001/replay_constraint_epochs.csv")
        assert len(erows) == 400
        epochs.extend(erows)
        for arm in ("constraint", "mlr", "ifa_corrected"):
            result = data["runs"][arm][str(seed)]
            r = read_fixed_matrix(result["source_log"])
            for metric, value in score_matrix(r).items():
                assert value == result[metric], (seed, arm, metric)
            for i in range(4):
                for j in range(i + 1):
                    matrices.append(dict(method=arm, seed=seed, stage=i+6, task=j+6, success_percent=r[i][j]))
                per_task.append(dict(method=arm, seed=seed, task=i+6, fwt=r[i][i],
                                     task_auc=statistics.mean(r[q][i] for q in range(i, 4)),
                                     final_success=r[3][i]))
    import torch
    audit = {"status": "PASS", "runs": evidence, "matrix_cells": len(matrices),
             "candidate_epoch_rows": len(epochs), "environment": {
                 "python": platform.python_version(), "torch": torch.__version__,
                 "cuda_runtime": torch.version.cuda,
                 **{p: importlib.metadata.version(p) for p in ("mujoco", "robosuite", "transformers")}},
             "cost_limitation": "No contemporaneous full native/shadow cost run; no relative overhead claim."}
    (OUT / "COMPLETION_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_csv("success_matrices.csv", matrices)
    write_csv("per_task_results.csv", per_task)
    write_csv("projection_epochs.csv", epochs)
    for task in range(6, 10):
        stats = {arm: {k: statistics.mean(r[k] for r in per_task if r["method"] == arm and r["task"] == task)
                       for k in ("fwt", "task_auc", "final_success")} for arm in ("constraint", "mlr")}
        print(task, json.dumps({k: stats["constraint"][k] - stats["mlr"][k] for k in stats["mlr"]}))
    print(json.dumps(audit["environment"]))
    print("PASS: 9 matrices, 1200 candidate epochs; all training configurations and LR logs match MLR")


if __name__ == "__main__":
    main()
