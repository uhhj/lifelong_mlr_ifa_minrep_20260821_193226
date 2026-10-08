#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

LINE = re.compile(r"\[task (\d+) epoch (\d+) and (\d+) succ\.\]\s+([0-9.]+)")
EPOCHS = (70, 75, 80, 85, 90, 95)


def load_matrix(path, epochs=EPOCHS):
    records = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = LINE.search(line)
        if not match:
            continue
        task, epoch, stage = int(match[1]), int(match[2]), int(match[3])
        rate = float(match[4])
        if 6 <= task <= stage <= 9:
            records[(stage, epoch, task)] = rate * 100.0

    missing_diagonal = [
        (stage, epoch, stage)
        for stage in range(6, 10)
        for epoch in epochs
        if (stage, epoch, stage) not in records
    ]
    if missing_diagonal:
        raise ValueError(
            f"missing current-task checkpoint-selection entries in {path}: "
            f"{missing_diagonal}"
        )

    matrix = [[None] * 4 for _ in range(4)]
    selected_epochs = {}
    for stage in range(6, 10):
        best_epoch = max(
            epochs,
            key=lambda epoch: (records[(stage, epoch, stage)], epoch),
        )
        selected_epochs[str(stage)] = best_epoch
        missing_selected = [
            (stage, best_epoch, task)
            for task in range(6, stage + 1)
            if (stage, best_epoch, task) not in records
        ]
        if missing_selected:
            raise ValueError(
                f"missing selected-checkpoint entries in {path}: "
                f"{missing_selected}"
            )
        for task in range(6, stage + 1):
            matrix[stage - 6][task - 6] = records[(stage, best_epoch, task)]
    return matrix, selected_epochs


def metrics(r):
    m = len(r)
    fwt = sum(r[i][i] for i in range(m)) / m
    per_task_nbt = [sum(r[i][i] - r[q][i] for q in range(i + 1, m)) / (m - i - 1) for i in range(m - 1)]
    nbt = sum(per_task_nbt) / (m - 1)
    per_task_auc = [sum(r[q][i] for q in range(i, m)) / (m - i) for i in range(m)]
    auc = sum(per_task_auc) / m
    return {"fwt": round(fwt, 3), "nbt": round(nbt, 3), "auc": round(auc, 3), "r": r}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlr-log", required=True)
    parser.add_argument("--mlr-ifa-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    mlr_r, mlr_epochs = load_matrix(args.mlr_log)
    mlr_ifa_r, mlr_ifa_epochs = load_matrix(args.mlr_ifa_log)
    result = {
        "benchmark": "LIBERO_OBJECT",
        "seed": 42,
        "checkpoint_selection": "best current-task success; latest epoch on ties",
        "mlr": {**metrics(mlr_r), "selected_epochs": mlr_epochs},
        "mlr_ifa": {**metrics(mlr_ifa_r), "selected_epochs": mlr_ifa_epochs},
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()


# Existing primary matrix parser, unchanged.
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

