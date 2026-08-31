#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "paper_reproduction_v2" / "task6"
ARMS = ("current_legacy", "release18_legacy", "batch10_linear_step", "batch10_constant")
FORMAL = {70, 75, 80, 85, 90, 95}


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main():
    summary = []
    eval_curves = []
    train_curves = []
    lr_rows = []
    for arm in ARMS:
        eval_rows = read_rows(REPORT / arm / "task6_eval.csv")
        eval_curves.extend(eval_rows)
        selected = max(
            (row for row in eval_rows if int(row["epoch"]) in FORMAL),
            key=lambda row: (float(row["success_rate"]), int(row["epoch"])),
        )
        run_root = ROOT / "runs" / f"paper_v2_object_seed42_task6_{arm}"
        run_dir = sorted(run_root.glob("run_*"))[-1]
        train = read_rows(run_dir / "paper_repro_train_curve.csv")
        lr = read_rows(run_dir / "lr_protocol.csv")
        train_curves.extend(train)
        lr_rows.extend(lr)
        late = [float(row["mean_bc_loss"]) for row in train if int(row["epoch"]) >= 90]
        summary.append({
            "arm": arm,
            "batch_size": lr[0]["batch_size"],
            "scheduler_mode": lr[0]["scheduler_mode"],
            "selected_epoch": selected["epoch"],
            "selected_success": selected["success_rate"],
            "ep99_success": next(row["success_rate"] for row in eval_rows if int(row["epoch"]) == 99),
            "late_mean_bc_loss": sum(late) / len(late),
        })

    for name, rows in (("arm_summary.csv", summary), ("task6_eval_curves.csv", eval_curves),
                       ("task6_train_curves.csv", train_curves), ("lr_protocol.csv", lr_rows)):
        with (REPORT / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    by_arm = {row["arm"]: float(row["selected_success"]) for row in summary}
    c_gain = 100 * (by_arm["batch10_linear_step"] - by_arm["current_legacy"])
    d_gain = 100 * (by_arm["batch10_constant"] - by_arm["current_legacy"])
    batch_gap = 100 * abs(by_arm["release18_legacy"] - by_arm["current_legacy"])
    if c_gain >= 15:
        verdict = "PAPER_TASK6_PROTOCOL_RECONCILIATION_SUPPORTED"
    elif c_gain >= 5 or d_gain >= 5 or batch_gap >= 15:
        verdict = "PAPER_TASK6_PROTOCOL_RECONCILIATION_PARTIAL"
    else:
        verdict = "PAPER_TASK6_PROTOCOL_RECONCILIATION_NOT_SUPPORTED"
    print(verdict)


if __name__ == "__main__":
    main()
