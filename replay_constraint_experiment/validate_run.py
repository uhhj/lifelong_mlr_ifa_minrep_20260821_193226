"""Check execution completeness and fixed recipe, never gate on success rates."""
import argparse
import csv
import json
import math
from pathlib import Path

from summarize import diagnostic_summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    args = p.parse_args()
    run = args.run
    cfg = json.loads((run / "run_001/config.json").read_text())
    assert (run / "TRAINING_COMPLETE").is_file()
    assert cfg["train"]["n_epochs"] == 100
    assert cfg["train"]["batch_size"] == 10
    assert cfg["train"]["optimizer"]["kwargs"]["lr"] == 1e-4
    assert cfg["train"]["optimizer"]["kwargs"]["weight_decay"] == 0.1
    assert cfg["repro_scheduler_mode"] == "batch_step"
    assert cfg["ifa_enabled"] is False
    assert cfg["lifelong"]["projection_enabled"] is True
    assert cfg["past_data_rate"] == 0.5
    path = run / "run_001/replay_constraint_epochs.csv"
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        assert int(row["batches"]) > 0
        for value in row.values():
            if value and value not in ("True", "False"):
                assert math.isfinite(float(value)), row
    summary = diagnostic_summary(path)
    assert summary["complete_epoch_grid"]
    for stage in (7, 8, 9):
        assert sum(int(r["reference_batches"]) for r in rows if int(r["stage"]) == stage) > 0
    (run / "protocol_validation.json").write_text(json.dumps(
        {"status": "PASS", "diagnostics": summary, "performance_gate": False}, indent=2) + "\n")
    print("PASS: full training protocol and finite diagnostics")


if __name__ == "__main__":
    main()
