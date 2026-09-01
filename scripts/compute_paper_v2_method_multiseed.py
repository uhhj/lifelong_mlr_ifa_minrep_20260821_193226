#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
from pathlib import Path

from compute_object_metrics import load_matrix, metrics


PAPER = {
    "mlr": {"fwt": (83.3, 2.6), "nbt": (12.3, 2.4), "auc": (77.6, 3.0)},
    "mlr_ifa": {"fwt": (84.6, 1.9), "nbt": (11.4, 5.6), "auc": (79.4, 1.5)},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("mlr", "mlr_ifa"), required=True)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows, results = [], {}
    for seed in (42, 43, 44):
        if args.method == "mlr" and seed == 42:
            log = args.runs_root / "object_seed42_mlr_paper_candidate" / "eval_seed42.log"
        else:
            log = args.runs_root / f"object_seed{seed}_{args.method}_paper_v2" / f"eval_seed{seed}.log"
        matrix, selected = load_matrix(log)
        score = metrics(matrix)
        results[str(seed)] = {**score, "selected_epochs": selected}
        rows.append({"seed": seed, "fwt": score["fwt"], "nbt": score["nbt"],
                     "auc": score["auc"], "task6": matrix[0][0]})

    summary = {}
    for name in ("fwt", "nbt", "auc"):
        values = [row[name] for row in rows]
        paper_mean, paper_std = PAPER[args.method][name]
        summary[name] = {"mean": round(statistics.mean(values), 3),
                         "sample_std": round(statistics.stdev(values), 3),
                         "paper_mean": paper_mean, "paper_reported_std": paper_std}
    if args.method == "mlr":
        reproduced = 80.7 <= summary["fwt"]["mean"] <= 85.9 and 74.6 <= summary["auc"]["mean"] <= 80.6
        consistent = abs(summary["fwt"]["mean"] - 83.3) <= 3.0 and abs(summary["auc"]["mean"] - 77.6) <= 3.0
        classification = "MLR_PAPER_LEVEL_REPRODUCED" if reproduced else ("MLR_PAPER_CONSISTENT" if consistent else "MLR_PAPER_STILL_UNRECONCILED")
        pass_next = reproduced or consistent
    else:
        reproduced = 82.7 <= summary["fwt"]["mean"] <= 86.5 and 77.9 <= summary["auc"]["mean"] <= 80.9
        classification = "MLR_IFA_PAPER_LEVEL_REPRODUCED" if reproduced else "MLR_IFA_PAPER_NOT_REPRODUCED"
        pass_next = reproduced
    payload = {"benchmark": "LIBERO_OBJECT", "method": args.method, "seeds": [42,43,44],
               "per_seed": results, "summary": summary, "classification": classification,
               "pass_next_gate": pass_next}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "three_seed_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
