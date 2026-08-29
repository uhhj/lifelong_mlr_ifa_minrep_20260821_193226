#!/usr/bin/env python3
import argparse
import csv
import gc
from collections import defaultdict
from pathlib import Path

import torch

from libero.lifelong.analysis.finite_step_functional_probe import (
    FORMAL_RHOS,
    classify_finite_step,
    load_probe_manifest,
    set_interpolated_backend,
)
from libero.lifelong.analysis.first_order_retention_validity import (
    capture_backend_parameters,
    evaluate_mean_loss,
    load_checkpoint_into_policy,
)
from libero.lifelong.models.brief import BriefPolicy
from scripts.diagnose_first_order_retention_validity import (
    checkpoint_path,
    run_dir,
)
from scripts.diagnose_per_task_temporal_interference import load_config


RAW_FIELDS = [
    "method", "seed", "stage", "old_task", "start_epoch", "end_epoch",
    "interval_type", "rho", "loss_start", "loss_probe", "loss_end",
    "probe_delta", "relative_probe_delta", "actual_delta",
    "relative_actual_delta", "pred_all", "taylor_at_rho",
    "nonlinearity_residual", "first_order_false_safe", "probe_false_safe",
    "probe_false_alarm", "probe_sign_agreement",
    "rescued_first_order_false_safe", "late_harmful_reversal",
    "material_change",
]
KEY_FIELDS = (
    "method", "seed", "stage", "old_task", "start_epoch", "end_epoch",
    "interval_type",
)


def read_canonical(path, expected_count, interval_type):
    rows = list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} {interval_type} cases, found {len(rows)}")
    if any(row["interval_type"] != interval_type for row in rows):
        raise ValueError(f"non-{interval_type} row in {path}")
    keys = [tuple(row[field] for field in KEY_FIELDS) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate canonical cases in {path}")
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def grouped_cases(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["method"], int(row["seed"]), int(row["stage"]),
            int(row["start_epoch"]), int(row["end_epoch"]), row["interval_type"],
        )
        groups[key].append(row)
    return groups


def evaluate_cases(project_root, cases, probes, rhos, device, probe_batch_size=32):
    output = []
    groups = grouped_cases(cases)
    by_method_seed = defaultdict(list)
    for key, rows in groups.items():
        by_method_seed[(key[0], key[1])].append((key, rows))
    for (method, seed), transitions in sorted(by_method_seed.items()):
        cfg = load_config(run_dir(project_root, method, seed) / "config.json", project_root, seed, device)
        policy = BriefPolicy(cfg, None).to(device).eval()
        for (case_method, case_seed, stage, start_epoch, end_epoch, interval_type), rows in sorted(transitions):
            start_stage = stage if interval_type == "short" else stage - 1
            start_path = checkpoint_path(
                project_root, case_method, case_seed, start_stage, start_epoch
            )
            end_path = checkpoint_path(
                project_root, case_method, case_seed, stage, end_epoch
            )
            load_checkpoint_into_policy(policy, start_path)
            start_snapshot = capture_backend_parameters(policy)
            load_checkpoint_into_policy(policy, end_path)
            end_snapshot = capture_backend_parameters(policy)
            for rho in rhos:
                set_interpolated_backend(policy, start_snapshot, end_snapshot, rho)
                for canonical in sorted(rows, key=lambda row: int(row["old_task"])):
                    old_task = int(canonical["old_task"])
                    loss_start = float(canonical["loss_start"])
                    loss_probe = evaluate_mean_loss(
                        policy, probes[(case_seed, old_task)], batch_size=probe_batch_size
                    )
                    actual_delta = float(canonical["actual_delta"])
                    probe_delta = loss_probe - loss_start
                    pred_all = float(canonical["pred_all"])
                    first_order_false_safe = str(
                        canonical["full_first_order_false_safe"]
                    ).strip().lower() in ("1", "true")
                    output.append(
                        {
                            "method": case_method,
                            "seed": case_seed,
                            "stage": stage,
                            "old_task": old_task,
                            "start_epoch": start_epoch,
                            "end_epoch": end_epoch,
                            "interval_type": interval_type,
                            "rho": rho,
                            "loss_start": loss_start,
                            "loss_probe": loss_probe,
                            "loss_end": float(canonical["loss_end"]),
                            "probe_delta": probe_delta,
                            "relative_probe_delta": probe_delta / max(abs(loss_start), 1e-12),
                            "actual_delta": actual_delta,
                            "relative_actual_delta": float(canonical["relative_actual_delta"]),
                            "pred_all": pred_all,
                            "taylor_at_rho": float(rho) * pred_all,
                            "nonlinearity_residual": probe_delta - float(rho) * pred_all,
                            "first_order_false_safe": first_order_false_safe,
                            **classify_finite_step(
                                probe_delta, actual_delta, first_order_false_safe
                            ),
                            "material_change": canonical["material_change"],
                        }
                    )
            print(
                f"[probe] method={case_method} seed={case_seed} stage={stage} "
                f"epochs={start_epoch}->{end_epoch} type={interval_type} "
                f"old_tasks={len(rows)} rhos={','.join(str(value) for value in rhos)}"
            )
        del policy
        gc.collect()
        torch.cuda.empty_cache()
    return output


def endpoint_parity(project_root, canonical, probes, device, probe_batch_size):
    method = canonical["method"]
    seed = int(canonical["seed"])
    stage = int(canonical["stage"])
    start_epoch = int(canonical["start_epoch"])
    end_epoch = int(canonical["end_epoch"])
    cfg = load_config(run_dir(project_root, method, seed) / "config.json", project_root, seed, device)
    policy = BriefPolicy(cfg, None).to(device).eval()
    load_checkpoint_into_policy(
        policy, checkpoint_path(project_root, method, seed, stage, start_epoch)
    )
    start_snapshot = capture_backend_parameters(policy)
    load_checkpoint_into_policy(
        policy, checkpoint_path(project_root, method, seed, stage, end_epoch)
    )
    end_snapshot = capture_backend_parameters(policy)
    paths = probes[(seed, int(canonical["old_task"]))]
    losses = {}
    for rho in (0.0, 0.10, 1.0):
        set_interpolated_backend(policy, start_snapshot, end_snapshot, rho)
        losses[rho] = evaluate_mean_loss(policy, paths, batch_size=probe_batch_size)
    for rho, field in ((0.0, "loss_start"), (1.0, "loss_end")):
        expected = float(canonical[field])
        error = abs(losses[rho] - expected)
        tolerance = max(1e-6, 1e-6 * abs(expected))
        if error > tolerance:
            raise AssertionError(
                f"rho={rho} endpoint parity failed: {losses[rho]} vs {expected}"
            )
    print(
        f"[endpoint-parity-pass] rho0={losses[0.0]:.10f} "
        f"rho1={losses[1.0]:.10f} rho0.10={losses[0.10]:.10f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("smoke", "formal"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--input-dir", type=Path,
        default=Path("reports/first_order_retention_validity"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("reports/finite_step_functional_probe_validity"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-batch-size", type=int, default=32)
    args = parser.parse_args()
    root = args.project_root.resolve()
    short = read_canonical(args.input_dir / "checkpoint_transition_raw.csv", 180, "short")
    long_rows = read_canonical(args.input_dir / "long_horizon_raw.csv", 36, "long_horizon")
    probes = load_probe_manifest(args.input_dir / "probe_manifest.csv", root)
    if args.mode == "smoke":
        parity_case = next(
            row for row in short
            if row["method"] == "V2" and row["seed"] == "42"
            and row["stage"] == "8" and row["old_task"] == "7"
            and row["start_epoch"] == "90" and row["end_epoch"] == "95"
        )
        endpoint_parity(root, parity_case, probes, args.device, args.probe_batch_size)
        selected = []
        for method in ("V1", "V2"):
            selected.extend([row for row in short if row["method"] == method][:5])
        rows = evaluate_cases(
            root, selected, probes, FORMAL_RHOS, args.device, args.probe_batch_size
        )
        write_csv(args.output_dir / "finite_step_probe_raw.csv", rows)
        if len(rows) != 40:
            raise AssertionError(f"expected 40 smoke rows, found {len(rows)}")
        print(f"[smoke-pass] cases=10 case_horizon_rows={len(rows)}")
    else:
        short_output = evaluate_cases(
            root, short, probes, FORMAL_RHOS, args.device, args.probe_batch_size
        )
        long_output = evaluate_cases(
            root, long_rows, probes, FORMAL_RHOS, args.device, args.probe_batch_size
        )
        if len(short_output) != 720 or len(long_output) != 144:
            raise AssertionError(
                f"formal row count mismatch: short={len(short_output)} long={len(long_output)}"
            )
        write_csv(args.output_dir / "finite_step_probe_raw.csv", short_output)
        write_csv(args.output_dir / "long_horizon_probe_raw.csv", long_output)
        print(
            f"[formal-pass] short_cases=180 short_rows={len(short_output)} "
            f"long_cases=36 long_rows={len(long_output)} probes=2304"
        )


if __name__ == "__main__":
    main()
