#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, field):
    return statistics.mean(float(row[field]) for row in rows)


def parse_flag(value):
    text = str(value).strip().lower()
    if text in ("1", "true"):
        return 1
    if text in ("0", "false"):
        return 0
    raise ValueError(f"invalid boolean value: {value!r}")


def rate(rows, field):
    return statistics.mean(parse_flag(row[field]) for row in rows)


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            result[index] = average_rank
        start = end
    return result


def pearson(first, second):
    if len(first) < 2:
        return float("nan")
    mean_first = statistics.mean(first)
    mean_second = statistics.mean(second)
    numerator = sum((a - mean_first) * (b - mean_second) for a, b in zip(first, second))
    first_norm = math.sqrt(sum((a - mean_first) ** 2 for a in first))
    second_norm = math.sqrt(sum((b - mean_second) ** 2 for b in second))
    return numerator / (first_norm * second_norm) if first_norm and second_norm else float("nan")


def summarize_layer_a(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["seed"], row["stage"])].append(row)
    result = []
    for (method, seed, stage), selected in sorted(grouped.items()):
        result.append(
            {
                "method": method, "seed": seed, "stage": stage,
                "candidate_relations": len(selected),
                "projected_current_safe_rate": rate(selected, "projected_current_safe"),
                "head_composition_leak_rate": rate(selected, "head_composition_leak"),
                "transformer_flip_rate": rate(selected, "transformer_flip"),
                "full_first_order_unsafe_rate": rate(selected, "full_first_order_unsafe"),
                "mean_projected_current_cosine": mean(selected, "projected_current_cosine"),
                "mean_final_head_cosine": mean(selected, "final_head_cosine"),
                "mean_transformer_contribution": mean(selected, "transformer_dot"),
                "mean_full_cosine": mean(selected, "full_cosine"),
            }
        )
    return result


def checkpoint_summary_row(method, seed, interval_type, rows):
    predicted = [float(row["pred_all"]) for row in rows]
    actual = [float(row["actual_delta"]) for row in rows]
    material = [row for row in rows if parse_flag(row["material_change"])]
    return {
        "method": method,
        "seed": seed,
        "interval_type": interval_type,
        "cases": len(rows),
        "actual_degradation_rate": statistics.mean(value > 0.0 for value in actual),
        "head_false_safe_rate": rate(rows, "head_false_safe"),
        "checkpoint_transformer_flip_rate": rate(rows, "checkpoint_transformer_flip"),
        "full_first_order_false_safe_rate": rate(rows, "full_first_order_false_safe"),
        "sign_agreement_rate": rate(rows, "sign_agreement"),
        "pearson": pearson(predicted, actual),
        "spearman": pearson(ranks(predicted), ranks(actual)),
        "mean_absolute_prediction_error": mean(rows, "absolute_prediction_error"),
        "material_cases": len(material),
        "material_full_false_safe_rate": "" if not material else rate(material, "full_first_order_false_safe"),
        "material_sign_agreement_rate": "" if not material else rate(material, "sign_agreement"),
    }


def summarize_checkpoints(rows, interval_type):
    result = []
    for method in ("V1", "V2"):
        method_rows = [row for row in rows if row["method"] == method]
        for seed in ("42", "43", "44"):
            selected = [row for row in method_rows if row["seed"] == seed]
            if selected:
                result.append(checkpoint_summary_row(method, seed, interval_type, selected))
        result.append(checkpoint_summary_row(method, "ALL", interval_type, method_rows))
    return result


def format_rate(value):
    return f"{100 * float(value):.2f}%"


def overall(summary, method):
    return next(row for row in summary if row["method"] == method and row["seed"] == "ALL")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=Path("reports/first_order_retention_validity"))
    args = parser.parse_args()
    layer_raw = read_csv(args.report_dir / "layer_a_raw.csv")
    short_raw = read_csv(args.report_dir / "checkpoint_transition_raw.csv")
    long_raw = read_csv(args.report_dir / "long_horizon_raw.csv")
    if len(short_raw) != 180:
        raise ValueError(f"expected 180 short old-task cases, found {len(short_raw)}")
    if len(long_raw) != 36:
        raise ValueError(f"expected 36 long-horizon old-task cases, found {len(long_raw)}")
    layer_summary = summarize_layer_a(layer_raw)
    short_summary = summarize_checkpoints(short_raw, "short")
    long_summary = summarize_checkpoints(long_raw, "long_horizon")
    write_csv(args.report_dir / "layer_a_summary.csv", layer_summary)
    write_csv(args.report_dir / "checkpoint_transition_summary.csv", short_summary)
    write_csv(args.report_dir / "long_horizon_summary.csv", long_summary)

    seed44_task7 = [
        row for row in short_raw
        if row["seed"] == "44" and row["old_task"] == "7"
        and row["stage"] in ("8", "9")
    ]
    write_csv(args.report_dir / "seed44_task7_case.csv", seed44_task7)

    all_short = [row for row in short_raw]
    overall_false_safe = rate(all_short, "full_first_order_false_safe")
    seed_false_safe = {
        seed: rate([row for row in all_short if row["seed"] == seed], "full_first_order_false_safe")
        for seed in ("42", "43", "44")
    }
    seed_sign = {
        seed: rate([row for row in all_short if row["seed"] == seed], "sign_agreement")
        for seed in ("42", "43", "44")
    }
    if overall_false_safe >= 0.20 and sum(value >= 0.15 for value in seed_false_safe.values()) >= 2:
        verdict = "LOCAL_FIRST_ORDER_RETENTION_MISMATCH_SUPPORTED"
    elif overall_false_safe < 0.05 and all(value >= 0.85 for value in seed_sign.values()):
        verdict = "LOCAL_FIRST_ORDER_RETENTION_MISMATCH_NOT_SUPPORTED"
    else:
        verdict = "LOCAL_FIRST_ORDER_RETENTION_MISMATCH_INCONCLUSIVE"

    v2_layer_primary = [
        row for row in layer_raw if row["method"] == "V2" and row["stage"] in ("8", "9")
    ]
    v2_layer_seed = {
        seed: rate(
            [row for row in v2_layer_primary if row["seed"] == seed],
            "head_composition_leak",
        )
        for seed in ("42", "43", "44")
    }
    composition = (
        rate(v2_layer_primary, "head_composition_leak") >= 0.15
        and sum(value >= 0.10 for value in v2_layer_seed.values()) >= 2
    )
    v2_short = [row for row in short_raw if row["method"] == "V2"]
    v2_transformer_seed = {
        seed: rate(
            [row for row in v2_short if row["seed"] == seed],
            "checkpoint_transformer_flip",
        )
        for seed in ("42", "43", "44")
    }
    transformer = (
        rate(v2_short, "checkpoint_transformer_flip") >= 0.15
        and sum(value >= 0.10 for value in v2_transformer_seed.values()) >= 2
    )
    attributions = []
    if composition:
        attributions.append("MIXED_GRADIENT_COMPOSITION_LEAKAGE")
    if transformer:
        attributions.append("TRANSFORMER_LEAKAGE")
    if verdict == "LOCAL_FIRST_ORDER_RETENTION_MISMATCH_SUPPORTED":
        attributions.append("FINITE_TRAJECTORY_FIRST_ORDER_BREAKDOWN")
    if not attributions:
        attributions.append("NO_PREREGISTERED_LAYER_ATTRIBUTION")

    def layer_overall(method, field, stages=(8, 9)):
        selected = [
            row for row in layer_raw
            if row["method"] == method and int(row["stage"]) in stages
        ]
        return rate(selected, field)

    v1_short = overall(short_summary, "V1")
    v2_short_summary = overall(short_summary, "V2")
    v1_long = overall(long_summary, "V1")
    v2_long = overall(long_summary, "V2")
    if verdict == "LOCAL_FIRST_ORDER_RETENTION_MISMATCH_SUPPORTED":
        next_action = "Test a finite-step loss-change or functional-loss probe criterion before adding further hard gradient constraints."
    elif composition:
        next_action = "Test an update-level functional constraint without increasing task/bin granularity."
    elif transformer:
        next_action = "Test policy-head plus selected-transformer regulation in a separate intervention stage."
    else:
        next_action = "Diagnose latent BC-loss drift versus rollout-success retention before modifying gradient regulation."

    lines = [
        "# First-Order Retention Validity Diagnosis", "",
        "## Identity", "",
        f"- Verdict: `{verdict}`",
        f"- Layer attributions: `{', '.join(attributions)}`",
        "- Branch: `research/first-order-retention-validity-diagnosis`",
        "- Starting SHA: `f51e10436d03404804068831acb81ceeaa2d249d`", "",
        "## V2 closure", "",
        "The V2 negative result was committed and pushed before this diagnosis as `f51e10436d03404804068831acb81ceeaa2d249d`. Its preregistered verdict remains `PER_TASK_TEMPORAL_ROBOT_IGFA_NOT_SUPPORTED`.", "",
        "## Protocol", "",
        "This is an offline latent/checkpoint diagnosis. It reused frozen V1/V2 checkpoints, seed-specific feature caches, temporal sidecars, and deterministic distribution-matched batches. No optimizer step, retraining, or rollout evaluation was performed. Checkpoint displacement includes the realized AdamW training trajectory, while the checkpoint-A Taylor prediction remains only a start-point first-order proxy.", "",
        "## Layer A", "",
        "Primary rates below use stages 8-9; stage 7 was retained as a single-old sanity check.", "",
        f"- V1 head composition leakage: {format_rate(layer_overall('V1', 'head_composition_leak'))}",
        f"- V2 head composition leakage: {format_rate(layer_overall('V2', 'head_composition_leak'))}",
        f"- V1 transformer flip: {format_rate(layer_overall('V1', 'transformer_flip'))}",
        f"- V2 transformer flip: {format_rate(layer_overall('V2', 'transformer_flip'))}",
        f"- V1 full first-order unsafe: {format_rate(layer_overall('V1', 'full_first_order_unsafe'))}",
        f"- V2 full first-order unsafe: {format_rate(layer_overall('V2', 'full_first_order_unsafe'))}", "",
        "## Short checkpoint transitions", "",
        f"- Cases: {len(short_raw)}", 
        f"- V1 head false-safe / transformer flip / full false-safe: {format_rate(v1_short['head_false_safe_rate'])} / {format_rate(v1_short['checkpoint_transformer_flip_rate'])} / {format_rate(v1_short['full_first_order_false_safe_rate'])}",
        f"- V2 head false-safe / transformer flip / full false-safe: {format_rate(v2_short_summary['head_false_safe_rate'])} / {format_rate(v2_short_summary['checkpoint_transformer_flip_rate'])} / {format_rate(v2_short_summary['full_first_order_false_safe_rate'])}",
        f"- V1 sign agreement; Pearson/Spearman: {format_rate(v1_short['sign_agreement_rate'])}; {float(v1_short['pearson']):.4f}/{float(v1_short['spearman']):.4f}",
        f"- V2 sign agreement; Pearson/Spearman: {format_rate(v2_short_summary['sign_agreement_rate'])}; {float(v2_short_summary['pearson']):.4f}/{float(v2_short_summary['spearman']):.4f}",
        f"- Per-seed full false-safe (both methods): {', '.join(f'seed{seed}={format_rate(value)}' for seed, value in seed_false_safe.items())}", "",
        "## Long horizon", "",
        f"- V1 full false-safe / sign agreement: {format_rate(v1_long['full_first_order_false_safe_rate'])} / {format_rate(v1_long['sign_agreement_rate'])}",
        f"- V2 full false-safe / sign agreement: {format_rate(v2_long['full_first_order_false_safe_rate'])} / {format_rate(v2_long['sign_agreement_rate'])}", "",
        "## V1 vs V2", "",
        "V1 and V2 use identical probes and checkpoint intervals. Differences therefore describe their realized local geometry and checkpoint trajectories rather than probe resampling.", "",
        "## Seed44 task7", "",
        f"The dedicated CSV contains {len(seed44_task7)} short transition cases for stage8/9 old task7 across V1/V2.", "",
        "## Observed", "",
        "Layer-A signs, checkpoint predictions, actual BC-loss changes, and displacement norms are direct outputs of the frozen formal artifacts.", "",
        "## Inferred", "",
        f"H8 is classified as `{verdict}` under the preregistered descriptive thresholds. Layer attributions are `{', '.join(attributions)}`.", "",
        "## Not established", "",
        "A false-safe result does not uniquely identify Hessian curvature; gradient-field rotation, optimizer trajectory, and multi-batch nonstationarity remain possible contributors. This diagnosis also does not establish how BC-loss drift maps to rollout-success retention.", "",
        "## Verdict", "",
        f"`{verdict}`", "",
        "## Layer attributions", "",
        *[f"- `{value}`" for value in attributions], "",
        "## Next action", "",
        next_action, "",
    ]
    (args.report_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"verdict={verdict}")
    print(f"attributions={','.join(attributions)}")


if __name__ == "__main__":
    main()
