#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

from libero.lifelong.analysis.finite_step_functional_probe import confusion_metrics


RHOS = (0.05, 0.10, 0.25, 0.50)
CASE_KEY = (
    "method", "seed", "stage", "old_task", "start_epoch", "end_epoch",
    "interval_type",
)


def read_csv(path):
    return list(csv.DictReader(Path(path).open(newline="", encoding="utf-8")))


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_flag(value):
    return str(value).strip().lower() in ("1", "true")


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in order[start:end]:
            result[index] = rank
        start = end
    return result


def pearson(first, second):
    if len(first) < 2:
        return float("nan")
    first_mean = statistics.mean(first)
    second_mean = statistics.mean(second)
    numerator = sum((a - first_mean) * (b - second_mean) for a, b in zip(first, second))
    first_norm = math.sqrt(sum((a - first_mean) ** 2 for a in first))
    second_norm = math.sqrt(sum((b - second_mean) ** 2 for b in second))
    return numerator / (first_norm * second_norm) if first_norm and second_norm else float("nan")


def summarize_rows(method, seed, rho, rows):
    probe = [float(row["probe_delta"]) for row in rows]
    actual = [float(row["actual_delta"]) for row in rows]
    first_order = [float(row["pred_all"]) for row in rows]
    finite = confusion_metrics(probe, actual)
    baseline = confusion_metrics(first_order, actual)
    first_order_false_safe = sum(
        parse_flag(row["first_order_false_safe"]) for row in rows
    )
    rescued = sum(parse_flag(row["rescued_first_order_false_safe"]) for row in rows)
    material = [row for row in rows if parse_flag(row["material_change"])]
    material_finite = confusion_metrics(
        [float(row["probe_delta"]) for row in material],
        [float(row["actual_delta"]) for row in material],
    )
    material_baseline = confusion_metrics(
        [float(row["pred_all"]) for row in material],
        [float(row["actual_delta"]) for row in material],
    )
    return {
        "method": method,
        "seed": seed,
        "rho": rho,
        "cases": finite["cases"],
        "endpoint_harmful_rate": finite["endpoint_harmful_rate"],
        "finite_false_safe_rate": finite["false_safe_rate"],
        "finite_false_alarm_rate": finite["false_alarm_rate"],
        "finite_sign_agreement_rate": finite["sign_agreement_rate"],
        "sensitivity": finite["sensitivity"],
        "specificity": finite["specificity"],
        "balanced_accuracy": finite["balanced_accuracy"],
        "pearson": pearson(probe, actual),
        "spearman": pearson(ranks(probe), ranks(actual)),
        "first_order_false_safe_rate": baseline["false_safe_rate"],
        "first_order_false_alarm_rate": baseline["false_alarm_rate"],
        "first_order_sign_agreement_rate": baseline["sign_agreement_rate"],
        "first_order_balanced_accuracy": baseline["balanced_accuracy"],
        "false_safe_reduction_pp": 100.0 * (
            baseline["false_safe_rate"] - finite["false_safe_rate"]
        ),
        "sign_agreement_gain_pp": 100.0 * (
            finite["sign_agreement_rate"] - baseline["sign_agreement_rate"]
        ),
        "balanced_accuracy_gain_pp": 100.0 * (
            finite["balanced_accuracy"] - baseline["balanced_accuracy"]
        ),
        "rescue_rate": rescued / first_order_false_safe if first_order_false_safe else float("nan"),
        "mean_nonlinearity_residual": statistics.mean(
            float(row["nonlinearity_residual"]) for row in rows
        ),
        "mean_absolute_nonlinearity_residual": statistics.mean(
            abs(float(row["nonlinearity_residual"])) for row in rows
        ),
        "material_cases": len(material),
        "material_finite_false_safe_rate": material_finite["false_safe_rate"],
        "material_finite_false_alarm_rate": material_finite["false_alarm_rate"],
        "material_finite_sign_agreement_rate": material_finite["sign_agreement_rate"],
        "material_balanced_accuracy": material_finite["balanced_accuracy"],
        "material_first_order_false_safe_rate": material_baseline["false_safe_rate"],
        "material_first_order_false_alarm_rate": material_baseline["false_alarm_rate"],
        "material_first_order_sign_agreement_rate": material_baseline["sign_agreement_rate"],
        "material_first_order_balanced_accuracy": material_baseline["balanced_accuracy"],
    }


def make_summary(rows):
    output = []
    methods = sorted({row["method"] for row in rows})
    seeds = sorted({row["seed"] for row in rows}, key=int)
    group_specs = [("ALL", "ALL", rows)]
    group_specs.extend((method, "ALL", [row for row in rows if row["method"] == method]) for method in methods)
    group_specs.extend(("ALL", seed, [row for row in rows if row["seed"] == seed]) for seed in seeds)
    group_specs.extend(
        (method, seed, [row for row in rows if row["method"] == method and row["seed"] == seed])
        for method in methods for seed in seeds
    )
    for method, seed, selected in group_specs:
        for rho in RHOS:
            at_rho = [row for row in selected if float(row["rho"]) == rho]
            output.append(summarize_rows(method, seed, rho, at_rho))
    return output


def summary_row(summary, method, seed, rho):
    return next(
        row for row in summary
        if row["method"] == method and row["seed"] == seed
        and float(row["rho"]) == rho
    )


def earliest_detection(rows):
    cases = defaultdict(list)
    for row in rows:
        cases[tuple(row[field] for field in CASE_KEY)].append(row)
    harmful = []
    for key, selected in cases.items():
        if float(selected[0]["actual_delta"]) <= 0.0:
            continue
        detected = [float(row["rho"]) for row in selected if float(row["probe_delta"]) > 0.0]
        harmful.append(
            {
                **dict(zip(CASE_KEY, key)),
                "earliest_harmful_detection_rho": min(detected) if detected else "NA",
            }
        )
    summaries = []
    methods = sorted({row["method"] for row in harmful})
    seeds = sorted({row["seed"] for row in harmful}, key=int)
    specs = [("ALL", "ALL", harmful)]
    specs.extend((method, "ALL", [row for row in harmful if row["method"] == method]) for method in methods)
    specs.extend(("ALL", seed, [row for row in harmful if row["seed"] == seed]) for seed in seeds)
    specs.extend(
        (method, seed, [row for row in harmful if row["method"] == method and row["seed"] == seed])
        for method in methods for seed in seeds
    )
    for method, seed, selected in specs:
        result = {"method": method, "seed": seed, "endpoint_harmful_cases": len(selected)}
        for rho in RHOS:
            result[f"detected_by_{rho:.2f}"] = sum(
                row["earliest_harmful_detection_rho"] != "NA"
                and float(row["earliest_harmful_detection_rho"]) <= rho
                for row in selected
            ) / len(selected)
        result["not_detected_by_0.50"] = sum(
            row["earliest_harmful_detection_rho"] == "NA" for row in selected
        ) / len(selected)
        summaries.append(result)
    return harmful, summaries


def pct(value):
    return f"{100.0 * float(value):.2f}%"


def pp(value):
    return f"{float(value):+.2f} pp"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("reports/finite_step_functional_probe_validity"),
    )
    args = parser.parse_args()
    short = read_csv(args.report_dir / "finite_step_probe_raw.csv")
    long_rows = read_csv(args.report_dir / "long_horizon_probe_raw.csv")
    if len(short) != 720 or len(long_rows) != 144:
        raise ValueError(f"formal row count mismatch: short={len(short)} long={len(long_rows)}")
    summary = make_summary(short)
    long_summary = make_summary(long_rows)
    write_csv(args.report_dir / "finite_step_probe_summary.csv", summary)
    write_csv(args.report_dir / "horizon_curve.csv", summary)
    write_csv(args.report_dir / "long_horizon_probe_summary.csv", long_summary)
    _, earliest_summary = earliest_detection(short)
    write_csv(args.report_dir / "earliest_detection_summary.csv", earliest_summary)
    seed44_task7 = [
        row for row in short
        if row["seed"] == "44" and row["old_task"] == "7"
        and row["stage"] in ("8", "9")
    ]
    write_csv(args.report_dir / "seed44_task7_case.csv", seed44_task7)

    primary = summary_row(summary, "ALL", "ALL", 0.10)
    seed_primary = {
        seed: summary_row(summary, "ALL", seed, 0.10)
        for seed in ("42", "43", "44")
    }
    method_primary = {
        method: summary_row(summary, method, "ALL", 0.10)
        for method in ("V1", "V2")
    }
    supported = (
        primary["false_safe_reduction_pp"] >= 15.0
        and primary["balanced_accuracy_gain_pp"] >= 10.0
        and sum(row["false_safe_reduction_pp"] >= 10.0 for row in seed_primary.values()) >= 2
        and all(
            row["finite_false_safe_rate"] < row["first_order_false_safe_rate"]
            for row in method_primary.values()
        )
    )
    rho25 = summary_row(summary, "ALL", "ALL", 0.25)
    not_supported = (
        primary["false_safe_reduction_pp"] < 5.0
        and primary["balanced_accuracy_gain_pp"] < 5.0
        and rho25["false_safe_reduction_pp"] < 10.0
        and rho25["balanced_accuracy_gain_pp"] < 10.0
    )
    if supported:
        verdict = "FINITE_STEP_FUNCTIONAL_PROBE_PREDICTIVE_GAIN_SUPPORTED"
    elif not_supported:
        verdict = "FINITE_STEP_FUNCTIONAL_PROBE_PREDICTIVE_GAIN_NOT_SUPPORTED"
    else:
        verdict = "FINITE_STEP_FUNCTIONAL_PROBE_PREDICTIVE_GAIN_INCONCLUSIVE"
    strong_horizons = [
        rho for rho in RHOS
        if summary_row(summary, "ALL", "ALL", rho)["false_safe_reduction_pp"] >= 15.0
        and summary_row(summary, "ALL", "ALL", rho)["balanced_accuracy_gain_pp"] >= 10.0
    ]
    if 0.10 in strong_horizons:
        signal_horizon = "EARLY"
    elif any(rho in strong_horizons for rho in (0.25, 0.50)):
        signal_horizon = "DELAYED"
    else:
        signal_horizon = "WEAK"

    if verdict == "FINITE_STEP_FUNCTIONAL_PROBE_PREDICTIVE_GAIN_SUPPORTED" and signal_horizon == "EARLY":
        next_action = "Test an Online Optimizer-Aware Functional Probe on Temporal V1, using the real optimizer state to construct the candidate step."
    elif verdict == "FINITE_STEP_FUNCTIONAL_PROBE_PREDICTIVE_GAIN_INCONCLUSIVE" and signal_horizon == "DELAYED":
        next_action = "Test a short-horizon trajectory probe or periodic retention validation before attempting a one-step gate."
    else:
        next_action = "Model the realized training path or test old BC-loss drift against rollout-success forgetting."

    horizon_lines = []
    for rho in RHOS:
        row = summary_row(summary, "ALL", "ALL", rho)
        horizon_lines.append(
            f"| {rho:.2f} | {pct(row['finite_false_safe_rate'])} | "
            f"{pp(row['false_safe_reduction_pp'])} | {pct(row['balanced_accuracy'])} | "
            f"{pp(row['balanced_accuracy_gain_pp'])} | {pct(row['finite_sign_agreement_rate'])} |"
        )
    seed_lines = []
    for seed, row in seed_primary.items():
        seed_lines.append(
            f"| {seed} | {pct(row['first_order_false_safe_rate'])} | "
            f"{pct(row['finite_false_safe_rate'])} | {pp(row['false_safe_reduction_pp'])} | "
            f"{pct(row['balanced_accuracy'])} |"
        )
    method_lines = []
    for method, row in method_primary.items():
        method_lines.append(
            f"| {method} | {pct(row['first_order_false_safe_rate'])} | "
            f"{pct(row['finite_false_safe_rate'])} | {pct(row['balanced_accuracy'])} | "
            f"{pct(row['finite_sign_agreement_rate'])} | {pct(row['rescue_rate'])} |"
        )
    long_lines = []
    for rho in (0.10, 0.25, 0.50):
        row = summary_row(long_summary, "ALL", "ALL", rho)
        long_lines.append(
            f"| {rho:.2f} | {pct(row['finite_false_safe_rate'])} | "
            f"{pp(row['false_safe_reduction_pp'])} | {pct(row['balanced_accuracy'])} | "
            f"{pct(row['finite_sign_agreement_rate'])} |"
        )
    seed44_lines = []
    for method in ("V1", "V2"):
        selected = [row for row in seed44_task7 if row["method"] == method]
        first_order_rate = sum(parse_flag(row["first_order_false_safe"]) for row in selected if float(row["rho"]) == 0.10) / 10
        rho10_rate = sum(parse_flag(row["probe_false_safe"]) for row in selected if float(row["rho"]) == 0.10) / 10
        rho25_rate = sum(parse_flag(row["probe_false_safe"]) for row in selected if float(row["rho"]) == 0.25) / 10
        seed44_lines.append(
            f"- {method}: first-order {pct(first_order_rate)}; rho0.10 {pct(rho10_rate)}; rho0.25 {pct(rho25_rate)}."
        )

    lines = [
        "# Finite-Step Functional Loss Probe Validity Diagnosis", "",
        "## Identity", "",
        f"- Verdict: `{verdict}`",
        f"- Signal horizon: `{signal_horizon}`",
        "- Branch: `research/finite-step-functional-probe-validity`",
        "- Starting SHA: `e49c1e1e9b4188528f457de29432c54df75f99af`", "",
        "## Prior evidence", "",
        "The prior diagnosis established `LOCAL_FIRST_ORDER_RETENTION_MISMATCH_SUPPORTED` with `FINITE_TRAJECTORY_FIRST_ORDER_BREAKDOWN` attribution.", "",
        "## Protocol", "",
        "The analysis reused the committed 180 short cases, 36 long-horizon cases, 2,304-record probe manifest, endpoint losses, and first-order predictions. It evaluated only the frozen policy head and temporal transformer along the realized checkpoint chord at fixed formal horizons 0.05/0.10/0.25/0.50. No training, optimizer step, rollout, endpoint recomputation, or probe resampling was performed. Rho=1 was used only for smoke endpoint parity.", "",
        "## Primary rho=0.10 table", "",
        "| Scope | FO false-safe | Finite false-safe | Finite balanced accuracy | Finite sign agreement | Rescue rate |",
        "|---|---:|---:|---:|---:|---:|",
        *method_lines,
        f"| Pooled | {pct(primary['first_order_false_safe_rate'])} | {pct(primary['finite_false_safe_rate'])} | {pct(primary['balanced_accuracy'])} | {pct(primary['finite_sign_agreement_rate'])} | {pct(primary['rescue_rate'])} |", "",
        "## Horizon table", "",
        "| rho | Finite false-safe | Reduction | Balanced accuracy | BA gain | Sign agreement |",
        "|---:|---:|---:|---:|---:|---:|",
        *horizon_lines, "",
        "## Seed replication", "",
        "| Seed | FO false-safe | Finite false-safe | Reduction | Finite balanced accuracy |",
        "|---:|---:|---:|---:|---:|",
        *seed_lines, "",
        "## Rescue analysis", "",
        f"At rho=0.10, the pooled rescue rate among committed first-order false-safe cases is {pct(primary['rescue_rate'])}. The all-case false-safe reduction is {pp(primary['false_safe_reduction_pp'])}.", "",
        "## Nonlinearity residual", "",
        f"At rho=0.10, mean signed residual `probe_delta - rho*pred_all` is {float(primary['mean_nonlinearity_residual']):.6f}; mean absolute residual is {float(primary['mean_absolute_nonlinearity_residual']):.6f}. This is descriptive nonlinearity evidence, not an identified Hessian term.", "",
        "## Long horizon", "",
        "| rho | Finite false-safe | Reduction | Balanced accuracy | Sign agreement |",
        "|---:|---:|---:|---:|---:|",
        *long_lines, "",
        "## Seed44 task7", "",
        *seed44_lines, "",
        "## Observed", "",
        "Intermediate functional losses and their classifications are direct evaluations on the frozen, committed probe records along the realized checkpoint chord.", "",
        "## Inferred", "",
        f"H9 is `{verdict}` under the preregistered rho=0.10 thresholds. The descriptive signal horizon is `{signal_horizon}`.", "",
        "## Not established", "",
        "The realized checkpoint chord uses the future endpoint direction, so it is not an online-deployable criterion and is not the actual optimizer path. The diagnosis does not evaluate rollout success and does not prove Hessian causality; gradient-field rotation, optimizer dynamics, and multi-batch nonstationarity remain possible contributors.", "",
        "## Verdict", "",
        f"`{verdict}`", "",
        "## Signal horizon", "",
        f"`{signal_horizon}`", "",
        "## Next action", "",
        next_action, "",
    ]
    (args.report_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"verdict={verdict}")
    print(f"signal_horizon={signal_horizon}")


if __name__ == "__main__":
    main()
