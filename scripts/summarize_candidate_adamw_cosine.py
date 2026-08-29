#!/usr/bin/env python3
import argparse
import csv
import math
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from libero.lifelong.analysis.online_first_order_margin import build_future_cases
from libero.lifelong.analysis.retention_score_calibration import (
    average_ranks,
    binary_metrics,
    loso_predict,
    pooled_loso_metrics,
    quantile,
    ranking_metrics,
)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def merge_anchors(raw_dir):
    paths = sorted(Path(raw_dir).glob("seed*_stage*_anchor.csv"))
    if len(paths) != 6:
        raise ValueError(f"expected six anchor files, found {len(paths)}")
    rows = []
    for path in paths:
        selected = read_csv(path)
        if not selected:
            raise ValueError(f"empty anchor file: {path}")
        rows.extend(selected)
    keys = [
        (int(row["seed"]), int(row["stage"]), int(row["global_update"]))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate anchor key")
    finite_fields = (
        "aggregate_old_loss", "candidate_step_score", "candidate_step_cosine",
        "candidate_delta_norm", "actual_step_score", "actual_step_cosine",
        "actual_delta_norm", "candidate_actual_delta_cosine",
        "relative_l2_error", "max_abs_delta_error", "risk_cosine_abs_error",
        "risk_score_abs_error",
    )
    for row in rows:
        if not all(math.isfinite(float(row[field])) for field in finite_fields):
            key = (int(row["seed"]), int(row["stage"]), int(row["global_update"]))
            raise ValueError(f"non-finite anchor row: {key}")
    return rows


def pearson(first, second):
    first = [float(value) for value in first]
    second = [float(value) for value in second]
    if len(first) != len(second) or len(first) < 2:
        return float("nan")
    mean_first = statistics.mean(first)
    mean_second = statistics.mean(second)
    centered_first = [value - mean_first for value in first]
    centered_second = [value - mean_second for value in second]
    numerator = sum(a * b for a, b in zip(centered_first, centered_second))
    denominator = math.sqrt(
        sum(value * value for value in centered_first)
        * sum(value * value for value in centered_second)
    )
    return numerator / denominator if denominator else float("nan")


def parity_row(scope, rows):
    delta_cosines = [float(row["candidate_actual_delta_cosine"]) for row in rows]
    relative = [float(row["relative_l2_error"]) for row in rows]
    risk_errors = [float(row["risk_cosine_abs_error"]) for row in rows]
    candidate_risk = [float(row["candidate_step_cosine"]) for row in rows]
    actual_risk = [float(row["actual_step_cosine"]) for row in rows]
    return {
        "scope": scope,
        "anchors": len(rows),
        "delta_cosine_mean": statistics.mean(delta_cosines),
        "delta_cosine_median": statistics.median(delta_cosines),
        "delta_cosine_min": min(delta_cosines),
        "delta_cosine_p01": quantile(delta_cosines, 0.01),
        "relative_l2_mean": statistics.mean(relative),
        "relative_l2_median": statistics.median(relative),
        "relative_l2_p95": quantile(relative, 0.95),
        "relative_l2_p99": quantile(relative, 0.99),
        "relative_l2_max": max(relative),
        "max_abs_delta_error": max(float(row["max_abs_delta_error"]) for row in rows),
        "risk_cosine_mae": statistics.mean(risk_errors),
        "risk_cosine_median_error": statistics.median(risk_errors),
        "risk_cosine_p95_error": quantile(risk_errors, 0.95),
        "risk_cosine_p99_error": quantile(risk_errors, 0.99),
        "risk_cosine_max_error": max(risk_errors),
        "risk_cosine_pearson": pearson(candidate_risk, actual_risk),
        "risk_cosine_spearman": pearson(
            average_ranks(candidate_risk), average_ranks(actual_risk)
        ),
        "risk_sign_agreement": statistics.mean(
            int(row["risk_sign_agreement"]) for row in rows
        ),
    }


def parity_summary(anchors):
    scopes = [("ALL", anchors)]
    scopes.extend(
        (f"seed{seed}", [row for row in anchors if int(row["seed"]) == seed])
        for seed in (42, 43, 44)
    )
    scopes.extend(
        (f"stage{stage}", [row for row in anchors if int(row["stage"]) == stage])
        for stage in (8, 9)
    )
    return [parity_row(scope, rows) for scope, rows in scopes]


def ranking_outputs(cases_by_horizon):
    summaries = []
    folds_out = []
    cache = {}
    for horizon, cases in cases_by_horizon.items():
        horizon_rows = {}
        for score_type in ("candidate_step_cosine", "actual_step_cosine"):
            ranking = ranking_metrics(cases, score_type)
            predictions, folds = loso_predict(cases, score_type)
            pooled = pooled_loso_metrics(predictions)
            horizon_rows[score_type] = (ranking, pooled)
            cache[(horizon, score_type)] = (predictions, folds)
            for fold in folds:
                folds_out.append(
                    {
                        "horizon": horizon,
                        "score_type": score_type,
                        "heldout_seed": fold["heldout_seed"],
                        "train_seeds": fold["train_seeds"],
                        "threshold": fold["threshold"],
                        "train_BA": fold["train_balanced_accuracy"],
                        "test_cases": fold["test_cases"],
                        "test_BA": fold["test_balanced_accuracy"],
                        "test_sensitivity": fold["test_sensitivity"],
                        "test_specificity": fold["test_specificity"],
                        "test_false_safe": fold["test_false_safe_rate_all"],
                    }
                )
        candidate_ranking, candidate_pooled = horizon_rows["candidate_step_cosine"]
        actual_ranking, actual_pooled = horizon_rows["actual_step_cosine"]
        auc_delta = candidate_ranking["roc_auc"] - actual_ranking["roc_auc"]
        ba_delta = 100.0 * (
            candidate_pooled["balanced_accuracy"] - actual_pooled["balanced_accuracy"]
        )
        for score_type in ("candidate_step_cosine", "actual_step_cosine"):
            ranking, pooled = horizon_rows[score_type]
            summaries.append(
                {
                    "horizon": horizon,
                    "score_type": score_type,
                    "cases": ranking["cases"],
                    "harmful_prevalence": ranking["harmful_prevalence"],
                    "roc_auc": ranking["roc_auc"],
                    "average_precision": ranking["average_precision"],
                    "loso_balanced_accuracy": pooled["balanced_accuracy"],
                    "loso_sensitivity": pooled["sensitivity"],
                    "loso_specificity": pooled["specificity"],
                    "loso_false_safe": pooled["false_safe_rate_all"],
                    "delta_auc_candidate_minus_actual": auc_delta,
                    "delta_loso_ba_pp_candidate_minus_actual": ba_delta,
                }
            )
    return summaries, folds_out, cache


def seed_stage_summary(anchors, cases100):
    output = []
    for seed in (42, 43, 44):
        for stage in (8, 9):
            anchor_rows = [
                row for row in anchors
                if int(row["seed"]) == seed and int(row["stage"]) == stage
            ]
            cases = [
                row for row in cases100
                if int(row["seed"]) == seed and int(row["stage"]) == stage
            ]
            candidate = ranking_metrics(cases, "candidate_step_cosine")
            actual = ranking_metrics(cases, "actual_step_cosine")
            output.append(
                {
                    "seed": seed,
                    "stage": stage,
                    "anchors": len(anchor_rows),
                    "median_delta_cosine": statistics.median(
                        float(row["candidate_actual_delta_cosine"])
                        for row in anchor_rows
                    ),
                    "p99_relative_l2": quantile(
                        [float(row["relative_l2_error"]) for row in anchor_rows], 0.99
                    ),
                    "candidate_risk_cosine_mae": statistics.mean(
                        float(row["risk_cosine_abs_error"]) for row in anchor_rows
                    ),
                    "h100_cases": len(cases),
                    "h100_candidate_auc": candidate["roc_auc"],
                    "h100_actual_auc": actual["roc_auc"],
                }
            )
    return output


def lookup(rows, **criteria):
    return next(
        row for row in rows
        if all(str(row[key]) == str(value) for key, value in criteria.items())
    )


def decide_verdict(anchors, cases100, ranking_rows, parity):
    candidate100 = lookup(
        ranking_rows, horizon="H100", score_type="candidate_step_cosine"
    )
    actual100 = lookup(
        ranking_rows, horizon="H100", score_type="actual_step_cosine"
    )
    candidate300 = lookup(
        ranking_rows, horizon="H300", score_type="candidate_step_cosine"
    )
    actual300 = lookup(
        ranking_rows, horizon="H300", score_type="actual_step_cosine"
    )
    per_seed = Counter(int(row["seed"]) for row in cases100)
    complete = (
        len(anchors) >= 250 and len(cases100) >= 240
        and all(per_seed[seed] >= 75 for seed in (42, 43, 44))
    )
    supported = (
        complete
        and parity["delta_cosine_median"] >= 0.999999
        and parity["delta_cosine_min"] >= 0.9990
        and parity["relative_l2_median"] <= 1e-5
        and parity["relative_l2_p99"] <= 1e-4
        and parity["risk_cosine_mae"] <= 1e-4
        and parity["risk_cosine_p99_error"] <= 1e-3
        and parity["risk_cosine_pearson"] >= 0.999
        and parity["risk_cosine_spearman"] >= 0.999
        and parity["risk_sign_agreement"] >= 0.99
        and candidate100["roc_auc"] >= 0.70
        and abs(candidate100["roc_auc"] - actual100["roc_auc"]) <= 0.01
        and abs(
            candidate100["loso_balanced_accuracy"]
            - actual100["loso_balanced_accuracy"]
        ) <= 0.02
        and abs(candidate300["roc_auc"] - actual300["roc_auc"]) <= 0.015
    )
    not_supported = (
        parity["delta_cosine_median"] < 0.99
        or parity["relative_l2_p99"] > 1e-2
        or parity["risk_cosine_mae"] > 0.01
        or parity["risk_cosine_pearson"] < 0.95
    )
    if supported:
        verdict = "CANDIDATE_ADAMW_STEP_COSINE_FEASIBILITY_SUPPORTED"
        next_action = "Run a preregistered Candidate Directional Guard Intervention Smoke on seeds 42/43 and stages 8/9; do not start formal guard training."
    elif not_supported:
        verdict = "CANDIDATE_ADAMW_STEP_COSINE_FEASIBILITY_NOT_SUPPORTED"
        next_action = "Stop the candidate-AdamW guard route and use the post-update signal only as a periodic monitor."
    else:
        verdict = "CANDIDATE_ADAMW_STEP_COSINE_FEASIBILITY_INCONCLUSIVE"
        next_action = "Explain candidate/actual parity or predictive-ranking variability before any intervention."
    return verdict, complete, next_action


def fmt(value):
    return f"{float(value):.6g}"


def pct(value):
    return f"{100.0 * float(value):.2f}%"


def write_report(
    output_dir, anchors, cases100, cases300, parity_rows, ranking_rows,
    seed_stage_rows, metadata, verdict, complete, next_action,
):
    parity = lookup(parity_rows, scope="ALL")
    candidate100 = lookup(ranking_rows, horizon="H100", score_type="candidate_step_cosine")
    actual100 = lookup(ranking_rows, horizon="H100", score_type="actual_step_cosine")
    candidate300 = lookup(ranking_rows, horizon="H300", score_type="candidate_step_cosine")
    actual300 = lookup(ranking_rows, horizon="H300", score_type="actual_step_cosine")
    seed_lines = [
        f"| {row['seed']} | {row['stage']} | {row['anchors']} | {float(row['median_delta_cosine']):.9f} | "
        f"{float(row['p99_relative_l2']):.3g} | {float(row['candidate_risk_cosine_mae']):.3g} | "
        f"{float(row['h100_candidate_auc']):.4f} | {float(row['h100_actual_auc']):.4f} |"
        for row in seed_stage_rows
    ]
    def ranking_line(label, row):
        return (
            f"| {label} | {float(row['roc_auc']):.4f} | {float(row['average_precision']):.4f} | "
            f"{pct(row['loso_balanced_accuracy'])} | {pct(row['loso_sensitivity'])} | {pct(row['loso_false_safe'])} |"
        )
    lines = [
        "# Candidate AdamW-Step Cosine Feasibility Diagnosis", "",
        "## Identity", "",
        f"- Verdict: `{verdict}`",
        "- Branch: `research/candidate-adamw-step-cosine-feasibility`",
        "- Starting SHA: `25065cdeed8e4e6fe7ebf7b9b5c20a4419ffcc90`",
        "- Final SHA: branch HEAD containing this report (reported in the external handoff)", "",
        "## Protocol", "",
        "Temporal V1 targeted continuations used seeds 42/43/44, stages 8/9, previous-stage ep95 entry proxies, 15 epochs, batch size 10, replay rate 0.5, five temporal bins, gate strength 1.0, and IFA disabled. Anchors begin after two burn-in epochs and occur every 100 optimizer updates. H100 is primary and H300 secondary.", "",
        f"Formal completeness: {'met' if complete else 'not met'}; anchors={len(anchors)}, H100={len(cases100)}, H300={len(cases300)}.", "",
        "## Optimizer semantics", "",
        "| Optimizer | Groups | Configured LR | Betas | Eps | Weight decay | AMSGrad | Maximize | Grad clipping |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|",
        f"| {metadata['optimizer_class']} | {metadata['param_groups']} | {metadata['configured_lr']} | {metadata['betas']} | {metadata['eps']} | {metadata['weight_decay']} | {metadata['amsgrad']} | {metadata['maximize']} | {metadata['grad_clip']} |", "",
        "## Candidate formula", "",
        "The candidate uses the current post-clip gradient and current AdamW moments/step, bias-corrected first and second moments, and decoupled weight decay. It mirrors Torch 1.13's decay-then-addcdiv parameter operations on a private clone and does not mutate parameters, gradients, or optimizer state.", "",
        "## Numerical displacement parity", "",
        f"- Candidate-actual cosine mean/median/min: {parity['delta_cosine_mean']:.9f} / {parity['delta_cosine_median']:.9f} / {parity['delta_cosine_min']:.9f}",
        f"- Relative L2 median/p99/max: {fmt(parity['relative_l2_median'])} / {fmt(parity['relative_l2_p99'])} / {fmt(parity['relative_l2_max'])}",
        f"- Maximum absolute tensor delta error: {fmt(parity['max_abs_delta_error'])}", "",
        "## Risk cosine parity", "",
        f"- MAE/p99/max: {fmt(parity['risk_cosine_mae'])} / {fmt(parity['risk_cosine_p99_error'])} / {fmt(parity['risk_cosine_max_error'])}",
        f"- Pearson/Spearman: {parity['risk_cosine_pearson']:.9f} / {parity['risk_cosine_spearman']:.9f}",
        f"- Sign agreement: {pct(parity['risk_sign_agreement'])}", "",
        "## H100", "",
        "| Score | AUC | AP | LOSO BA | Sensitivity | False-safe |",
        "|---|---:|---:|---:|---:|---:|",
        ranking_line("candidate cosine", candidate100),
        ranking_line("actual cosine", actual100), "",
        f"Candidate-actual AUC delta {candidate100['delta_auc_candidate_minus_actual']:+.4f}; LOSO BA delta {candidate100['delta_loso_ba_pp_candidate_minus_actual']:+.2f} pp.", "",
        "## H300", "",
        "| Score | AUC | AP | LOSO BA | Sensitivity | False-safe |",
        "|---|---:|---:|---:|---:|---:|",
        ranking_line("candidate cosine", candidate300),
        ranking_line("actual cosine", actual300), "",
        "## Seed-stage", "",
        "| Seed | Stage | Anchors | Median delta cosine | p99 relative L2 | Risk cosine MAE | H100 candidate AUC | H100 actual AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
        *seed_lines, "",
        "## Observed", "",
        "Candidate/actual displacement parity, aggregate old-gradient cosine parity, future BC-loss labels, ranking metrics, and LOSO folds are direct outputs of the six formal targeted continuations.", "",
        "## Inferred", "",
        f"H14 is `{verdict}`. The pre-step AdamW candidate "
        + ("reconstructs" if verdict.endswith("_SUPPORTED") and "NOT" not in verdict else "does not establish reconstruction of")
        + " the post-step directional diagnostic under the preregistered criteria.", "",
        "## Not established", "",
        "Candidate cosine is pre-step computable only because the current gradient and current AdamW state are available. This diagnosis does not show that rejecting or modifying a high-risk candidate improves lifelong performance. The score still uses an old-task probe gradient, so runtime cost and probe representativeness remain open issues. The targeted continuations are not rollout evaluation. BC-loss risk is not identical to rollout-success forgetting. The ep95 entry remains a late-stage proxy.", "",
        "## Next action", "",
        next_action, "",
    ]
    (output_dir / "FINAL_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report-dir", type=Path,
        default=Path("reports/candidate_adamw_step_cosine"),
    )
    args = parser.parse_args()
    output = args.report_dir
    anchors = merge_anchors(output / "raw")
    cases100 = build_future_cases(anchors, 100)
    cases300 = build_future_cases(anchors, 300)
    parity_rows = parity_summary(anchors)
    ranking_rows, folds, _ = ranking_outputs({"H100": cases100, "H300": cases300})
    seed_stage_rows = seed_stage_summary(anchors, cases100)
    metadata_rows = read_csv(output / "optimizer_metadata.csv")
    if len(metadata_rows) != 6:
        raise ValueError(f"expected six optimizer metadata rows, found {len(metadata_rows)}")
    signatures = {
        tuple(row[field] for field in (
            "optimizer_class", "param_groups", "configured_lr", "betas", "eps",
            "weight_decay", "amsgrad", "maximize", "grad_clip",
        ))
        for row in metadata_rows
    }
    if len(signatures) != 1:
        raise ValueError("optimizer metadata differs across continuations")
    parity = lookup(parity_rows, scope="ALL")
    verdict, complete, next_action = decide_verdict(
        anchors, cases100, ranking_rows, parity
    )
    write_csv(output / "anchor_raw.csv", anchors)
    write_csv(output / "cases_h100.csv", cases100)
    write_csv(output / "cases_h300.csv", cases300)
    write_csv(output / "parity_summary.csv", parity_rows)
    write_csv(output / "ranking_summary.csv", ranking_rows)
    write_csv(output / "loso_fold_metrics.csv", folds)
    write_csv(output / "seed_stage_summary.csv", seed_stage_rows)
    write_report(
        output, anchors, cases100, cases300, parity_rows, ranking_rows,
        seed_stage_rows, metadata_rows[0], verdict, complete, next_action,
    )
    print(
        f"[summary] anchors={len(anchors)} h100={len(cases100)} h300={len(cases300)}"
    )
    print(f"verdict={verdict}")


if __name__ == "__main__":
    main()
