"""Read-only EPR Stage A aggregation. No torch, model, optimizer or environment.

Inputs: ROOT/{GOAL42,OBJECT41}/{pairs.jsonl,FINAL_AUDIT.json}.
Output defaults to ROOT/analysis/ANALYSIS.json and GROUP_SUMMARIES.csv.
The rules below are fixed diagnostics, not significance tests or deployment gates.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

RULES = {
    "teacher_error_max": 0.5,
    "teacher_cross_mean_max": 0.25,
    "teacher_cross_std_max": 0.25,
    "motion_damage_delta_min": 0.1,
    "motion_improvement_delta_max": -0.1,
    "discovery_demos": [0, 1, 2, 3, 4],
    "validation_demos": [5, 6, 7, 8, 9],
    "native_validation_selected_damage_unique_targets_min": 10,
    "native_validation_selected_damage_distinct_demo_indices_min": 3,
    "native_validation_selected_damage_task_stage_cells_min": 2,
    "damage_jaccard_min": 0.5,
}
FLOAT_FIELDS = [
    "teacher_error", "student_error", "teacher_expected_mse", "student_expected_mse",
    "teacher_grip_error_prob", "student_grip_error_prob", "teacher_margin",
    "student_margin", "distribution_w2", "mean_shift_scaled", "std_shift_scaled",
]
BOOL_FIELDS = [
    "selected", "native_sample", "teacher_grip_correct", "student_grip_correct",
    "matched_is_winner",
]
METRICS = FLOAT_FIELDS + [
    "motion_delta", "expected_mse_delta", "grip_error_prob_delta",
    "gross_motion_damage", "gross_motion_improvement",
]
SLICES = {
    "suite": ("suite", "view", "selected", "split"),
    "task_stage": ("suite", "task", "stage", "view", "selected", "split"),
    "demo": ("suite", "task", "stage", "view", "selected", "split", "demo_id"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(row):
    """Per-window checkpoint comparison identity, excluding input view."""
    return (row["suite"], row["seed"], row["task"], row["stage"], row["sample_id"])


def target_identity(row):
    """One real supervision moment, not one padded window or later checkpoint."""
    return (row["task"], row["demo_id"], row["target"])


def task_demo(row):
    return (row["task"], row["demo_id"])


def teacher_valid(row):
    return row["teacher_error"] <= RULES["teacher_error_max"] and row["teacher_grip_correct"]


def normalize_row(raw):
    """Normalize labels, retaining raw identity fields and never inventing missing IDs."""
    row = dict(raw)
    suite = str(row.get("suite", "")).upper()
    for prefix in ("LIBERO_", "LIBERO-"):
        if suite.startswith(prefix):
            suite = suite[len(prefix):]
    if suite in ("GOAL42", "OBJECT41"):
        suite = suite[:-2]
    row["suite_raw"] = row.get("suite")
    row["suite"] = suite
    demo = row.get("demo_id")
    row["demo_id_raw"] = demo
    if isinstance(demo, str) and demo.startswith("demo_") and demo[5:].isdigit():
        demo = int(demo[5:])
    row["demo_id"] = demo if type(demo) is int and 0 <= demo <= 9 else None
    target, length = row.get("target"), row.get("length")
    row["identity_valid"] = (
        row.get("identity_unique", True) is True
        and row["demo_id"] is not None
        and type(target) is int and type(length) is int and 0 <= target < length)
    return row


def validate_row(row):
    required = ["suite", "seed", "task", "stage", "sample_id", "demo_id",
                "target", "length", "view"] + FLOAT_FIELDS + BOOL_FIELDS
    absent = sorted(set(required).difference(row))
    if absent:
        raise ValueError("Missing row fields: " + ", ".join(absent))
    if row["view"] not in ("cache", "native"):
        raise ValueError("Unknown view: " + repr(row["view"]))
    for key in ("seed", "task", "stage"):
        if not isinstance(row[key], int) or isinstance(row[key], bool):
            raise ValueError("Non-integer " + key)
    if row["task"] not in (6, 7, 8) or not row["task"] < row["stage"] <= 9:
        raise ValueError("Expected one of the six old-task transitions")
    if row["suite"] not in ("GOAL", "OBJECT"):
        raise ValueError("Unknown suite")
    if not isinstance(row["sample_id"], str) or not row["sample_id"]:
        raise ValueError("sample_id must be a nonempty string")
    for key in BOOL_FIELDS:
        if type(row[key]) is not bool:
            raise ValueError("Non-boolean " + key)
    for key in FLOAT_FIELDS:
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("Nonfinite/non-numeric " + key)
        if key not in ("teacher_margin", "student_margin") and value < 0:
            raise ValueError("Negative " + key)
    for key in ("teacher_grip_error_prob", "student_grip_error_prob"):
        if not 0 <= row[key] <= 1:
            raise ValueError("Invalid probability " + key)
    if row["view"] == "native" and not row["native_sample"]:
        raise ValueError("Native row must be in the fixed native sample")
    if row["native_sample"] and not row["identity_valid"]:
        raise ValueError("Native sample has unverified demonstration identity")
    if row["native_sample"]:
        for key in ("teacher_cross_mean_scaled", "teacher_cross_std_scaled"):
            value = row.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError("Invalid/missing paired-view metric " + key)


def enrich_rows(raw_rows):
    """Validate one-to-one paired input views, then apply fixed teacher-only gates."""
    rows = [normalize_row(r) for r in raw_rows]
    by_identity = defaultdict(dict)
    teacher_by_window_view = {}
    for row in rows:
        validate_row(row)
        key = identity(row)
        if row["view"] in by_identity[key]:
            raise ValueError("Duplicate comparison/view: " + repr(key))
        by_identity[key][row["view"]] = row
        teacher_key = (row["suite"], row["seed"], row["task"], row["sample_id"], row["view"])
        previous = teacher_by_window_view.setdefault(teacher_key, row)
        for field in ("demo_id", "target", "length", "selected", "native_sample", "identity_valid"):
            if previous[field] != row[field]:
                raise ValueError("Same-window identity changed across stages: " + field)
        for field in ("teacher_error", "teacher_expected_mse", "teacher_grip_error_prob", "teacher_margin"):
            if not math.isclose(previous[field], row[field], rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Teacher changed across later stages: " + field)
        if previous["teacher_grip_correct"] != row["teacher_grip_correct"]:
            raise ValueError("Teacher changed across later stages: teacher_grip_correct")
    for key, views in by_identity.items():
        if "cache" not in views:
            raise ValueError("Native comparison missing cached counterpart: " + repr(key))
        cache = views["cache"]
        if cache["native_sample"]:
            if set(views) != {"cache", "native"}:
                raise ValueError("Fixed native sample missing native counterpart: " + repr(key))
            native = views["native"]
            for field in ("demo_id", "target", "length", "selected", "native_sample"):
                if cache[field] != native[field]:
                    raise ValueError("Paired identity mismatch " + field + ": " + repr(key))
            for field in ("teacher_cross_mean_scaled", "teacher_cross_std_scaled"):
                if not math.isclose(cache[field], native[field], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError("Paired cross-view metric mismatch: " + field)
            eligible = (teacher_valid(cache) and teacher_valid(native)
                        and cache["teacher_cross_mean_scaled"] <= RULES["teacher_cross_mean_max"]
                        and cache["teacher_cross_std_scaled"] <= RULES["teacher_cross_std_max"])
        else:
            if len(views) != 1:
                raise ValueError("Unexpected native counterpart outside fixed sample")
            eligible = None  # Not checked natively; never silently classify as validated.
        for row in views.values():
            delta = row["student_error"] - row["teacher_error"]
            row.update(
                split=("unknown" if not row["identity_valid"] else
                       "discovery" if row["demo_id"] < 5 else "validation"),
                teacher_valid=teacher_valid(row),
                cross_eligible=eligible,
                motion_delta=delta,
                expected_mse_delta=row["student_expected_mse"] - row["teacher_expected_mse"],
                grip_error_prob_delta=row["student_grip_error_prob"] - row["teacher_grip_error_prob"],
                motion_damage=delta >= RULES["motion_damage_delta_min"],
                motion_improvement=delta <= RULES["motion_improvement_delta_max"],
                grip_damage=row["teacher_grip_correct"] and not row["student_grip_correct"],
                grip_improvement=not row["teacher_grip_correct"] and row["student_grip_correct"],
                gross_motion_damage=max(delta, 0.0),
                gross_motion_improvement=max(-delta, 0.0),
            )
            row["damage"] = row["motion_damage"] or row["grip_damage"]
            row["improvement"] = row["motion_improvement"] or row["grip_improvement"]
    return rows


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else None


def summarize(rows):
    """Window means and equal-weight real-target means are deliberately both retained."""
    targets = defaultdict(list)
    for row in rows:
        if row["identity_valid"]:
            targets[target_identity(row)].append(row)
    result = {
        "rows": len(rows),
        "identity_valid_rows": sum(r["identity_valid"] for r in rows),
        "identity_invalid_rows": sum(not r["identity_valid"] for r in rows),
        "unique_targets": len(targets),
        "distinct_task_demos": len({task_demo(r) for r in rows if r["identity_valid"]}),
        "distinct_task_stage_cells": len({(r["task"], r["stage"]) for r in rows}),
        "teacher_valid_rows": sum(r["teacher_valid"] for r in rows),
        "cross_checked_rows": sum(r["cross_eligible"] is not None for r in rows),
        "cross_eligible_rows": sum(r["cross_eligible"] is True for r in rows),
        "damage_rows": sum(r["damage"] for r in rows),
        "improvement_rows": sum(r["improvement"] for r in rows),
        "both_damage_and_improvement_rows": sum(r["damage"] and r["improvement"] for r in rows),
        "motion_damage_rows": sum(r["motion_damage"] for r in rows),
        "motion_improvement_rows": sum(r["motion_improvement"] for r in rows),
        "grip_damage_rows": sum(r["grip_damage"] for r in rows),
        "grip_improvement_rows": sum(r["grip_improvement"] for r in rows),
        "damage_unique_targets": len({target_identity(r) for r in rows if r["identity_valid"] and r["damage"]}),
        "improvement_unique_targets": len({target_identity(r) for r in rows if r["identity_valid"] and r["improvement"]}),
        "matched_is_winner_rows": sum(r["matched_is_winner"] for r in rows),
        "gripper_SS": sum(r["teacher_grip_correct"] and r["student_grip_correct"] for r in rows),
        "gripper_SF": sum(r["teacher_grip_correct"] and not r["student_grip_correct"] for r in rows),
        "gripper_FS": sum(not r["teacher_grip_correct"] and r["student_grip_correct"] for r in rows),
        "gripper_FF": sum(not r["teacher_grip_correct"] and not r["student_grip_correct"] for r in rows),
    }
    for key in METRICS:
        result["mean_" + key] = mean(r[key] for r in rows)
        result["unique_target_mean_" + key] = mean(mean(r[key] for r in rs) for rs in targets.values())
    for key in ("gross_motion_damage", "gross_motion_improvement"):
        result["sum_" + key] = sum(r[key] for r in rows)
        result["unique_target_sum_" + key] = sum(mean(r[key] for r in rs) for rs in targets.values())
    result["matched_is_winner_rate"] = result["matched_is_winner_rows"] / len(rows) if rows else None
    result["teacher_valid_rate"] = result["teacher_valid_rows"] / len(rows) if rows else None
    result["cross_eligible_among_checked_rate"] = (
        result["cross_eligible_rows"] / result["cross_checked_rows"]
        if result["cross_checked_rows"] else None)
    return result


def engineering_pass(audit):
    """Require an explicit positive status and reject any explicit contradictory signal."""
    indicators = []
    for key in ("passed", "engineering_pass", "all_passed"):
        if key in audit:
            if type(audit[key]) is not bool:
                return False
            indicators.append(audit[key])
    if "status" in audit:
        indicators.append(str(audit["status"]).upper() in ("PASS", "PASSED"))
    return bool(indicators) and all(indicators)


def jaccard(first, second):
    union = first | second
    return len(first & second) / len(union) if union else None


def coverage_acceptance(rows, audit):
    """Independently verify the six cells and the upstream per-task row accounting."""
    for field in ("cache_windows", "native_windows", "pair_rows"):
        if field not in audit:
            raise ValueError("FINAL_AUDIT missing count field " + field)
    if type(audit["pair_rows"]) is not int or audit["pair_rows"] != len(rows):
        raise ValueError("pair_rows does not match pairs.jsonl")
    cells = {(task, stage) for task in (6, 7, 8) for stage in range(task + 1, 10)}
    details = []
    for view, field in (("cache", "cache_windows"), ("native", "native_windows")):
        counts = audit[field]
        if not isinstance(counts, dict) or set(str(k) for k in counts) != {"6", "7", "8"}:
            raise ValueError("Invalid per-task counts in " + field)
        counts = {int(k): v for k, v in counts.items()}
        if any(type(n) is not int or n < 1 for n in counts.values()):
            raise ValueError("Counts must be positive in " + field)
        observed_cells = {(r["task"], r["stage"]) for r in rows if r["view"] == view}
        if observed_cells != cells:
            raise ValueError("Missing or extra task-stage cells for " + view)
        for task in (6, 7, 8):
            first_ids = None
            for stage in range(task + 1, 10):
                members = [r for r in rows if r["view"] == view
                           and r["task"] == task and r["stage"] == stage]
                if len(members) != counts[task]:
                    raise ValueError("Per-cell row count differs from FINAL_AUDIT")
                ids = {r["sample_id"] for r in members}
                if first_ids is None:
                    first_ids = ids
                elif first_ids != ids:
                    raise ValueError("Per-task sample universe changed across later stages")
                details.append(dict(view=view, task=task, stage=stage,
                                    observed=len(members), expected=counts[task]))
    expected_total = sum((9 - t) * (audit["cache_windows"][str(t)] +
                                   audit["native_windows"][str(t)]) for t in (6, 7, 8))
    if expected_total != len(rows):
        raise ValueError("Per-task count identity does not match pair_rows")
    return dict(status="PASS", cells_per_view=6, pair_rows=len(rows), details=details,
                teacher_consistent_across_later_stages=True,
                unknown_cache_identity_rows_retained=sum(not r["identity_valid"] for r in rows))


def suite_gate(rows, audit):
    """No stage-B work is launched by this recommendation."""
    eligible = [r for r in rows if r["native_sample"] and r["selected"]
                and r["split"] == "validation" and r["cross_eligible"] is True]
    native = [r for r in eligible if r["view"] == "native"]
    cache = [r for r in eligible if r["view"] == "cache"]
    native_damage = [r for r in native if r["damage"]]
    damage_targets = {target_identity(r) for r in native_damage}
    damage_demos = {task_demo(r) for r in native_damage}
    damage_demo_indices = {r["demo_id"] for r in native_damage}
    damage_cells = {(r["task"], r["stage"]) for r in native_damage}
    native_keys = {identity(r) for r in native if r["damage"]}
    cache_keys = {identity(r) for r in cache if r["damage"]}
    agreement = jaccard(native_keys, cache_keys)
    native_target_stage = {(r["task"], r["stage"], r["demo_id"], r["target"])
                           for r in native if r["damage"]}
    cache_target_stage = {(r["task"], r["stage"], r["demo_id"], r["target"])
                          for r in cache if r["damage"]}
    benefits = summarize([r for r in native if r["improvement"]])
    native_summary = summarize(native)
    checks = {
        "engineering_pass": engineering_pass(audit),
        "damage_unique_targets_at_least_10": len(damage_targets) >= 10,
        "damage_distinct_demo_indices_at_least_3": len(damage_demo_indices) >= 3,
        "damage_task_stage_cells_at_least_2": len(damage_cells) >= 2,
        "cache_native_damage_jaccard_at_least_0_5": agreement is not None and agreement >= 0.5,
    }
    sufficient_coverage = (len({target_identity(r) for r in native}) >= 10
                           and len({r["demo_id"] for r in native}) >= 3
                           and len({(r["task"], r["stage"]) for r in native}) >= 2)
    if all(checks.values()):
        decision = "WORTH_STAGE_B"
        reason = "All fixed opportunity gates passed; no update efficacy or safety claim."
    elif not checks["engineering_pass"]:
        decision = "INCONCLUSIVE"
        reason = "Engineering acceptance is absent or failed."
    elif not sufficient_coverage:
        decision = "INCONCLUSIVE"
        reason = "Insufficient cross-view-eligible validation coverage for the fixed gate."
    else:
        decision = "NO_GO"
        reason = "Complete diagnostic evidence did not meet the fixed opportunity gate."
    benefit_dominance = {
        "more_improvement_than_damage_rows":
            native_summary["improvement_rows"] > native_summary["damage_rows"],
        "more_improvement_than_damage_unique_targets":
            native_summary["improvement_unique_targets"] > native_summary["damage_unique_targets"],
        "gross_motion_improvement_exceeds_damage":
            native_summary["sum_gross_motion_improvement"] > native_summary["sum_gross_motion_damage"],
        "unique_target_motion_improvement_exceeds_damage":
            native_summary["unique_target_sum_gross_motion_improvement"] >
            native_summary["unique_target_sum_gross_motion_damage"],
    }
    return {
        "decision": decision, "recommend_B": all(checks.values()), "reason": reason,
        "checks": checks, "sufficient_eligible_validation_coverage": sufficient_coverage,
        "native_validation_selected_cross_eligible": native_summary,
        "native_benefit_rows_summary": benefits,
        "native_damage_unique_targets": len(damage_targets),
        "native_damage_distinct_task_demos": len(damage_demos),
        "native_damage_distinct_demo_indices": len(damage_demo_indices),
        "native_damage_demo_indices": sorted({r["demo_id"] for r in native_damage}),
        "native_damage_task_demos": [list(x) for x in sorted(damage_demos)],
        "native_damage_task_stage_cells": [list(x) for x in sorted(damage_cells)],
        "damage_cache_native_jaccard": agreement,
        "damage_cache_native_jaccard_unique_target_stage":
            jaccard(native_target_stage, cache_target_stage),
        "benefit_dominance_flags": benefit_dominance,
        "benefit_dominance_warning": any(benefit_dominance.values()),
        "benefits_are_not_a_gate": True,
        "stage_B_started": False,
    }


def analyze_suite(raw_rows, audit):
    rows = enrich_rows(raw_rows)
    suites = {r["suite"] for r in rows}
    seeds = {r["seed"] for r in rows}
    if len(suites) != 1 or len(seeds) != 1:
        raise ValueError("One suite input must contain exactly one suite and seed")
    coverage = coverage_acceptance(rows, audit)
    groups = []
    for level, fields in SLICES.items():
        buckets = defaultdict(list)
        for row in rows:
            buckets[tuple(row[k] for k in fields)].append(row)
        for key, members in sorted(buckets.items(), key=lambda item: repr(item[0])):
            dimensions = dict(zip(fields, key))
            # Teacher-only and dual-view subsets are explicit instead of silently filtering.
            for population in ("all", "teacher_valid", "cross_eligible"):
                subset = (members if population == "all" else
                          [r for r in members if r[population] is True])
                groups.append(dict(level=level, population=population, **dimensions, **summarize(subset)))
    return {
        "suite": next(iter(suites)), "seed": next(iter(seeds)),
        "coverage_acceptance": coverage,
        "rows": len(rows), "cache_rows": sum(r["view"] == "cache" for r in rows),
        "native_rows": sum(r["view"] == "native" for r in rows),
        "gate": suite_gate(rows, audit), "groups": groups,
    }


def analyze_root(root, out_dir=None):
    root = Path(root).resolve()
    out_dir = Path(out_dir).resolve() if out_dir else root / "analysis"
    evidence = {}
    results = []
    for name in ("GOAL42", "OBJECT41"):
        folder = root / name
        pairs = folder / "pairs.jsonl"
        final_audit = folder / "FINAL_AUDIT.json"
        if not pairs.is_file() or not final_audit.is_file():
            raise FileNotFoundError("Required input missing in " + str(folder))
        before = {str(p.relative_to(root)): sha256(p) for p in (pairs, final_audit)}
        with pairs.open(encoding="utf-8-sig") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        audit = json.loads(final_audit.read_text(encoding="utf-8-sig"))
        result = analyze_suite(rows, audit)
        expected = ("GOAL", 42) if name == "GOAL42" else ("OBJECT", 41)
        if (result["suite"], result["seed"]) != expected:
            raise ValueError("Unexpected suite/seed for " + name)
        result["input_subdir"] = name
        results.append(result)
        after = {str(p.relative_to(root)): sha256(p) for p in (pairs, final_audit)}
        if before != after:
            raise ValueError("Input changed during analysis")
        evidence[name] = dict(before=before, after=after, unchanged=True)
    output = {
        "status": "READ_ONLY_DIAGNOSTIC_COMPLETE",
        "rules": RULES, "suites": results, "input_protection": evidence,
        "optimizer_calls": 0, "environment_calls": 0, "model_forwards_in_local_analysis": 0,
        "stage_B_started": False,
        "limits": [
            "Observed checkpoints all saw these demonstrations; validation is diagnostic held-out grouping.",
            "Window and checkpoint comparisons are correlated, not independent trials.",
            "Expert action agreement is not closed-loop task success or safety.",
            "Task-demo pairs identify distinct trajectories; the conservative gate also requires three distinct demo indices.",
            "Native sample conclusions apply to the frozen stratified sample, not every native input.",
            "Thresholds are fixed exploratory rules, not statistically calibrated significance levels.",
            "Stage A may establish an opportunity; only a later authorized update test can test EPR efficacy.",
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ANALYSIS.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    groups = [g for s in results for g in s["groups"]]
    fields = list(dict.fromkeys(k for g in groups for k in g))
    with (out_dir / "GROUP_SUMMARIES.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(groups)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    result = analyze_root(args.root, args.out_dir)
    print(json.dumps({s["input_subdir"]: s["gate"]["decision"] for s in result["suites"]}))


if __name__ == "__main__":
    main()
