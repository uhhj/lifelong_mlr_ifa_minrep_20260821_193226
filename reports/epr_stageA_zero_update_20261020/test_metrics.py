"""Synthetic program tests only; none of these rows are experimental observations."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("epr_analyze", Path(__file__).with_name("analyze.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def pair(index=0, *, suite="GOAL", seed=42, task=6, stage=7, demo=5,
         target=None, selected=True, teacher_error=0.2, student_error=0.4,
         teacher_grip=True, student_grip=True, cross_mean=0.05,
         native_student_error=None, sampled=True):
    row = dict(
        suite=suite, seed=seed, task=task, stage=stage, sample_id="s%04d" % index,
        demo_id=demo, target=index if target is None else target, length=1000,
        selected=selected, native_sample=sampled, view="cache",
        teacher_error=teacher_error, student_error=student_error,
        teacher_expected_mse=0.1, student_expected_mse=0.2,
        teacher_grip_correct=teacher_grip, student_grip_correct=student_grip,
        teacher_grip_error_prob=0.1, student_grip_error_prob=0.2,
        teacher_margin=1.0, student_margin=0.5, distribution_w2=0.01,
        mean_shift_scaled=0.1, std_shift_scaled=0.02, matched_is_winner=True,
    )
    if not sampled:
        return [row]
    row.update(teacher_cross_mean_scaled=cross_mean, teacher_cross_std_scaled=0.02)
    native = dict(row, view="native")
    if native_student_error is not None:
        native["student_error"] = native_student_error
    return [row, native]


def passing_rows():
    return complete_rows()


def complete_rows(start=0, stop=12, **kwargs):
    return [r for task in (6, 7, 8) for stage in range(task + 1, 10)
            for i in range(start, stop)
            for r in pair(i, demo=5 + i % 3, task=task, stage=stage, **kwargs)]


def audit_for(rows, **overrides):
    audit = dict(status="PASS", pair_rows=len(rows))
    for view in ("cache", "native"):
        audit[view + "_windows"] = {
            str(task): len({r["sample_id"] for r in rows if r["task"] == task and r["view"] == view})
            for task in (6, 7, 8)}
    audit.update(overrides)
    return audit


class MetricsTests(unittest.TestCase):
    def test_fixed_gate_passes_but_never_starts_B(self):
        rows = passing_rows()
        result = MODULE.analyze_suite(rows, audit_for(rows))
        gate = result["gate"]
        self.assertEqual(gate["decision"], "WORTH_STAGE_B")
        self.assertTrue(gate["recommend_B"])
        self.assertFalse(gate["stage_B_started"])
        self.assertEqual(gate["native_damage_unique_targets"], 36)
        self.assertEqual(gate["damage_cache_native_jaccard"], 1.0)

    def test_repeated_target_does_not_inflate_opportunity(self):
        rows = passing_rows()
        for row in rows:
            row.update(target=1, demo_id=5)
        gate = MODULE.analyze_suite(rows, audit_for(rows))["gate"]
        self.assertEqual(gate["native_damage_unique_targets"], 3)
        self.assertFalse(gate["recommend_B"])

    def test_damage_and_improvement_can_coexist(self):
        rows = MODULE.enrich_rows(pair(student_error=0.05, teacher_grip=True, student_grip=False))
        summary = MODULE.summarize(rows)
        self.assertEqual(summary["damage_rows"], 2)
        self.assertEqual(summary["improvement_rows"], 2)
        self.assertEqual(summary["both_damage_and_improvement_rows"], 2)
        self.assertEqual(summary["gripper_SF"], 2)

    def test_teacher_native_invalid_excludes_both_views(self):
        rows = pair()
        rows[1]["teacher_error"] = 0.7
        enriched = MODULE.enrich_rows(rows)
        self.assertTrue(enriched[0]["teacher_valid"])
        self.assertFalse(any(r["cross_eligible"] for r in enriched))

    def test_cross_view_threshold_and_missing_pair(self):
        rows = MODULE.enrich_rows(pair(cross_mean=0.25001))
        self.assertFalse(any(r["cross_eligible"] for r in rows))
        with self.assertRaisesRegex(ValueError, "missing native"):
            MODULE.enrich_rows(pair()[:1])

    def test_unsampled_cache_is_not_natively_verified(self):
        row = MODULE.enrich_rows(pair(sampled=False))[0]
        self.assertIsNone(row["cross_eligible"])

    def test_low_jaccard_blocks_even_when_native_damage_large(self):
        rows = passing_rows()
        for row in rows:
            if row["view"] == "cache":
                row["student_error"] = row["teacher_error"]
        gate = MODULE.analyze_suite(rows, audit_for(rows, engineering_pass=True))["gate"]
        self.assertEqual(gate["native_damage_unique_targets"], 36)
        self.assertEqual(gate["damage_cache_native_jaccard"], 0)
        self.assertEqual(gate["decision"], "NO_GO")

    def test_discovery_does_not_help_validation_gate(self):
        rows = passing_rows()
        for row in rows:
            row["demo_id"] -= 5
        gate = MODULE.analyze_suite(rows, audit_for(rows))["gate"]
        self.assertFalse(gate["recommend_B"])
        self.assertEqual(gate["native_damage_unique_targets"], 0)

    def test_benefit_dominance_is_reported_not_silent_gate(self):
        rows = passing_rows()
        rows += complete_rows(12, 30, teacher_error=0.4, student_error=0.1)
        gate = MODULE.analyze_suite(rows, audit_for(rows))["gate"]
        self.assertTrue(gate["recommend_B"])
        self.assertTrue(gate["benefit_dominance_warning"])
        self.assertEqual(gate["native_benefit_rows_summary"]["rows"], 108)

    def test_no_motion_damage_is_no_go_with_sufficient_coverage(self):
        rows = passing_rows()
        for row in rows:
            row["student_error"] = row["teacher_error"]
        gate = MODULE.analyze_suite(rows, audit_for(rows))["gate"]
        self.assertTrue(gate["sufficient_eligible_validation_coverage"])
        self.assertEqual(gate["decision"], "NO_GO")
        self.assertIsNone(gate["damage_cache_native_jaccard"])

    def test_engineering_absent_or_contradictory_is_inconclusive(self):
        self.assertFalse(MODULE.engineering_pass({}))
        self.assertFalse(MODULE.engineering_pass({"status": "PASS", "passed": False}))
        rows = passing_rows()
        gate = MODULE.analyze_suite(rows, audit_for(rows, status="FAIL"))["gate"]
        self.assertEqual(gate["decision"], "INCONCLUSIVE")

    def test_duplicate_nan_bool_and_identity_rejected(self):
        rows = pair()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            MODULE.enrich_rows(rows + rows)
        bad = pair()
        bad[0]["teacher_error"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            MODULE.enrich_rows(bad)
        bad = pair()
        bad[0]["selected"] = "true"
        with self.assertRaisesRegex(ValueError, "Non-boolean"):
            MODULE.enrich_rows(bad)
        bad = pair()
        bad[1]["target"] = 3
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            MODULE.enrich_rows(bad)

    def test_full_root_input_hash_protection(self):
        with tempfile.TemporaryDirectory(prefix="epr_synthetic_") as tmp:
            root = Path(tmp)
            for name, suite, seed in (("GOAL42", "GOAL", 42), ("OBJECT41", "OBJECT", 41)):
                folder = root / name
                folder.mkdir()
                rows = passing_rows()
                for row in rows:
                    row.update(suite=suite, seed=seed)
                (folder / "pairs.jsonl").write_text(
                    "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
                (folder / "FINAL_AUDIT.json").write_text(json.dumps(audit_for(rows)), encoding="utf-8")
            output = MODULE.analyze_root(root)
            self.assertEqual(output["optimizer_calls"], 0)
            self.assertEqual(output["environment_calls"], 0)
            self.assertTrue(all(v["unchanged"] for v in output["input_protection"].values()))
            self.assertTrue((root / "analysis" / "GROUP_SUMMARIES.csv").is_file())

    def test_runner_labels_and_unknown_cache_identity(self):
        rows = passing_rows()
        for row in rows:
            row.update(suite="GOAL42", demo_id="demo_" + str(row["demo_id"]))
        for task in (6, 7, 8):
            for stage in range(task + 1, 10):
                extra = pair(100, task=task, stage=stage, sampled=False)[0]
                extra.update(suite="GOAL42", demo_id="unknown", target=None,
                             length=None, identity_unique=False)
                rows.append(extra)
        result = MODULE.analyze_suite(rows, audit_for(rows))
        self.assertEqual(result["suite"], "GOAL")
        self.assertTrue(result["gate"]["recommend_B"])
        unknown = [g for g in result["groups"] if g["level"] == "suite"
                   and g["population"] == "all" and g["split"] == "unknown"]
        self.assertEqual(sum(g["rows"] for g in unknown), 6)
        self.assertEqual(sum(g["unique_targets"] for g in unknown), 0)
        self.assertEqual(result["coverage_acceptance"]["unknown_cache_identity_rows_retained"], 6)

    def test_teacher_must_be_unchanged_across_stages(self):
        rows = passing_rows()
        rows[0]["teacher_error"] += 0.01
        with self.assertRaisesRegex(ValueError, "Teacher changed"):
            MODULE.analyze_suite(rows, audit_for(rows))

    def test_missing_cell_and_incorrect_upstream_counts_are_rejected(self):
        rows = passing_rows()
        incomplete = [r for r in rows if not (r["task"] == 6 and r["stage"] == 8)]
        with self.assertRaisesRegex(ValueError, "Missing or extra"):
            MODULE.analyze_suite(incomplete, audit_for(incomplete))
        with self.assertRaisesRegex(ValueError, "pair_rows"):
            MODULE.analyze_suite(rows, audit_for(rows, pair_rows=0))
        wrong = audit_for(rows)
        wrong["native_windows"]["6"] += 1
        with self.assertRaisesRegex(ValueError, "Per-cell row count"):
            MODULE.analyze_suite(rows, wrong)

    def test_three_tasks_with_same_demo_does_not_satisfy_demo_gate(self):
        rows = passing_rows()
        for row in rows:
            row["demo_id"] = 5
        gate = MODULE.analyze_suite(rows, audit_for(rows))["gate"]
        self.assertEqual(gate["native_damage_distinct_task_demos"], 3)
        self.assertEqual(gate["native_damage_distinct_demo_indices"], 1)
        self.assertFalse(gate["recommend_B"])

    def test_improvement_boundary_is_inclusive(self):
        enriched = MODULE.enrich_rows(pair(teacher_error=0.1, student_error=0.0))
        self.assertTrue(all(r["motion_improvement"] for r in enriched))


if __name__ == "__main__":
    unittest.main(verbosity=2)
