"""Synthetic scheduler tests ONLY; none of these values are experiment evidence."""
import copy
import unittest
from branch_gate import next_action, HASH_FIELDS, METHODS, SEEDS


def row(auc="3/4", baseline="1/2"):
    return dict(engineering="PASS", auc=auc, baseline_auc=baseline,
                **{k: "a"*64 for k in HASH_FIELDS})


class GateTests(unittest.TestCase):
    def test_initial(self):
        r = next_action({})
        self.assertEqual((r["method"], r["suite"], r["seed"]), ("RCTR", "OBJECT", 42))

    def test_tie_moves_next(self):
        r = next_action({"RCTR": {"OBJECT": {"42": row("2/4")}}})
        self.assertEqual((r["method"], r["seed"]), ("CAET", 42))

    def test_expansion_fixed_two(self):
        e = {"RCTR": {"OBJECT": {"42": row()}}}
        self.assertEqual(next_action(e)["seed"], 41)
        e["RCTR"]["OBJECT"]["41"] = row("1/4")
        self.assertEqual(next_action(e)["seed"], 43)

    def test_mean_not_majority(self):
        # Two negative seeds, yet positive mean: no unapproved 2/3 gate.
        e = {"RCTR": {"OBJECT": {"42": row("9/10"), "41": row("49/100"),
                                 "43": row("49/100")}}}
        r = next_action(e)
        self.assertEqual((r["suite"], r["seed"]), ("GOAL", 41))

    def test_three_seed_tie_skips_goal(self):
        e = {"RCTR": {"OBJECT": {"42": row("3/4"), "41": row("1/4"),
                                 "43": row("1/2")}}}
        self.assertEqual(next_action(e)["method"], "CAET")

    def test_goal_negative_does_not_stop_fixed_budget(self):
        e = {"RCTR": {"OBJECT": {str(s): row() for s in SEEDS},
                      "GOAL": {"41": row("1/10")}}}
        r = next_action(e)
        self.assertEqual((r["suite"], r["seed"]), ("GOAL", 42))

    def test_failures_block_not_skip(self):
        e = {"RCTR": {"OBJECT": {"42": row("1/4")}}}
        e["RCTR"]["OBJECT"]["42"]["engineering"] = "FAIL"
        self.assertEqual(next_action(e)["action"], "BLOCKED")

    def test_unrounded_exact_gate(self):
        e = {"RCTR": {"OBJECT": {"42": row("500000001/1000000000")}}}
        self.assertEqual(next_action(e)["seed"], 41)

    def test_reject_missing_receipts_and_decimal(self):
        for key in HASH_FIELDS:
            e = {"RCTR": {"OBJECT": {"42": row()}}}
            del e["RCTR"]["OBJECT"]["42"][key]
            self.assertEqual(next_action(e)["action"], "BLOCKED")
        e = {"RCTR": {"OBJECT": {"42": row("0.75")}}}
        self.assertEqual(next_action(e)["action"], "BLOCKED")

    def test_final_libero50_all_three_required(self):
        e = {m: {"OBJECT": {"42": row("1/4")}} for m in METHODS}
        for s in SEEDS:
            r = next_action(e)
            self.assertEqual((r["suite"], r["seed"]), ("LIBERO50", s))
            e.setdefault("RA_IFA", {}).setdefault("LIBERO50", {})[str(s)] = row()
        self.assertEqual(next_action(e)["action"], "ALL_ACCEPTED")
        e["RA_IFA"]["LIBERO50"]["42"]["engineering"] = "FAIL"
        self.assertEqual(next_action(e)["action"], "BLOCKED")

    def test_goal_complete_moves_next(self):
        e = {"RCTR": {suite: {str(s): row() for s in SEEDS}
                      for suite in ("OBJECT", "GOAL")}}
        self.assertEqual(next_action(e)["method"], "CAET")


if __name__ == "__main__":
    unittest.main()
