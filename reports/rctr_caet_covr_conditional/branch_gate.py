"""Pure, fail-closed branch decisions. This is not an experiment launcher.

Evidence schema: evidence[method][suite][seed] = {
  "engineering": "PASS", "acceptance_sha256": ...,
  "baseline_acceptance_sha256": ..., "episodes_sha256": ...,
  "baseline_episodes_sha256": ..., "auc": "229/320",
  "baseline_auc": "229/320"
}
An independent verifier must recompute the exact AUC and check these receipts
against immutable artifacts before this function is used by a real scheduler.
No success rate or acceptance assertion is generated here.
"""
from fractions import Fraction
import json
import re
import sys

METHODS = ("RCTR", "CAET", "COVR")
SEEDS = (41, 42, 43)
HASH_FIELDS = ("acceptance_sha256", "baseline_acceptance_sha256",
               "episodes_sha256", "baseline_episodes_sha256")


def verified_record(evidence, method, suite, seed):
    row = evidence.get(method, {}).get(suite, {}).get(str(seed))
    if row is None:
        return None, None
    if row.get("engineering") != "PASS":
        return None, "Missing or failed engineering acceptance"
    if not all(re.fullmatch(r"[0-9a-f]{64}", str(row.get(k, "")))
               for k in HASH_FIELDS):
        return None, "Missing evidence hash receipts"
    try:
        # Reject floats and rounded decimal metrics: require fraction strings.
        values = [row["auc"], row["baseline_auc"]]
        if not all(isinstance(v, str) and re.fullmatch(r"\d+/\d+", v)
                   for v in values):
            raise ValueError("exact fraction required")
        auc, baseline = map(Fraction, values)
        if not (0 <= auc <= 1 and 0 <= baseline <= 1):
            raise ValueError("rate out of range")
    except (KeyError, ValueError, ZeroDivisionError):
        return None, "Invalid exact AUC"
    return (auc, baseline), None


def next_action(evidence):
    decisions = []
    for method in METHODS:
        r, error = verified_record(evidence, method, "OBJECT", 42)
        if error:
            return dict(action="BLOCKED", method=method, suite="OBJECT",
                        seed=42, reason=error, decisions=decisions)
        if r is None:
            return dict(action="RUN_IF_ADMITTED", method=method, suite="OBJECT",
                        seed=42, decisions=decisions)
        if r[0] <= r[1]:
            decisions.append(dict(method=method, gate="OBJECT_seed42",
                                  delta=str(r[0]-r[1]), result="NO_GAIN"))
            continue
        for seed in (41, 43):
            r, error = verified_record(evidence, method, "OBJECT", seed)
            if error or r is None:
                return dict(action="BLOCKED" if error else "RUN_IF_ADMITTED",
                            method=method, suite="OBJECT", seed=seed,
                            reason=error, decisions=decisions)
        rows = [verified_record(evidence, method, "OBJECT", s)[0]
                for s in SEEDS]
        delta = sum((r[0]-r[1] for r in rows), Fraction(0))/3
        decisions.append(dict(method=method, gate="OBJECT_three_seed_mean",
                              delta=str(delta),
                              result="GOAL" if delta > 0 else "NO_GAIN"))
        if delta <= 0:
            continue
        for seed in SEEDS:
            r, error = verified_record(evidence, method, "GOAL", seed)
            if error or r is None:
                return dict(action="BLOCKED" if error else "RUN_IF_ADMITTED",
                            method=method, suite="GOAL", seed=seed,
                            reason=error, decisions=decisions)
    for seed in SEEDS:
        row = evidence.get("RA_IFA", {}).get("LIBERO50", {}).get(str(seed))
        if row is None:
            return dict(action="RUN_IF_ADMITTED", method="RA_IFA",
                        suite="LIBERO50", seed=seed, decisions=decisions)
        if (row.get("engineering") != "PASS" or not
                re.fullmatch(r"[0-9a-f]{64}", str(row.get("acceptance_sha256", "")))):
            return dict(action="BLOCKED", method="RA_IFA", suite="LIBERO50",
                        seed=seed, reason="Missing or failed acceptance",
                        decisions=decisions)
    return dict(action="ALL_ACCEPTED", decisions=decisions)


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as f:
        result = next_action(json.load(f))
    print(json.dumps(result, indent=2))
    sys.exit(2 if result["action"] == "BLOCKED" else 0)
