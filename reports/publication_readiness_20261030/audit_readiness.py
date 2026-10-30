"""Recompute existing scalar/outcome evidence; no model, server or environment calls."""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict
import csv
import datetime
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
SOURCES = {}


def fingerprint(path):
    raw = path.read_bytes()
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read(path):
    path = Path(path)
    if not path.is_absolute():
        path = WORK / path
    SOURCES[path.relative_to(WORK).as_posix()] = fingerprint(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def bind(path, expected):
    path = Path(path)
    if not path.is_absolute():
        path = WORK / path
    value = fingerprint(path)
    assert value["sha256"] == expected, str(path)
    SOURCES[path.relative_to(WORK).as_posix()] = value


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9), (a, b)


def metrics(r):
    return {
        "AUC": sum(sum(r[s, t] for s in range(t, 10)) / (10-t) for t in range(6, 10)) / 4,
        "FWT": sum(r[t, t] for t in range(6, 10)) / 4,
        "NBT": sum(sum(r[t, t]-r[s, t] for s in range(t+1, 10)) / (9-t) for t in range(6, 9)) / 3,
        "OldFinal": sum(r[9, t] for t in range(6, 9)) / 3,
        "FinalAll": sum(r[9, t] for t in range(6, 10)) / 4,
    }


def main():
    folders = {
        "GOAL": "execution/20261029_support_gate_full_s_goal/final_readonly_20261030_111546",
        "OBJECT": "execution/20261028_support_gate_full_s_object/final_readonly_20261029_201235",
    }
    result = {}
    cbase = WORK / "execution/20261027_c_goal42_43_object41_42_43/final_readonly_20261027_222119"
    for suite, folder in folders.items():
        base = WORK / folder
        pair = read(base / "PRIMARY_C_COMPARISON.json")
        audit = read(base / "LOCAL_FINAL_AUDIT.json")
        summary = read(base / "RESULT_SUMMARY.json")
        state = read(base.parent / "RUN_STATUS.json")
        assert audit["status"] == "PASS_COMPLETE_LOCAL_ACCEPTANCE"
        assert state["status"] == "COMPLETE_ACCEPTED"
        assert pair["paired_episodes"] == 600 and len(pair["paired_outcomes"]) == 600
        bind(base / "EVIDENCE.json", audit["evidence_sha256"])
        bind(base / "LOCAL_FINAL_AUDIT.json", summary["final_audit_sha256"])
        if suite == "GOAL":
            for path, digest in pair["sources"].items():
                p = Path(path)
                bind(p if p.is_absolute() else base.parent / p, digest)
        else:
            for kind, directory in [("C", cbase), ("S", base)]:
                bind(directory / "LOCAL_FINAL_AUDIT.json", pair[kind + "_final_audit_sha256"])
                bind(directory / "EVIDENCE.json", pair[kind + "_evidence_sha256"])
        cells = defaultdict(list)
        identities = set()
        for row in pair["paired_outcomes"]:
            key = (str(row["seed"]), row["stage"], row["task"], row["initial_index"])
            assert key not in identities
            identities.add(key)
            assert len(row["initial_sha256"]) == len(row["noise_sha256"]) == 64
            assert type(row["C_success"]) is bool and type(row["S_success"]) is bool
            cells[key[:3]].append(row)
        assert len(cells) == 30
        by_seed = {}
        negatives = []
        for seed in ["41", "42", "43"]:
            matrices = {"C": {}, "S": {}}
            for stage in range(6, 10):
                for task in range(6, stage+1):
                    rows = cells[seed, stage, task]
                    assert sorted(x["initial_index"] for x in rows) == list(range(20))
                    old = pair["seeds"][seed]["cells"][f"{stage}_{task}"]
                    for arm in matrices:
                        matrices[arm][stage, task] = F(sum(x[arm+"_success"] for x in rows), 20)
                        near(matrices[arm][stage, task]*100, old[arm+"_percent"])
                    sf = sum(x["C_success"] and not x["S_success"] for x in rows)
                    fs = sum(not x["C_success"] and x["S_success"] for x in rows)
                    assert sf == old["C_success_S_fail"] and fs == old["C_fail_S_success"]
                    if fs < sf:
                        negatives.append({"seed": seed, "cell": f"{stage}_{task}", **old})
            by_seed[seed] = {}
            for arm, matrix in matrices.items():
                mm = metrics(matrix)
                for k, v in mm.items():
                    near(v*100, pair["seeds"][seed][arm+"_metrics"][k]["percent"])
                by_seed[seed][arm] = {k: float(v*100) for k, v in mm.items()}
                by_seed[seed][arm]["FinalNew"] = float(matrix[9, 9]*100)
            by_seed[seed]["delta_pp"] = {k: by_seed[seed]["S"][k]-by_seed[seed]["C"][k]
                                              for k in by_seed[seed]["C"]}
        assert negatives == pair["negative_cells"]
        means = {arm: {k: statistics.mean(by_seed[s][arm][k] for s in by_seed)
                       for k in by_seed["41"][arm]} for arm in ["C", "S"]}
        for arm in means:
            for k in pair["means"][arm]:
                near(means[arm][k], pair["means"][arm][k])
        for k, val in summary["statistics"].items():
            near(means["S"][k], val["mean"])
            near(statistics.stdev(by_seed[s]["S"][k] for s in by_seed), val["sample_sd"])
        costs = pair["costs"]
        result[suite] = {"status": "PASS_EXISTING_OUTCOMES_RECOMPUTED", "paired_episodes": 600,
                         "by_seed": by_seed, "means": means,
                         "delta_pp": {k: means["S"][k]-means["C"][k] for k in means["C"]},
                         "negative_cells": negatives, "costs": costs,
                         "optimizer_ratio": costs["S_production_optimizer"] / costs["C_production_optimizer"]}
    fork = read("execution/20261027_support_gate_forks/RUN_STATUS.json")
    assert fork["status"] == "USER_STOPPED_PARTIAL_PRESERVED" and fork["formal_episodes"] == 0
    old_path = WORK / "execution/20261026_shbr_novelty_review/LATEST_METRICS.csv"
    SOURCES[old_path.relative_to(WORK).as_posix()] = fingerprint(old_path)
    with old_path.open(encoding="utf-8-sig", newline="") as f:
        old_rows = list(csv.DictReader(f))
    original_s = {r["suite"]: {k: float(r[k]) for k in ["AUC", "FWT", "NBT", "OldFinal", "FinalAll"]}
                  for r in old_rows if r["method"] == "S"}
    correction = read(WORK / folders["OBJECT"] / "PRIMARY_C_COMPARISON.json")["seeds"]["43"]["cells"]["9_8"]
    assert correction["C_percent"] == correction["S_percent"] == 90
    for rel in [
        "execution/20261029_support_gate_full_s_goal/PROTOCOL.md",
        "execution/20261029_support_gate_full_s_goal/FINAL_REPORT.md",
        "execution/20261028_support_gate_full_s_object/PROTOCOL.md",
        "execution/20261028_support_gate_full_s_object/FINAL_REPORT.md",
        "execution/20261028_support_gate_full_s_object/final_readonly_20261029_201235/PRIMARY_C_COMPARISON.md",
        "execution/20261027_support_gate_forks/PROTOCOL.md",
        "execution/20261027_c_goal42_43_object41_42_43/s_vs_c_review_20261027/REPORT.md",
        "execution/20261026_shbr_novelty_review/SUBMISSION_STRATEGY_20261027.md",
    ]:
        SOURCES[rel] = fingerprint(WORK / rel)
    output = {"status": "PASS_LOCAL_PUBLICATION_READINESS_EVIDENCE_REVIEW",
              "observed_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
              "scope": "Existing local outcome/scalar recomputation only; does not repeat CPU model acceptance or remote audits",
              "source_bindings": SOURCES, "results": result, "original_s_reference": original_s,
              "canceled_mechanism": {"status": fork["status"], "formal_episodes": 0, "not_resumed": True},
              "report_correction": {"suite": "OBJECT", "seed": 43, "cell": "9_8",
                                    "incorrect_prose_C_percent": 85, "verified_C_percent": 90,
                                    "verified_S_percent": 90, "delta_pp": 0,
                                    "note": "Prior prose preserved; underlying comparison JSON and metrics already correct"},
              "remote_commands": 0, "model_forwards": 0, "training_updates": 0, "new_episodes": 0}
    (HERE / "EVIDENCE_REVIEW.json").write_text(json.dumps(output, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "outcome_pairs": 1200,
                      "deltas": {k: v["delta_pp"] for k, v in result.items()},
                      "sources": len(SOURCES)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
