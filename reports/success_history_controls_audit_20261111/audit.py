"""Local read-only evidence audit; no model, training, environment, or SSH calls."""
from pathlib import Path
import datetime
import hashlib
import json
import math
import re
import statistics

ROOT = Path(__file__).resolve().parent
WS = ROOT.parents[1]
BINDINGS = {}

def read(rel):
    p = WS / rel
    raw = p.read_bytes()
    BINDINGS[rel] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    return raw.decode("utf-8-sig")

def obj(rel):
    return json.loads(read(rel))

def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-9), (a, b)

def recompute(rel, arm):
    data = obj(rel)
    assert data["status"].startswith("PASS_")
    assert data["paired_episodes"] == 1200 and len(data["cases"]) == 6
    computed = {}
    for case, entry in data["cases"].items():
        rows = entry["paired_outcomes"]
        assert len(rows) == 200
        assert len({(r["stage"], r["task"], r["initial_index"]) for r in rows}) == 200
        computed[case] = {}
        for method in [arm, "S"]:
            cells = {}
            for stage in range(6, 10):
                for task in range(6, stage + 1):
                    selected = [r for r in rows if r["stage"] == stage and r["task"] == task]
                    assert sorted(r["initial_index"] for r in selected) == list(range(20))
                    assert all(isinstance(r[method], bool) for r in selected)
                    cells[stage, task] = sum(r[method] for r in selected) * 5.0
            metrics = {
                "AUC": statistics.mean(statistics.mean(cells[s, t] for s in range(t, 10)) for t in range(6, 10)),
                "FWT": statistics.mean(cells[t, t] for t in range(6, 10)),
                "OldFinal": statistics.mean(cells[9, t] for t in range(6, 9)),
                "FinalAll": statistics.mean(cells[9, t] for t in range(6, 10)),
            }
            for key, value in metrics.items():
                close(value, entry[method + "_metrics"][key])
            computed[case][method] = metrics
        if arm == "C":
            for b in entry["stage_budgets"].values():
                assert b["C_production"] == b["S_production"]
                assert b["C_isolated"] == b["S_isolated"]
    suites = {}
    for suite in ["GOAL", "OBJECT"]:
        suites[suite] = {}
        for method in [arm, "S"]:
            suites[suite][method] = {}
            for key in ["AUC", "FWT", "OldFinal", "FinalAll"]:
                value = statistics.mean(computed[f"{suite}_{seed}"][method][key] for seed in [41, 42, 43])
                close(value, data["suites"][suite][key][method + "_mean"])
                suites[suite][method][key] = value
    return data, suites

cdir = "execution/20261030_equal_optimizer_budget_cs_two_suites"
qdir = "execution/20261031_full_q_vs_latest_s_two_suites"
q1dir = "execution/20261109_q1_user_resume"
c_accept = obj(cdir + "/ACCEPTANCE.json")
q_accept = obj(qdir + "/ACCEPTANCE.json")
q1_accept = obj(q1dir + "/ACCEPTANCE.json")
assert c_accept["status"] == "ACCEPTED_SIX_C_EQUAL_OPTIMIZER_BUDGET"
assert q_accept["status"] == "PASS_COMPLETE_LOCAL_ACCEPTANCE_AND_INTERPRETATION"
assert q1_accept["status"] == "PASS_COMPLETE_TWO_SUITES_Q1_RERUN_ACCEPTANCE"
for base, receipt in [(cdir, c_accept), (qdir, q_accept), (q1dir, q1_accept)]:
    read(base + "/FINAL_REPORT.md")
    assert BINDINGS[base + "/FINAL_REPORT.md"]["sha256"] == receipt["report_sha256"]

cp = cdir + "/final_readonly_20261031_065009/PRIMARY_S_COMPARISON.json"
qp = qdir + "/final_readonly_20261101_032229/PRIMARY_S_COMPARISON.json"
cd, cm = recompute(cp, "C")
qd, qm = recompute(qp, "Q")
assert BINDINGS[cp]["sha256"] == c_accept["comparison_sha256"]
assert BINDINGS[qp]["sha256"] == q_accept["comparison_sha256"]
for case in cd["cases"]:
    def identity(r):
        return (r["stage"], r["task"], r["initial_index"], r["S"], r["initial_sha256"], r["noise_sha256"])
    assert sorted(map(identity, cd["cases"][case]["paired_outcomes"])) == sorted(map(identity, qd["cases"][case]["paired_outcomes"]))

sources = [
    "20261024_ccbp/PROTOCOL.md", "20261024_ccbp/losses.py", "20261024_ccbp/REPORT.md",
    "20261024_ccbp_v2_full/PROTOCOL.md", "20261024_ccbp_v2_full/REPORT.md",
    "20261025_ccbp_v3_full/PROTOCOL.md", "20261025_ccbp_v3_full/REPORT.md",
    "20261025_ccbp_v4_full/PROTOCOL.md", "20261025_ccbp_v4_full/FINAL_ANALYSIS.md",
    "20261025_ccbp_v4_full/FINAL_CLOSURE.json", "20261025_ccbp_v4_full/collect.py",
    "20261025_ccbp_v4_full/losses.py", "20261025_ccbp_v4_full/memory.py", "20261025_ccbp_v4_full/updates.py",
    "20261026_ccbp_v4_s_goal42_43/PROTOCOL.md", "20261026_ccbp_v4_s_goal42_43/REPORT.md",
    "20261026_ccbp_v4_s_object41_42_43/REPORT.md", "20261026_shbr_novelty_review/REPORT.md",
    "20261027_support_gate_forks/PROTOCOL.md", "20261027_support_gate_forks/REPORT.md",
    "20261028_support_gate_full_s_object/PROTOCOL.md", "20261029_support_gate_full_s_goal/PROTOCOL.md",
    "20261030_equal_optimizer_budget_cs_two_suites/PROTOCOL.md", "20261030_publication_readiness/REPORT.md",
    "20261031_full_q_vs_latest_s_two_suites/PROTOCOL.md", "20261031_publication_readiness/REPORT.md",
    "20261031_full_q_vs_latest_s_two_suites/engineering_startup_v2/GOAL_41/losses.py",
    "20261031_full_q_vs_latest_s_two_suites/engineering_startup_v2/GOAL_41/support_memory.py",
    "20261031_full_q_vs_latest_s_two_suites/engineering_startup_v2/GOAL_41/train.py",
    "20261106_debr_experiment_inventory/REPORT.md", "20261107_align_cqs_audit/REPORT.md",
    "20261108_support_calibration_ablation_review/REPORT.md", "20261109_debr_revised_ras_review/REPORT.md",
    "20261110_s_stripped_mlr_ifa_align/PROTOCOL.md", "20261110_s_stripped_mlr_ifa_align/REPORT.md",
    "20261110_reverse_order_main_protocol/PROTOCOL.md", "20261110_reverse_order_main_protocol/RUN_STATUS.json",
]
for rel in sources:
    read("execution/" + rel)
assert "不是采样动作" in read("execution/20261025_ccbp_v4_full/PROTOCOL.md")
assert "无S采集、成功保持缓存或保持损失" in read(cdir + "/PROTOCOL.md")
assert "各Q用自己的缓存，不能复制S缓存" in read(qdir + "/PROTOCOL.md")
assert "1120正式评估取消" in read("execution/20261027_support_gate_forks/REPORT.md").splitlines()[2]

directories = sorted(p for p in (WS / "execution").iterdir() if p.is_dir() and p != ROOT)
patterns = re.compile(r"直接回放|动作回放|普通.{0,8}蒸馏|同一.{0,8}缓存|相同.{0,8}轨迹|self.imitation|action.{0,15}replay|bc.replay|same.{0,10}cache", re.I)
scan = {"directories": len(directories), "markdown_files": 0, "python_files": 0, "matches": []}
for directory in directories:
    for extension, key in [("*.md", "markdown_files"), ("*.py", "python_files")]:
        for p in sorted(directory.glob(extension)):
            if any(s in p.name.lower() for s in ["credential", "session", "transport"]):
                continue
            scan[key] += 1
            content = p.read_text(encoding="utf-8-sig", errors="replace")
            lines = [i for i, line in enumerate(content.splitlines(), 1) if patterns.search(line)]
            if lines:
                scan["matches"].append({"path": p.relative_to(WS).as_posix(), "lines": lines,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})

result = {
    "status": "PASS_LOCAL_SUCCESS_HISTORY_CONTROL_EVIDENCE_AUDIT",
    "observed": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
    "scope": "Local accepted receipts, reports, protocols and source. Negative finding bounded to this workspace; no remote status refresh.",
    "conclusion": "Related ablations completed; no accepted same-success-cache direct-action-BC versus ordinary-distillation versus full-SHBR closed-loop comparison identified.",
    "new_training_updates": 0, "new_environment_episodes": 0, "new_model_forwards": 0,
    "recomputed_outcome_pairs": 2400, "unique_historical_S_outcomes": 1200,
    "recomputed_metrics": {"equal_optimizer_C_vs_S": cm, "Q_vs_S": qm},
    "q1_latest_acceptance": {k: q1_accept[k] for k in ["status", "observed", "formal_episodes", "training_collection_episodes"]},
    "search": scan, "source_bindings": BINDINGS,
}
(ROOT / "AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
print(json.dumps({k: result[k] for k in ["status", "observed", "recomputed_outcome_pairs", "recomputed_metrics"]}, ensure_ascii=False, indent=2))
print(json.dumps({"scan_counts": {k: v for k, v in scan.items() if k != "matches"}, "matched_files": len(scan["matches"]), "bound_sources": len(BINDINGS)}))
