"""Recompute downloaded admission evidence; no remote inference or environment."""
import hashlib
import json
from pathlib import Path
R = Path(__file__).resolve().parent
E = R/"rctr42_admission"


def read(name):
    return json.loads((E/name).read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


c, a, g = read("CONTRACT.json"), read("LAUNCH_ACCEPTANCE.json"), read("GPU_PREFLIGHT.json")
assert a["status"] == g["status"] == "PASS"
assert a["original_contract_sha256"] == sha(E/"CONTRACT.json")
assert a["preflight_sha256"] == sha(E/"GPU_PREFLIGHT.json")
assert g["optimizer_updates"] == g["formal_episodes"] == 0
assert g["engineering_steps"] == 28
assert all(sha(R/"source"/p) == h for p,h in c["source_sha256"].items())
gate = g["complete_snapshot_gate"]
assert gate["status"] == "PASS"
assert {x["history"] for x in gate["native_interface"]} == {1,4,8}
assert len(gate["repeat_checks"]) == 8
for r in gate["repeat_checks"]:
    assert r["physical"] <= 1e-10 and r["observations"] <= 1e-10
    assert r["gmm"] <= 1e-5 and r["action"] <= 1e-5
assert all(g[k] for k in ("native_to_brief_gmm_exact",
    "zero_record_loss_gradient_exact", "auxiliary_rng_restored",
    "auxiliary_synthetic_gradient_finite"))
out = dict(status="PASS", source_files=len(c["source_sha256"]),
    zero_optimizer_updates=True, engineering_steps=28,
    not_experimental_results=True, no_training_or_evaluation_added=True,
    limitation="Actual certified-record gradient admission occurs before first use, after collection")
(E/"LOCAL_ADMISSION_AUDIT.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out))
