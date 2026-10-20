"""Independent lightweight evidence acceptance; no torch/models/environment."""
from pathlib import Path
from collections import Counter
import json,hashlib,sys
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);E=D/"evidence"
js=lambda p:json.loads(p.read_text())
jl=lambda p:[json.loads(l) for l in p.read_text().splitlines()]
manifest=js(D/"DOWNLOAD_MANIFEST.json")
for n,m in manifest.items():
    b=(E/n).read_bytes();assert len(b)==m["bytes"] and hashlib.sha256(b).hexdigest()==m["sha256"]
assert js(E/"ADMISSION.json")["status"]=="PASS"
assert js(E/"PREFLIGHT.json")["optimizer_calls"]==4 and js(E/"PREFLIGHT.json")["environment_steps"]==0
assert js(E/"NOISE_INTERFACE.json")["status"]=="PASS"
assert len(jl(E/"preflight_calls.jsonl"))==4
binding=js(E/"BINDING.json");assert len(set(binding["initial_hashes"]))==40
source=[]
for f,h in js(E/"CONTRACT.json")["files"].items():
    prefix="/data/mlr_ifa/experiment_workspaces/SAZR_OBJECT_RA_seed41_task6_20261020/"
    if f.startswith(prefix) and not f.endswith("language.npz") and not f.endswith("env.sh"):
        p=R/f[len(prefix):]
        assert p.exists() and hashlib.sha256(p.read_bytes()).hexdigest()==h,p
        source.append(str(p.relative_to(R)))
counts={};records={}
for n,ids in (("anchor",range(20,30)),("validation_parent",range(30,40)),("validation_candidate",range(30,40)),("formal",range(20))):
    f=E/n/"episodes.jsonl";rs=jl(f) if f.exists() else []
    records[n]=rs;counts[n]=len(rs)
    assert len({r["initial_index"] for r in rs})==len(rs)
    for r in rs:assert r["initial_index"] in ids and r["initial_sha256"]==binding["initial_hashes"][r["initial_index"]]
    if (E/n/"COMPLETE.json").exists():assert len(rs)==len(ids)
final=(E/"PROCESS_EXIT_ACCEPTANCE.json").exists()
out=dict(status="PASS_FINAL" if final else "PASS_PARTIAL",episodes=counts,source_files_matched=len(source),
    package_files=len(manifest),no_new_training_or_evaluation=True)
if final:
    assert not js(D/"SNAPSHOT.json")["workers"]
    assert sum(counts.values())==50
    for phase,n in counts.items():
        events=Counter(r["event"] for r in jl(E/phase/"attempts.jsonl"))
        assert events==Counter(environment_construct_start=1,environment_construct_complete=1,environment_closed=1,vector_start=n//10,vector_complete=n//10)
    a,b=records["validation_parent"],records["validation_candidate"]
    pairs=[]
    for x,y in zip(a,b):
        for k in ("initial_index","initial_sha256","noise_seed","noise_sha256","env_seed","parent_sha256"):assert x[k]==y[k]
        pairs.append(("S" if x["success"] else "F")+("S" if y["success"] else "F"))
    c=Counter(pairs);enabled=c["FS"]>=2 and c["SF"]==0
    assert enabled==js(E/"DECISION.json")["enabled"]==js(E/"AUDIT.json")["enabled"]
    assert all(r["enabled"]==enabled for r in records["formal"])
    assert len(js(E/"epochs.json"))==10 and js(E/"CALIBRATED.json")["steps"]==1570
    assert len(jl(E/"STRUCTURE.jsonl"))==11 and all(r["passed"] for r in jl(E/"STRUCTURE.jsonl"))
    assert js(E/"AUDIT.json")["checkpoint_load"]=="strict PASS"
    for phase in ("pipeline","supervisor","anchor","anchors","train","validation_parent","validation_candidate","decide","formal","audit"):
        assert (E/(phase+".exit")).read_text().strip()=="0"
    out.update(enabled=enabled,validation_counts=dict(c),successes={n:sum(r["success"] for r in rs) for n,rs in records.items()},
        formal_percent=5*sum(r["success"] for r in records["formal"]),
        historical_parent_percent=60,delta_parent_pp=5*sum(r["success"] for r in records["formal"])-60)
(D/"LOCAL_ACCEPTANCE.json").write_text(json.dumps(out,indent=2)+"\n")
if final:
    st=js(R/"RUN_STATUS.json");st["status"]="COMPLETED";st["acceptance"]=out
    (R/"RUN_STATUS.json").write_text(json.dumps(st,indent=2)+"\n")
print(json.dumps(out,indent=2))
