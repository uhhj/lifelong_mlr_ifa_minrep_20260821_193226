"""Add reviewed GOAL branches without changing any active OBJECT contract."""
from pathlib import Path
import hashlib
import json
import os
import time

Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
bundle=json.loads((Q/"GOAL_REGISTRY_BUNDLE.json").read_text())
old=json.loads((Q/"REGISTRY.json").read_text())
new=json.loads(bundle["REGISTRY.json"]["text"])
assert all(new[k]==v for k,v in old.items())
assert set(new)-set(old)=={f"{m}_GOAL_{s}" for m in ("RCTR","COVR") for s in (41,42,43)}
for name,item in bundle.items():
    raw=item["text"].encode()
    assert hashlib.sha256(raw).hexdigest()==item["sha256"]
    if name=="REGISTRY.json":continue
    target=Q/"entries"/name
    assert not target.exists()
    target.write_bytes(raw)
for key in set(new)-set(old):
    entry=new[key]
    for path,h in entry["deployment_hashes"].items():assert sha(Path(path))==h,path
    assert sha(Path(entry["launcher"]))==entry["launcher_sha256"]
    a=json.loads(Path(entry["baseline_acceptance"]).read_text())
    assert a["status"]=="PASS"
stamp=str(time.time_ns())
(Q/("REGISTRY.before_GOAL."+stamp+".json")).write_bytes((Q/"REGISTRY.json").read_bytes())
temp=Q/"REGISTRY.goal_install.tmp"
temp.write_text(json.dumps(new,indent=2)+"\n")
os.replace(temp,Q/"REGISTRY.json")
(Q/"GOAL_REGISTRY_ACCEPTANCE.json").write_text(json.dumps(dict(
    status="REGISTERED_NOT_EXECUTED",entries=len(new),added=sorted(set(new)-set(old)),
    preserved_entries=sorted(old),registry_sha256=sha(Q/"REGISTRY.json"),
    CAET="Not admitted: zero moving cross-task positive pairs at stage7",
    environment_steps=0,optimizer_updates=0,unix=time.time()),indent=2)+"\n")
print("REGISTERED",len(new))
