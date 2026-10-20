"""Read-only checks and lightweight bundle for the version investigation."""
import json,hashlib,os,subprocess,datetime
from pathlib import Path
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
R=W/"provenance_diagnosis"
O=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020")
D=O/"provenance_diagnosis"
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assets=json.loads((W/"ASSETS.json").read_text())
for p,h in assets["input_hashes"].items():assert sha(p)==h,p
original=[]
for contract in (O/"restoration_repair/CONTRACT.json",O/"restoration_repair/integration_probe/CONTRACT.json"):
 for p,h in json.loads(contract.read_text())["files"].items():
  assert sha(p)==h,p
 original.append(dict(path=str(contract),sha256=sha(contract)))
versions={}
for name in ("mujoco_3.2.3","mujoco_2.3.5","mujoco_2.3.5_legacy_snapshot_recovery","controller_phase"):
 root=D/name;events=[json.loads(l) for l in (root/"attempts.jsonl").read_text().splitlines()]
 versions[name]=dict(steps_started=sum(x["event"]=="step_start" for x in events),
  steps_completed=sum(x["event"]=="step_complete" for x in events),
  resets=sum(x["event"]=="reset_complete" for x in events),closed=sum(x["event"]=="closed" for x in events),
  exceptions=sum(x["event"]=="exception" for x in events))
 assert versions[name]["closed"]==1
 assert versions[name]["steps_started"]==versions[name]["steps_completed"]
 for p,h in json.loads((root/"CONTRACT.json").read_text())["files"].items():assert sha(p)==h,p
exit_codes={p.name:int(p.read_text()) for p in D.glob("*.exit")}
assert exit_codes["probe323.exit"]==exit_codes["recovery235.exit"]==exit_codes["phase.exit"]==0
assert exit_codes["probe235.exit"]==1 and exit_codes["controller.exit"]==1
processes=subprocess.check_output(["ps","-eo","pid,args"],text=True).splitlines()
workers=[x for x in processes if any(n in x for n in ("version_probe.py","legacy_snapshot_adapter.py","controller_phase_probe.py","run_versions.sh"))]
assert not workers,workers
current=json.loads(subprocess.check_output(["/data/mlr_ifa/env/bin/python","-c",
 "import mujoco,json;print(json.dumps({'version':mujoco.__version__,'file':mujoco.__file__}))"],text=True))
assert current["version"]=="3.2.3"
libs={}
for root in (R/"mujoco235",Path("/data/mlr_ifa/env/lib/python3.8/site-packages/mujoco")):
 libs[str(root)]={str(p.relative_to(root)):sha(p) for p in root.rglob("*") if p.is_file() and
   (p.name.startswith("libmujoco") or p.name in ("METADATA","RECORD"))}
result=dict(status="BLOCKED_HISTORICAL_CONTINUATION",audit_time=datetime.datetime.now().astimezone().isoformat(),
 input_hashes_unchanged=True,input_hashes=assets["input_hashes"],preserved_contracts=original,
 attempts=versions,engineering_steps=sum(x["steps_completed"] for x in versions.values()),
 policy_continuations=0,optimizer_calls=0,exit_codes=exit_codes,no_remaining_workers=True,
 original_runtime=current,isolated_libraries=libs,
 state_threshold=1e-5,observation_threshold=1e-6,repeat_threshold=1e-10,
 reason="2.3.5 resolves fixed target transitions and 8-frame histories but demo_2 native next transition still fails; no 54 continuations.")
(D/"FINAL_DIAGNOSTIC_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
bundle={}
for p in D.rglob("*"):
 if not p.is_file() or p.suffix not in (".json",".jsonl",".log",".exit",".pid",".started"):continue
 if p.name=="LIGHT_BUNDLE.json":continue
 raw=p.read_bytes();assert len(raw)<5*1024**2
 bundle[str(p.relative_to(D))]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode())
(D/"LIGHT_BUNDLE.json").write_text(json.dumps(bundle)+"\n")
print(json.dumps({k:result[k] for k in ("status","engineering_steps","attempts","exit_codes","no_remaining_workers")},indent=2))
