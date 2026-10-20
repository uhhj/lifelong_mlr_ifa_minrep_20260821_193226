"""Exit/input/attempt audit and whitelisted metadata packaging; zero environment calls."""
import json,time,hashlib,subprocess
from pathlib import Path
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
O=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020")
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert (W/"probe.exit").read_text().strip()=="1"
assert (W/"recovery.exit").read_text().strip()=="0"
for label in ("probe","recovery"):
 pid=int((W/(label+".pid")).read_text())
 result=subprocess.run(["ps","-p",str(pid),"-o","stat="],stdout=subprocess.PIPE,text=True)
 assert not result.stdout.strip() or result.stdout.strip().startswith("Z"),(pid,result.stdout)
ps=subprocess.check_output(["ps","-eo","pid,args"],text=True)
workers=[line for line in ps.splitlines() if str(W) in line and
 any(("python "+str(W)+"/"+name) in line for name in ("state_probe.py","recover_asset_paths.py"))]
assert not workers,workers
r=js(O/"recovery_asset_paths/STATE_PROBE.json")
assert r["status"]=="BLOCKED_STATE_RESTORATION"
assert len(r["rows"])==3
events=[]
for folder in (O,O/"recovery_asset_paths"):
 for line in (folder/"state_attempts.jsonl").read_text().splitlines():events.append(json.loads(line))
counts={key:sum(x["event"]==key for x in events) for key in set(x["event"] for x in events)}
assert counts["engineering_step_start"]==counts["engineering_step_complete"]==316
assert counts["environment_construct_start"]==counts["environment_construct_complete"]==counts["environment_closed"]==2
assert counts["reset_start"]==counts["reset_complete"]==42
for folder in (O,O/"recovery_asset_paths"):
 for name,h in js(folder/"RESTORATION_CONTRACT.json")["files"].items():assert sha(Path(name))==h,name
audit=dict(status="PASS_BLOCKED_SCIENTIFIC_ADMISSION",experiment_status="BLOCKED_STATE_RESTORATION",
 original_exit=1,recovery_diagnostic_exit=0,worker_processes=workers,counts=counts,
 optimizer_calls=0,policy_continuations=0,authorized_max_continuations=54,
 frozen_inputs_unchanged=True,time=time.time())
(O/"PROCESS_EXIT_ACCEPTANCE.json").write_text(json.dumps(audit,indent=2)+"\n")
files={}
for name in ("ASSETS.json","STATIC_AUDIT.json","ALIGNMENT_DIAGNOSIS.json","RESOURCE_RECOVERY.json","probe.log","probe.exit","probe.started",
             "recovery.log","recovery.exit","recovery.started"):
 files["workspace/"+name]=W/name
for name in ("RESTORATION_CONTRACT.json","STATE_PROBE_FAILURE.json","state_attempts.jsonl","PROCESS_EXIT_ACCEPTANCE.json",
             "recovery_asset_paths/RESTORATION_CONTRACT.json","recovery_asset_paths/STATE_PROBE.json",
             "recovery_asset_paths/state_attempts.jsonl"):
 files["output/"+name]=O/name
bundle={name:dict(sha256=sha(p),bytes=p.stat().st_size,text=p.read_text()) for name,p in files.items()}
(W/"LIGHT_BUNDLE.json").write_text(json.dumps(bundle,indent=2)+"\n")
print(json.dumps(audit,indent=2))
