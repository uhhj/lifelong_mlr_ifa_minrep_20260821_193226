"""Read-only final repair gate: no threshold relaxation and no branch launch."""
import json,hashlib,time,subprocess
from pathlib import Path
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
R=W/"restoration_repair"
BASE=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020")
O=BASE/"restoration_repair"
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+"\n")
for name in ("run","integration"):assert (R/(name+".exit")).read_text().strip()=="0"
ps=subprocess.check_output(["ps","-eo","pid,args"],text=True)
workers=[s for s in ps.splitlines() if any(("python "+str(R)+"/"+n) in s
 for n in ("repair_probe.py","integration_probe.py"))]
assert not workers,workers
oldbundle=js(W/"LIGHT_BUNDLE.json")
for rel,v in oldbundle.items():
 parts=Path(rel).parts;p=(W if parts[0]=="workspace" else BASE).joinpath(*parts[1:])
 assert sha(p)==v["sha256"],("Original failure evidence changed",str(p))
for out in (O,O/"integration_probe"):
 for path,h in js(out/"CONTRACT.json")["files"].items():assert sha(Path(path))==h,path
j=js(O/"REPAIR_DIAGNOSTIC.json");k=js(O/"integration_probe/RESULT.json")
assert j["status"]=="DIAGNOSTIC_COMPLETE_NOT_BRANCH_ADMISSION"
assert k["status"]=="ENGINEERING_DIAGNOSTIC_COMPLETE"
events=[]
for folder in (O,O/"integration_probe"):
 events.extend(json.loads(line) for line in (folder/"attempts.jsonl").read_text().splitlines())
counts={key:sum(x["event"]==key for x in events) for key in set(x["event"] for x in events)}
assert counts["step_start"]==counts["step_complete"]==328
assert counts["reset_start"]==counts["reset_complete"]==42
assert counts["construct_start"]==counts["construct_complete"]==counts["closed"]==2
assert not counts.get("exception")
rows=[]
for r,s in zip(j["rows"],k["rows"]):
 assert r["demo_id"]==s["demo_id"]
 obs_joint=max(v["obs"]["joint_states"] for v in r["corrected_observations"])
 obs_grip=max(v["obs"]["gripper_states"] for v in r["corrected_observations"])
 obs_ee=max(v["obs"]["ee_pos"] for v in r["corrected_observations"])
 aligned=obs_joint<=1e-6 and obs_grip<=1e-6
 all_lowdim=aligned and obs_ee<=1e-6
 repeat=max(s["snapshot_repeat_error"],s["snapshot_observation_repeat_error"],
   max(v["restore_error"] for v in s["snapshot_trials"]))
 history=s["forced_prefix"][-1]
 assert aligned and repeat<=1e-10
 assert history["max_error"]>1e-5
 rows.append(dict(demo_id=r["demo_id"],joint_error=obs_joint,gripper_error=obs_grip,ee_error=obs_ee,
   joint_gripper_alignment_pass=aligned,all_lowdim_observation_pass=all_lowdim,
   full_snapshot_repeat_error=repeat,full_snapshot_repeat_pass=repeat<=1e-10,
   recorded_transition_error=history["max_error"],recorded_transition_pass=history["max_error"]<=1e-5,
   recorded_transition_qpos=history["qpos_error"],recorded_transition_qvel=history["qvel_error"],
   largest_errors=history["largest"]))
result=dict(status="BLOCKED_HISTORICAL_FIDELITY",audit_status="PASS_FOR_BLOCKED_REPORT",
 time=time.time(),rows=rows,counts=counts,previous_engineering_steps=316,
 current_engineering_steps=328,total_engineering_steps=644,
 policy_continuations=0,authorized_max_policy_continuations=54,optimizer_calls=0,
 thresholds_unchanged=True,original_failure_evidence_unchanged=True,inputs_unchanged=True,
 exits=dict(repair=0,integration=0),workers=workers,
 reason="Same-state full-snapshot repeatability and joint/gripper timing fixed; recorded transition and full observed-state fidelity remain above original thresholds. No continuation authorized by the unchanged gate.")
write(O/"FINAL_REPAIR_ACCEPTANCE.json",result)
write(O/"RUN_STATUS.json",result)
files={}
for name in ("CONTRACT.json","REPAIR_DIAGNOSTIC.json","attempts.jsonl","FINAL_REPAIR_ACCEPTANCE.json",
 "RUN_STATUS.json","integration_probe/CONTRACT.json","integration_probe/RESULT.json","integration_probe/attempts.jsonl"):
 files["output/"+name]=O/name
for name in ("run.log","run.exit","started","integration.log","integration.exit","integration.started"):
 files["workspace/"+name]=R/name
bundle={name:dict(bytes=p.stat().st_size,sha256=sha(p),text=p.read_text()) for name,p in files.items()}
write(R/"LIGHT_BUNDLE.json",bundle)
print(json.dumps(result,indent=2))
