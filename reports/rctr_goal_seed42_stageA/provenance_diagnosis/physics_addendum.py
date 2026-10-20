"""Identify compiled model differences and finalize a new additive audit."""
import sys,json,hashlib,subprocess
from pathlib import Path
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
sys.path.insert(0,str(W/"restoration_repair"))
import repair_probe as p
import mujoco,h5py,numpy as np
assert mujoco.__version__=="2.3.5"
D=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020/provenance_diagnosis")
a=json.loads((W/"ASSETS.json").read_text())
with h5py.File(a["hdf5"],"r") as f:
 model=mujoco.MjModel.from_xml_string(p.relocate(f["data/demo_0"].attrs["model_file"]))
prev=np.load(D/"private_physics_323_demo_0.npz")
bodies=[]
for i in range(model.nbody):
 dm=float(model.body_mass[i]-prev["body_mass"][i])
 di=float(np.max(np.abs(model.body_inertia[i]-prev["body_inertia"][i])))
 if abs(dm)>1e-12 or di>1e-12:
  bodies.append(dict(name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,i),
   mass_235=float(model.body_mass[i]),mass_323=float(prev["body_mass"][i]),mass_difference=dm,
   inertia_max_abs_difference=di))
for path,h in a["input_hashes"].items():assert p.old.sha(path)==h
for version in ("3.2.3","2.3.5"):
 r=json.loads((D/("physics_only_"+version)/"RESULT.json").read_text())
 assert r["inputs_unchanged"] and r["control_steps"]==3 and r["physics_substeps"]==75
 assert r["source_sha256"]==p.old.sha(W/"provenance_diagnosis/physics_only.py")
assert int((D/"physics323.exit").read_text())==int((D/"physics235.exit").read_text())==0
procs=subprocess.check_output(["ps","-eo","pid,args"],text=True).splitlines()
workers=[l for l in procs if any(n in l for n in ("version_probe.py","legacy_snapshot_adapter.py","controller_phase_probe.py","physics_only.py","run_versions.sh"))]
assert not workers,workers
previous=json.loads((D/"FINAL_DIAGNOSTIC_AUDIT.json").read_text())
p.write(D/"FINAL_PHYSICS_ADDENDUM.json",dict(status="BLOCKED_HISTORICAL_CONTINUATION",
 compiled_body_differences=bodies,previous_audit_sha256=p.old.sha(D/"FINAL_DIAGNOSTIC_AUDIT.json"),
 additional_control_steps=6,total_this_round_engineering_control_steps=previous["engineering_steps"]+6,
 total_including_previous_rounds=644+previous["engineering_steps"]+6,
 physics323_exit=0,physics235_exit=0,no_remaining_workers=True,inputs_unchanged=True,
 policy_continuations=0,optimizer_calls=0))
bundle={}
for file in D.rglob("*"):
 if not file.is_file() or file.suffix not in (".json",".jsonl",".log",".exit",".pid",".started"):continue
 if file.name=="LIGHT_BUNDLE.json":continue
 raw=file.read_bytes();assert len(raw)<5*1024**2
 bundle[str(file.relative_to(D))]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),text=raw.decode())
(D/"LIGHT_BUNDLE.json").write_text(json.dumps(bundle)+"\n")
print(json.dumps(bodies,indent=2))
