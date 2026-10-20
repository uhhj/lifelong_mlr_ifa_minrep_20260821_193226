"""Same XML, coordinates, actuator controls and calls; no robot/controller execution."""
import sys,json,hashlib,time
from pathlib import Path
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
sys.path.insert(0,str(W/"restoration_repair"))
import repair_probe as p
import mujoco,numpy as np,h5py
D=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020/provenance_diagnosis")
V=mujoco.__version__;assert V in ("3.2.3","2.3.5")
O=D/("physics_only_"+V);O.mkdir(exist_ok=False)
a=json.loads((W/"ASSETS.json").read_text());rows=[]
for path,h in a["input_hashes"].items():assert p.old.sha(path)==h
with h5py.File(a["hdf5"],"r") as f:
 for sel in a["chosen"]:
  d=sel["demo_id"];t=sel["target"];g=f["data"][d]
  xml=p.relocate(g.attrs["model_file"]);model=mujoco.MjModel.from_xml_string(xml);data=mujoco.MjData(model)
  state=g["states"][t];ref=g["states"][t+1]
  data.time=state[0];data.qpos[:]=state[1:1+model.nq];data.qvel[:]=state[1+model.nq:]
  controls=np.load(D/("private_controls_"+d+".npz"))["controls"]
  arrays={k:np.asarray(getattr(model,k)).copy() for k in
          ("body_mass","body_inertia","body_pos","body_quat","geom_pos","geom_quat","geom_size",
           "geom_friction","geom_solref","geom_solimp","dof_damping","dof_armature","mesh_vert")}
  for ctrl in controls:
   mujoco.mj_forward(model,data);data.ctrl[:]=ctrl;mujoco.mj_step(model,data)
  post=np.concatenate(([data.time],data.qpos,data.qvel))
  saved=D/("private_physics_323_"+d+".npz")
  row=dict(demo_id=d,version=V,substeps=len(controls),xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),
    control_hash=p.old.sha(D/("private_controls_"+d+".npz")),
    historical_max_error=p.old.err(post[1:],ref[1:]),historical_qpos_error=p.old.err(data.qpos,ref[1:1+model.nq]),
    compiled_hashes={k:hashlib.sha256(x.tobytes()).hexdigest() for k,x in arrays.items()})
  if V=="3.2.3":np.savez(saved,post=post,**arrays)
  else:
   prev=np.load(saved)
   row["versus_323_error"]=p.old.err(post[1:],prev["post"][1:])
   row["compiled_max_differences"]={k:p.old.err(x,prev[k]) if x.shape==prev[k].shape else "shape_changed" for k,x in arrays.items()}
  rows.append(row)
for path,h in a["input_hashes"].items():assert p.old.sha(path)==h
p.write(O/"RESULT.json",dict(status="PASS_DIAGNOSTIC_EXECUTION_NOT_ADMISSION",version=V,rows=rows,
 source_sha256=p.old.sha(__file__),inputs_unchanged=True,control_steps=3,physics_substeps=75,
 policy_continuations=0,optimizer_calls=0))
print(json.dumps(rows,indent=2))
