"""Static state/controller provenance; no simulator construction."""
import json,hashlib,importlib.metadata,inspect
from pathlib import Path
import h5py
import numpy as np
import state_probe as p
a=json.loads((p.W/"ASSETS.json").read_text())
with h5py.File(a["hdf5"],"r") as f:
 meta={k:(json.loads(v) if isinstance(v,str) and v.startswith("{") else str(v)[:800])
   for k,v in f["data"].attrs.items() if k!="model_file"}
 align=[]
 for row in a["chosen"]:
  g=f["data"][row["demo_id"]];s=g["states"][:];j=g["obs/joint_states"][:];t=row["target"]
  align.append(dict(demo_id=row["demo_id"],state_joint_observation_same_index_max=float(np.abs(s[:,1:8]-j).max()),
    state_time_increment_min=float(np.diff(s[:,0]).min()),state_time_increment_max=float(np.diff(s[:,0]).max()),
    selected_action_range=[float(g["actions"][t-7:t+2].min()),float(g["actions"][t-7:t+2].max())]))
files={}
for path in [
 p.S/"libero/libero/envs/env_wrapper.py",
 Path("/data/mlr_ifa/env/lib/python3.8/site-packages/robosuite/environments/base.py"),
 Path("/data/mlr_ifa/env/lib/python3.8/site-packages/robosuite/models/grippers/panda_gripper.py"),
 Path("/data/mlr_ifa/env/lib/python3.8/site-packages/robosuite/controllers/osc.py"),
 Path("/data/mlr_ifa/env/lib/python3.8/site-packages/robosuite/robots/single_arm.py")]:
 if path.exists():files[str(path)]=p.sha(path)
versions={}
for name in ("mujoco","mujoco-py","robosuite","robomimic","numpy","torch"):
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]="not installed as distribution"
p.write(p.W/"STATIC_AUDIT.json",dict(hdf5_metadata=meta,time_alignment=align,versions=versions,source_sha256=files,
 caveat="MjSim flat state does not include all controller goals, gripper current_action, observables or solver warmstart. State-array restoration alone is insufficient."))
