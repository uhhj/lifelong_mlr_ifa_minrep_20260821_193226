"""Post-gate read-only temporal alignment diagnostics, not additional simulation."""
import json,hashlib
from pathlib import Path
import h5py,numpy as np
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
a=json.loads((W/"ASSETS.json").read_text())
rows=[]
with h5py.File(a["hdf5"],"r") as f:
 for selected in a["chosen"]:
  d=selected["demo_id"];g=f["data"][d];states=g["states"][:];joint=g["obs/joint_states"][:]
  r=dict(demo_id=d,shift_checks=[],flat_state_dimensions=int(states.shape[1]))
  for shift in (-2,-1,0,1,2):
   idx=np.arange(2,len(states)-2);delta=states[idx,1:8]-joint[idx+shift]
   r["shift_checks"].append(dict(observation_index_minus_state_index=shift,
     max_joint_error=float(np.abs(delta).max()),mean_abs_joint_error=float(np.abs(delta).mean())))
  r["robot_states_vs_observation_joint"]=float(np.abs(g["robot_states"][:,:7]-joint).max())
  rows.append(r)
assert hashlib.sha256(Path(a["hdf5"]).read_bytes()).hexdigest()==a["input_hashes"][a["hdf5"]]
(W/"ALIGNMENT_DIAGNOSIS.json").write_text(json.dumps(dict(rows=rows,environment_calls=0,
  conclusion="Diagnostic shifts do not authorize reindexing training data or relaxing the restoration gate."),indent=2)+"\n")
