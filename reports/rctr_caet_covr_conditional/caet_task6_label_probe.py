"""Independent Task6 read-only label feasibility; does not write training labels."""
import json
from collections import Counter
from pathlib import Path
import h5py
import numpy as np
from caet_exp.labels import recorded_effects
from coreset_exp.common import sha
P=Path("/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019")
Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
b=json.loads((P/"bindings/task6.json").read_text())
assert sha(b["hdf5"])==b["hdf5_sha256"]
demos=sorted({x["demo_id"] for x in b["windows"].values() if x["identity_unique"]})
effects={}; audits={}
with h5py.File(b["hdf5"]) as h:
    for demo in demos:
        effects[demo],audits[demo]=recorded_effects(h["data"][demo],"butter")
counts=Counter()
for row in b["windows"].values():
    if not row["identity_unique"]:
        counts["ambiguous"]+=1
    elif row["target"]>=len(effects[row["demo_id"]]):
        counts["no_next_state"]+=1
    else:
        counts["valid"]+=1
        e=effects[row["demo_id"]][row["target"]]
        counts["moving"]+=bool(np.linalg.norm(e[:3])>=.001 or np.linalg.norm(e[3:])>=np.pi/180)
assert sha(b["hdf5"])==b["hdf5_sha256"]
out=dict(status="READ_ONLY_LABEL_MAPPING_PASS",counts=counts,demos=audits,
         candidate_windows=len(b["windows"]),source_hdf5_sha256=b["hdf5_sha256"],
         environment_steps=0,optimizer_updates=0,trained_effect_predictor=False,
         limitation="Temporal convention supported by official converter and actual joint/gripper alignment; historical simulator trajectory not replayed.")
(Q/"CAET_TASK6_LABEL_FEASIBILITY.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(dict(status=out["status"],counts=counts)))
