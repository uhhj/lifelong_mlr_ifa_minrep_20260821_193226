"""Read metadata/coordinate alignment only; no environment, labels or optimization."""
import json
from pathlib import Path
import h5py
import numpy as np
P = Path("/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019")
c = json.loads((P/"resolved_ra_coreset.json").read_text())
tasks = json.loads((P/"TASKS.json").read_text())
print("TASKS", json.dumps(tasks)[:10000])
print("BINDING", json.dumps(json.loads((P/"bindings/task6.json").read_text()))[:6000])
for path in sorted(Path(c["folder"]).rglob("*.hdf5")):
    if not any(x in path.name for x in ("butter", "milk", "chocolate_pudding", "orange_juice")):
        continue
    with h5py.File(path) as f:
        g = f["data/demo_0"]
        st, joint = g["states"][:], g["obs/joint_states"][:]
        print("FILE", path, "keys", list(g), "obs", list(g["obs"]),
              "shapes", dict(states=st.shape, joint=joint.shape),
              "joint_shift_max", float(np.abs(st[1:,1:8]-joint[:-1]).max()),
              "attrs", {k:str(v)[:500] for k,v in f["data"].attrs.items()})
        xml = str(g.attrs["model_file"])
        print("XML", xml[:1200])
        print("JOINTS", [line.strip() for line in xml.splitlines() if "<joint" in line])

