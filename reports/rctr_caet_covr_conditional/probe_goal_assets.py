"""Read-only GOAL admission resources, no policy loading or environment steps."""
from pathlib import Path
import json
import hashlib
import h5py
import numpy as np
import xml.etree.ElementTree as ET


def sha(p):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
    return h.hexdigest()


result={}
for seed in (41,42,43):
    date="20261018" if seed==42 else "20261019"
    root=Path("/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%d_%s"%(seed,date))
    c=json.loads((root/"resolved_ra_coreset.json").read_text())
    row=dict(root=str(root),seed=c["seed"],benchmark=c["benchmark_name"],gamma=c["gamma"],
        pretrain=c["pretrain"],acceptances={},source_binding=json.loads((root/"PARENT_BINDING.json").read_text()))
    for p in root.glob("*ACCEPTANCE*.json"):
        row["acceptances"][p.name]=dict(status=json.loads(p.read_text()).get("status"),sha256=sha(p))
    row["tasks"]={}
    for task in range(6,10):
        b=json.loads((root/"bindings"/("task%d.json"%task)).read_text())
        assert sha(b["hdf5"])==b["hdf5_sha256"]
        # This probe only locates physical joint layouts; identity certification
        # is performed separately with complete action-window matching.
        demos=sorted({v["demo_id"] for v in b["windows"].values() if v.get("demo_id")})
        with h5py.File(b["hdf5"]) as f:
            d=f["data"][demos[0]]
            xml=d.attrs["model_file"]
            if isinstance(xml,bytes):xml=xml.decode()
            tree=ET.fromstring(xml)
            joints=[]
            for body in tree.find("worldbody").iter("body"):
                js=[dict(j.attrib) for j in body if j.tag in ("joint","freejoint")]
                if js:joints.append(dict(body=body.attrib,joints=js))
            row["tasks"][str(task)]=dict(hdf5=b["hdf5"],sha256=b["hdf5_sha256"],
                windows=len(b["windows"]),example_demo=demos[0],state_shape=list(d["states"].shape),
                joints=joints)
    result[str(seed)]=row
print(json.dumps(result,indent=2))
