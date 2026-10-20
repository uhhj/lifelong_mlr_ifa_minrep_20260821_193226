"""Read-only asset admission. No environment construction, policy forward or optimizer."""
import json,hashlib
from pathlib import Path
import h5py
P=Path("/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018")
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
binding=js(P/"bindings/task6.json");storage=js(P/"ra_coreset/storage/task6.json")
hp=Path(binding["hdf5"]);assert sha(hp)==binding["hdf5_sha256"]
eligible={}
for item in storage["selected"]:
 name=Path(item["source"]).name;r=binding["windows"][name]
 if r.get("identity_unique",True) and len(r["possible_starts"])==1:
  start=r["possible_starts"][0];t=r["target"];length=r["length"]
  if t==start+7 and start>=7 and length//4<=t<3*length//4:
   eligible.setdefault(r["demo_id"],[]).append(dict(sample_id=name,start=start,target=t,length=length,
     source=item["source"],sha256=r["sha256"]))
chosen=[]
for d in sorted(eligible,key=lambda x:int(x.split("_")[-1]))[:3]:
 rows=sorted(eligible[d],key=lambda r:(abs(r["target"]/(r["length"]-1)-.5),r["sample_id"]))
 chosen.append(dict(demo_id=d,**rows[0]))
assert len(chosen)==3
schemas={}
with h5py.File(hp,"r") as f:
 for r in chosen:
  g=f["data"][r["demo_id"]]
  ds={}
  g.visititems(lambda n,o:ds.update({n:dict(shape=list(o.shape),dtype=str(o.dtype))}) if isinstance(o,h5py.Dataset) else None)
  schemas[r["demo_id"]]=dict(datasets=ds,attribute_names=list(g.attrs.keys()),
    xml_sha256=hashlib.sha256(str(g.attrs["model_file"]).encode()).hexdigest())
files={str(hp):sha(hp),str(P/"bindings/task6.json"):sha(P/"bindings/task6.json"),
 str(P/"ra_coreset/storage/task6.json"):sha(P/"ra_coreset/storage/task6.json")}
for r in chosen:assert sha(r["source"])==r["sha256"];files[r["source"]]=r["sha256"]
report=dict(status="ASSET_ADMITTED_NOT_STATE_VALIDATED",hdf5=str(hp),chosen=chosen,schemas=schemas,input_hashes=files,
 selection="First three numeric demo IDs; eligible unique unpadded RA members in middle half; closest target to normalized 0.5, sample ID tie",
 optimizer_calls=0,environment_calls=0)
(W/"ASSETS.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
