"""Read-only asset/provenance inventory; output schemas and hashes, no state payload."""
import os,json,hashlib,subprocess
from pathlib import Path
import h5py,numpy as np
W=Path("/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020")
O=Path("/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020/provenance_diagnosis")
O.mkdir(exist_ok=False)
def sha(p):
 h=hashlib.sha256()
 with open(p,"rb") as f:
  for b in iter(lambda:f.read(8*1024**2),b""):h.update(b)
 return h.hexdigest()
assets=json.loads((W/"ASSETS.json").read_text());out={"input_hashes":assets["input_hashes"]}
for p,h in out["input_hashes"].items():assert sha(p)==h,p
out["search_roots"]=["/data","/root"];out["hdf5"]=[];out["npz_by_directory"]={};out["raw_candidates"]=[]
for root in out["search_roots"]:
 for dp,dn,fn in os.walk(root):
  dn[:]=[n for n in dn if n not in (".cache",".git","site-packages","node_modules",".ssh",".config")]
  for n in fn:
   p=Path(dp)/n
   if p.is_symlink():continue
   if n.endswith((".hdf5",".h5")):out["hdf5"].append(dict(path=str(p),bytes=p.stat().st_size))
   if n.endswith(".npz"):
    out["npz_by_directory"][dp]=out["npz_by_directory"].get(dp,0)+1
    if n.startswith("state_") or "/demonstrations/" in dp:out["raw_candidates"].append(str(p))
out["dataset"]={"path":assets["hdf5"],"sha256":sha(assets["hdf5"]),"demos":{}}
with h5py.File(assets["hdf5"],"r") as f:
 out["dataset"]["data_attributes"]={k:str(v) for k,v in f["data"].attrs.items() if k not in ("bddl_file_content",)}
 for d in [v["demo_id"] for v in assets["chosen"]]:
  g=f["data"][d];schema={}
  def visit(name,obj):
   if isinstance(obj,h5py.Dataset):schema[name]=dict(shape=list(obj.shape),dtype=str(obj.dtype))
  g.visititems(visit)
  xml=str(g.attrs["model_file"])
  out["dataset"]["demos"][d]=dict(schema=schema,attribute_names=list(g.attrs.keys()),
   xml_sha256=hashlib.sha256(xml.encode()).hexdigest(),first_state_time=float(g["states"][0,0]),
   init_state_equals_first=bool(np.array_equal(g.attrs["init_state"],g["states"][0])))
for name in ("manifest.json","verified.json"):
 p=Path("/data/mlr_ifa/datasets")/name
 if p.exists():out[name]=json.loads(p.read_text())
out["processes"]=subprocess.check_output(["ps","-eo","pid,args"],text=True)
out["processes"]=[l for l in out["processes"].splitlines() if "RCTR" in l]
out["gpu"]=subprocess.check_output(["nvidia-smi","--query-gpu=name,memory.used,utilization.gpu","--format=csv,noheader"],text=True)
out["disk"]=subprocess.check_output(["df","-h","/data","/root"],text=True)
(O/"INVENTORY.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(dict(hdf5_files=len(out["hdf5"]),npz_directories=len(out["npz_by_directory"]),raw_candidates=out["raw_candidates"],gpu=out["gpu"],disk=out["disk"])))
