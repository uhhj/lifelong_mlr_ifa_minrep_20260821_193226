"""Read saved state/RNG/config/outcomes only. No policy imports, forward, CUDA calls or env."""
from pathlib import Path
import json,hashlib,time,math
import numpy as np
import torch
P=Path("/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed42_20261019")
O=Path("/data/mlr_ifa/experiments/OVR_OBJECT_RA_seed42_screen_20261020")
S=Path("/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019/seed42/source")
A=Path("/data/mlr_ifa/analyses/OVR_OBJECT_seed42_history_readonly_20261020")
A.mkdir(exist_ok=False)
inputs={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def track(p):inputs[str(p)]=sha(p);return Path(p)
def js(p):return json.loads(track(p).read_text())
def rows(p):return [json.loads(l) for l in track(p).read_text().splitlines() if l.strip()]
def write(n,v):(A/n).write_text(json.dumps(v,indent=2,allow_nan=False)+"\n")
def norm(x):
    if torch.is_tensor(x):return dict(kind="tensor",dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
    if isinstance(x,np.ndarray):return dict(kind="numpy",dtype=str(x.dtype),shape=list(x.shape),sha256=hashlib.sha256(x.tobytes()).hexdigest())
    if isinstance(x,(list,tuple)):return [norm(a) for a in x]
    if isinstance(x,dict):return {k:norm(v) for k,v in x.items()}
    if isinstance(x,np.generic):return x.item()
    return x
def component_hash(x):return hashlib.sha256(json.dumps(norm(x),sort_keys=True,separators=(",",":")).encode()).hexdigest()
checks={};rr={}
for stage in (7,9):
    rs=rows(P/("ra_coreset/evaluation/trials_stage%d_task6.jsonl"%stage))
    at=rows(P/("ra_coreset/evaluation/attempts_stage%d_task6.jsonl"%stage))
    f=track(P/("ra_coreset/models/mlrifa_%d_model_ep99.pth"%stage))
    assert len(rs)==20 and all(r["checkpoint_sha256"]==sha(f) for r in rs)
    checks["historical%d"%stage]=dict(episodes=rs,attempts=at,
        checkpoint_sha256=sha(f),successes=sum(r["success"] for r in rs))
    path=track(P/("ra_coreset/evaluation/rng_before_stage%d_task6.pth"%stage))
    rr["historical%d"%stage]=torch.load(path,map_location="cpu")
for phase in ("parent9","teacher7"):
    rs=rows(O/phase/"episodes.jsonl");at=rows(O/phase/"attempts.jsonl")
    checks[phase]=dict(episodes=rs,attempts=at,successes=sum(r["success"] for r in rs))
    rr[phase]=torch.load(track(O/phase/"rng_before.pth"),map_location="cpu")
    checks[phase]["rng_metadata"]=js(O/phase/"RNG_METADATA.json")
summaries={}
for label,r in rr.items():
    summaries[label]=dict(keys=sorted(r),components={k:component_hash(v) for k,v in r.items()},
        cuda_state_sizes=[int(v.numel()) for v in r.get("cuda",[])],
        cpu_state_size=int(r["cpu"].numel()))
pairs=[]
for a,b in (("historical7","historical9"),("historical7","teacher7"),("historical9","parent9"),("parent9","teacher7")):
    pairs.append(dict(a=a,b=b,components_equal={k:summaries[a]["components"][k]==summaries[b]["components"][k] for k in rr[a]}))
write("RNG_STATE_AUDIT.json",dict(summaries=summaries,pairs=pairs,
    scope="Hash semantic components of saved pre-environment snapshots, not pickle file bytes; no RNG draws or policy execution."))
configs={"historical":js(P/"ra_coreset/evaluation/resolved_actual.json"),
         "parent9":js(O/"parent9/resolved_actual.json"),"teacher7":js(O/"teacher7/resolved_actual.json")}
diffs=[]
def diff(a,b,path="",pair=""):
    if isinstance(a,dict) and isinstance(b,dict):
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:diffs.append(dict(pair=pair,path=path+"/"+k,a=a.get(k),b=b.get(k)))
            else:diff(a[k],b[k],path+"/"+k,pair)
    elif a!=b:diffs.append(dict(pair=pair,path=path,a=a,b=b))
for phase in ("parent9","teacher7"):diff(configs["historical"],configs[phase],pair="historical->"+phase)
write("CONFIG_AUDIT.json",dict(configs=configs,differences=diffs))
init=Path("/data/mlr_ifa/src/libero/libero/init_files/libero_object/pick_up_the_butter_and_place_it_in_the_basket.pruned_init")
states=np.asarray(torch.load(track(init),map_location="cpu"))
hs=[hashlib.sha256(v.tobytes()).hexdigest() for v in states]
assert len(set(hs))==50
for label,info in checks.items():
    for r in info["episodes"]:
        idx=r.get("initial_index",r.get("initial_state_index"))
        h=r.get("initial_sha256",r.get("initial_state_sha256"))
        assert hs[idx]==h
assert not set(hs[:20])&set(hs[20:30])
std=states.std(axis=0);vary=np.flatnonzero(std>1e-10)
normalized=(states[:,vary]-states[:,vary].mean(axis=0))/std[vary]
dist=np.linalg.norm(normalized[:30,None,:]-normalized[None,:30,:],axis=2)/np.sqrt(len(vary))
neighbors=[]
for i in range(20,30):
    j=int(np.argmin(dist[i,:20]))
    neighbors.append(dict(new_index=i,nearest_old_index=j,standardized_distance=float(dist[i,j]),
        old_teacher_success=bool(checks["historical7"]["episodes"][j]["success"]),
        new_teacher_success=bool(checks["teacher7"]["episodes"][i-20]["success"])))
within=dist[:20,:20].copy();np.fill_diagonal(within,np.inf)
write("INITIAL_AUDIT.json",dict(file_sha256=sha(init),shape=list(states.shape),unique_initials=50,
    old_indices=list(range(20)),new_indices=list(range(20,30)),overlap=0,
    hashes={str(i):hs[i] for i in range(30)},variable_coordinates=len(vary),
    constant_coordinates=int(states.shape[1]-len(vary)),
    distance_definition="RMS standardized difference across coordinates varying in the full 50-state file; descriptive only, not physical/task difficulty metric; no environment instantiated.",
    new_to_old_nearest=neighbors,old_leave_one_out_nearest_median=float(np.median(within.min(axis=1))),
    new_to_old_nearest_median=float(np.median(dist[20:30,:20].min(axis=1)))))
# Source snippets give exact execution provenance without importing policy/env code.
files=[S/"coreset_exp/evaluate.py",S/"coreset_exp/eval_metric.py",
       S/"libero/lifelong/models/complete_policy.py",S/"libero/lifelong/models/base_policy.py",
       S/"libero/lifelong/models/policy_head.py",S/"coreset_exp/common.py"]
sources={}
for p in files:
    content=track(p).read_text()
    sources[str(p)]=dict(sha256=sha(p),text=content)
write("SOURCE_EVIDENCE.json",sources)
write("OUTCOMES.json",checks)
for p,h in inputs.items():assert sha(p)==h,p
write("INPUT_MANIFEST.json",inputs)
write("READONLY_ACCEPTANCE.json",dict(status="PASS",time=time.time(),input_hashes_unchanged=True,
    policy_imports=0,model_forwards=0,optimizer_calls=0,environment_calls=0,rng_draws=0,
    torch_cuda_initialized=torch.cuda.is_initialized()))
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status="PASS",config_differences=diffs,rng_pairs=pairs,
    outcomes={k:v["successes"] for k,v in checks.items()},state_shape=list(states.shape),variable_coordinates=len(vary)),indent=2))
