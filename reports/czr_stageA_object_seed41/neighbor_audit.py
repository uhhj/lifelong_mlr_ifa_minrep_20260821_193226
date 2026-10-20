"""Post-execution descriptive distances on saved features; no threshold selection."""
import sys,json,hashlib,time
from pathlib import Path
import numpy as np
import torch
W=Path("/data/mlr_ifa/experiment_workspaces/CZR_A_OBJECT_seed41_20261020")
O=Path("/data/mlr_ifa/analyses/CZR_A_OBJECT_seed41_20261020")
sys.path.insert(0,str(W))
from czr_core import Memory
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert int((W/"run.exit").read_text())==0
f=O/"native_features.pt";before=sha(f)
saved=torch.load(f,map_location="cpu")
z=saved["z"].numpy();pa=saved["parent"].numpy();y=saved["label"].numpy()
rows=json.loads((O/"WINDOW_OUTCOMES_LIGHT.json").read_text())
train=np.array([r["split"]=="train" for r in rows])
def summary(v):
    v=np.asarray(v,dtype=float)
    return dict(n=len(v),min=float(v.min()),median=float(np.median(v)),max=float(v.max())) if len(v) else dict(n=0)
def describe(fit,query):
    b=Memory(z[fit],pa[fit],y[fit]);phi=b.phi(z[query]);ids=np.flatnonzero(query)
    out=[]
    for x,i in zip(phi,ids):
        eligible=(b.parent==pa[i])&(b.radius>0)
        if eligible.any():
            distance=np.linalg.norm(b.centers[eligible]-x,axis=1)
            ratios=distance/b.radius[eligible]
            out.append(dict(sample_id=rows[i]["sample_id"],demo_id=rows[i]["demo_id"],
                parent_correct=bool(pa[i]==y[i]),eligible_centers=int(eligible.sum()),
                nearest_distance=float(distance.min()),min_distance_to_radius_ratio=float(ratios.min()),
                inside_any_ball=bool((ratios<1).any())))
        else:
            out.append(dict(sample_id=rows[i]["sample_id"],demo_id=rows[i]["demo_id"],
                parent_correct=bool(pa[i]==y[i]),eligible_centers=0,nearest_distance=None,
                min_distance_to_radius_ratio=None,inside_any_ball=False))
    return out
held=describe(train,~train);cross=[]
for d in range(8):
    q=train&np.array([r["demo_id"]==f"demo_{d}" for r in rows])
    cross+=describe(train&~q,q)
groups={}
for name,rr in (("holdout",held),("cross_demo",cross)):
    for correct in (False,True):
        subset=[r for r in rr if r["parent_correct"]==correct]
        groups[name+("_correct" if correct else "_error")]=dict(n=len(subset),
            no_eligible_support=sum(r["eligible_centers"]==0 for r in subset),
            inside=sum(r["inside_any_ball"] for r in subset),
            ratio=summary([r["min_distance_to_radius_ratio"] for r in subset if r["eligible_centers"]]),
            distance=summary([r["nearest_distance"] for r in subset if r["eligible_centers"]]))
assert sha(f)==before
result=dict(status="PASS",groups=groups,heldout_rows=held,cross_demo_rows=cross,
            saved_features_unchanged=True,features_sha256=before,script_sha256=sha(__file__),
            new_model_forwards=0,optimizer_calls=0,environment_calls=0,
            note="Descriptive posthoc distances only; fixed radius unchanged; no new validation selection.",
            time=time.time())
(O/"NEIGHBOR_AUDIT.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(groups,indent=2))
