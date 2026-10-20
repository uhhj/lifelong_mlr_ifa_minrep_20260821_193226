"""Existing feature-table diagnostics only; no model imports, training or environment."""
import os
os.environ["OPENBLAS_NUM_THREADS"]="4"
os.environ["OMP_NUM_THREADS"]="4"
import json,time,hashlib
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4)
O=Path("/data/mlr_ifa/analyses/CZR_A_OBJECT_seed41_20261020")
A=Path("/data/mlr_ifa/analyses/CZR_A_OBJECT_seed41_20261020_generalization")
assert not A.exists(),"Preserve previous analysis"
A.mkdir()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
tic=time.time()
contract=json.loads((O/"CONTRACT.json").read_text())
tracked=dict(contract["files"])
for p in O.iterdir():
    if p.is_file():tracked[str(p)]=sha(p)
for p,h in tracked.items():assert sha(p)==h,p
saved=torch.load(O/"native_features.pt",map_location="cpu")
z=saved["z"].numpy().astype(np.float64)
pa=saved["parent"].numpy();y=saved["label"].numpy()
rows=json.loads((O/"WINDOW_OUTCOMES_LIGHT.json").read_text())
train=np.array([r["split"]=="train" for r in rows])
demo=np.array([int(r["demo_id"].split("_")[1]) for r in rows])
times=np.array([r["target_step"] for r in rows])
def stats(x):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)]
    return dict(n=len(x),min=float(x.min()),median=float(np.median(x)),max=float(x.max())) if len(x) else dict(n=0)
def distance(fit,mode):
    a=z.copy()
    if mode=="centered_unit":a-=z[fit].mean(0)
    if mode in ("centered_unit","unit"):a/=np.linalg.norm(a,axis=1,keepdims=True)+1e-12
    sq=(a*a).sum(1)
    d=np.sqrt(np.maximum(0,sq[:,None]+sq[None,:]-2*a@a.T))
    np.fill_diagonal(d,0)
    return d
def examine(fit,query,mode="centered_unit",protection="all_correct"):
    d=distance(fit,mode);ei=np.flatnonzero(fit&(pa!=y));ci=np.flatnonzero(fit&(pa==y))
    assert len(ci)>0
    radii=np.array([.45*d[i,ci if protection=="all_correct" else ci[pa[ci]==pa[i]]].min() for i in ei])
    out=[]
    fit_ids=np.flatnonzero(fit)
    for i in np.flatnonzero(query):
        allowed=pa[ei]==pa[i];valid=allowed&(radii>0)
        ratio=float((d[i,ei[valid]]/radii[valid]).min()) if valid.any() else None
        samec=ci[pa[ci]==pa[i]]
        near=fit_ids[np.argmin(d[i,fit_ids])]
        ne=ei[allowed][np.argmin(d[i,ei[allowed]])] if allowed.any() else None
        nc=samec[np.argmin(d[i,samec])] if len(samec) else None
        out.append(dict(sample_id=rows[i]["sample_id"],demo_id=rows[i]["demo_id"],
            parent_correct=bool(pa[i]==y[i]),ratio=ratio,
            nearest_error_distance=None if ne is None else float(d[i,ne]),
            nearest_same_parent_correct_distance=None if nc is None else float(d[i,nc]),
            nearest_expert_label_correct=bool(y[near]==y[i]),
            nearest_error_closer_than_correct=bool(ne is not None and nc is not None and d[i,ne]<d[i,nc]),
            nearest_any_same_demo=bool(demo[near]==demo[i]),
            nearest_any_time_gap=int(abs(times[near]-times[i])) if demo[near]==demo[i] else None))
    return out,d,ei,ci,radii
def summarize(rr):
    errors=[r for r in rr if not r["parent_correct"]];correct=[r for r in rr if r["parent_correct"]]
    result=dict(n=len(rr),parent_errors=len(errors),correct=len(correct),
                first_error_threshold=min((r["ratio"] for r in errors if r["ratio"] is not None),default=None),
                first_harm_threshold=min((r["ratio"] for r in correct if r["ratio"] is not None),default=None),
                error_ratio=stats([r["ratio"] for r in errors if r["ratio"] is not None]),
                correct_ratio=stats([r["ratio"] for r in correct if r["ratio"] is not None]))
    curve=[]
    for k in (1,2,3,4,6,8,10,16):
        fs=sum(r["ratio"] is not None and r["ratio"]<k for r in errors)
        sf=sum(r["ratio"] is not None and r["ratio"]<k for r in correct)
        curve.append(dict(radius_multiplier=k,FS=fs,SF=sf,net=fs-sf))
    result["fixed_counterfactual_curve"]=curve
    threshold=result["first_harm_threshold"]
    result["maximum_repairs_before_any_harm"]=sum(r["ratio"] is not None and r["ratio"]<threshold for r in errors) if threshold else None
    threshold=result["first_error_threshold"]
    result["harms_before_first_repair"]=sum(r["ratio"] is not None and r["ratio"]<=threshold for r in correct) if threshold else None
    er=np.array([r["ratio"] for r in errors if r["ratio"] is not None])
    cr=np.array([r["ratio"] for r in correct if r["ratio"] is not None])
    result["error_ranking_auc"]=float(((er[:,None]<cr).sum()+.5*(er[:,None]==cr).sum())/(len(er)*len(cr))) if len(er)*len(cr) else None
    result["nearest_error_beats_same_parent_correct"]=dict(
        errors=sum(r["nearest_error_closer_than_correct"] for r in errors),
        correct=sum(r["nearest_error_closer_than_correct"] for r in correct))
    result["nearest_expert_label_comparator"]=dict(
        FS=sum(r["nearest_expert_label_correct"] for r in errors),
        SF=sum(not r["nearest_expert_label_correct"] for r in correct))
    return result
analyses={};query_rows={}
for mode,protection in (("centered_unit","all_correct"),("centered_unit","matched_parent"),
                        ("unit","all_correct"),("euclidean","all_correct")):
    name=mode+"__"+protection
    held,d,ei,ci,radii=examine(train,~train,mode,protection)
    cross=[]
    for j in range(8):
        fit=train&(demo!=j);query=train&(demo==j)
        rr,_,_,_,_=examine(fit,query,mode,protection);cross+=rr
    analyses[name]=dict(holdout=summarize(held),cross_demo=summarize(cross))
    query_rows[name]=dict(holdout=held,cross_demo=cross)
    if name=="centered_unit__all_correct":
        centers=[]
        for k,i in enumerate(ei):
            nc=ci[np.argmin(d[i,ci])]
            same=ci[pa[ci]==pa[i]]
            nc_same=same[np.argmin(d[i,same])]
            other=ei[(demo[ei]!=demo[i])&(pa[ei]==pa[i])]
            nearest_other=other[np.argmin(d[i,other])] if len(other) else None
            centers.append(dict(sample_id=rows[i]["sample_id"],demo_id=rows[i]["demo_id"],
                 radius=float(radii[k]),nearest_correct_id=rows[nc]["sample_id"],
                 nearest_correct_same_demo=bool(demo[nc]==demo[i]),
                 nearest_correct_time_gap=int(abs(times[nc]-times[i])) if demo[nc]==demo[i] else None,
                 nearest_correct_parent_class_matches=bool(pa[nc]==pa[i]),
                 matched_protection_radius_ratio=float(d[i,nc_same]/d[i,nc]) if d[i,nc]>0 else None,
                 nearest_cross_demo_error_distance=None if nearest_other is None else float(d[i,nearest_other]),
                 cross_error_to_radius_ratio=None if nearest_other is None or radii[k]==0 else float(d[i,nearest_other]/radii[k])))
        (A/"CENTER_NEIGHBORS.json").write_text(json.dumps(centers,indent=2)+"\n")
        # Self-exclusion reveals whether overlapping same-demo windows inflate apparent generalization.
        loo=[]
        for i in np.flatnonzero(train):
            other=ei[(ei!=i)&(pa[ei]==pa[i])&(radii>0)]
            radius_lookup={v:radii[k] for k,v in enumerate(ei)}
            ratio=min((d[i,j]/radius_lookup[j] for j in other),default=None)
            loo.append(dict(sample_id=rows[i]["sample_id"],parent_correct=bool(pa[i]==y[i]),
                            ratio=ratio,nearest_expert_label_correct=False,
                            nearest_error_closer_than_correct=False))
        # For an error removed from centers, correct protection and normalization remain fixed.
        # This is explicitly center-self-exclusion, not refitted heldout validation.
        analyses[name]["center_self_exclusion"]=dict(
            errors=sum(not r["parent_correct"] for r in loo),
            errors_covered_by_another_center=sum(not r["parent_correct"] and r["ratio"] is not None and r["ratio"]<1 for r in loo))
        nearest_same=[]
        for i in np.flatnonzero(train):
            candidates=np.flatnonzero(train);candidates=candidates[candidates!=i]
            j=candidates[np.argmin(d[i,candidates])]
            nearest_same.append(dict(same_demo=bool(demo[j]==demo[i]),
                time_gap=int(abs(times[j]-times[i])) if demo[j]==demo[i] else None))
        (A/"SAME_DEMO_NEIGHBORS.json").write_text(json.dumps(dict(n=len(nearest_same),
            same_demo=sum(r["same_demo"] for r in nearest_same),
            same_demo_time_gaps=stats([r["time_gap"] for r in nearest_same if r["same_demo"]])),indent=2)+"\n")
    print(name,json.dumps(analyses[name]),flush=True)
for p,h in tracked.items():assert sha(p)==h,p
summary=dict(status="PASS",analyses=analyses,original_files_unchanged=True,
    input_hashes=tracked,new_model_forwards=0,optimizer_calls=0,environment_calls=0,
    original_radius_unchanged=True,seconds=time.time()-tic,script_sha256=sha(__file__),
    limitation="All counterfactuals are retrospective descriptive diagnostics, not tuning, new trained methods or independent confirmation.")
(A/"DIAGNOSTIC.json").write_text(json.dumps(summary,indent=2)+"\n")
(A/"QUERY_METADATA.json").write_text(json.dumps(query_rows,indent=2)+"\n")
