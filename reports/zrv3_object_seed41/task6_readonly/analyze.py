"""Read-only three-way episode, residual and projection analysis."""
import json, hashlib, sys
from pathlib import Path
from collections import Counter
from fractions import Fraction as F
R=Path(__file__).resolve().parent
P=R.parent
sys.path.insert(0,str(P))
from flow_metrics import metrics,pack
E=P/"progress_20261020_144750/evidence"
inputs={}
def read(p):
    b=p.read_bytes();inputs[str(p)]=hashlib.sha256(b).hexdigest();return b
def js(p):return json.loads(read(p))
def jl(p):return [json.loads(l) for l in read(p).decode().splitlines()]
remote=js(R/"REMOTE_AUDIT.json")
assert remote["input_hashes_after_equal"] and not remote["cuda_initialized"]
for p in [
    P/"template/attachment/execution_core.py",
    P/"template/source/zr3_exp/model.py",
    P/"template/source/zr3_exp/calibrate.py",
    P/"template/source/zr3_exp/diagnostics.py",
    P/"template/source/zr3_exp/evaluate.py",
    P.parent/"20261019_ra_object_conditional/server_auto/zr_template/source/zr2_exp/model.py",
]:
    read(p)
roots={"RA":E/"controls/RA","v2":E/"controls/ZR_v2","v3":E/"v3/evaluation"}
arms={"RA":"ra_coreset","v2":"ra_coreset_ifa_zrv2","v3":js(E/"v3/RUN_SPEC.json")["arm"]}
rows={};rates={};ms={}
for name,root in roots.items():
    for p in root.glob("trials*.jsonl"):read(p)
    ms[name],rates[name],rows[name],_=metrics(root,41,arms[name])
triples=[];paired=[]
for s in range(6,10):
    for t in range(6,s+1):
        for i in range(20):
            k=(s,t,i);rs={n:rows[n][k] for n in roots}
            assert len({r["initial_state_sha256"] for r in rs.values()})==1
            assert len({r["modal_sha256"] for r in rs.values()})==1
            assert all(rs[n]["parent_checkpoint_sha256"]==rs["RA"]["checkpoint_sha256"] for n in ("v2","v3"))
            triples.append(dict(stage=s,task=t,trial=i,initial_state_sha256=rs["RA"]["initial_state_sha256"],
                success={n:r["success"] for n,r in rs.items()},
                vector_steps={n:r["vector_group_steps"] for n,r in rs.items()}))
        for left,right in (("RA","v2"),("RA","v3"),("v2","v3")):
            cases={c:[] for c in ("SS","SF","FS","FF")}
            for i in range(20):
                label="".join("S" if rows[n][s,t,i]["success"] else "F" for n in (left,right))
                cases[label].append(i)
            assert len(cases["FS"])-len(cases["SF"])==int((rates[right][s,t]-rates[left][s,t])*20)
            paired.append(dict(stage=s,task=t,left=left,right=right,counts={k:len(v) for k,v in cases.items()},ids=cases))
task6history=[]
for i in range(20):
    task6history.append(dict(trial=i,initial_state_sha256=rows["RA"][6,6,i]["initial_state_sha256"],
        history={n:"".join("S" if rows[n][s,6,i]["success"] else "F" for s in range(6,10)) for n in roots}))
proj=[]
for s in range(6,10):
    for t in range(6,s+1):
        c=js(E/"v3/evaluation"/f"cell_stage{s}_task{t}.json")
        proj.append(dict(stage=s,task=t,success_percent=float(rates["v3"][s,t]*100),
            raw_KL_max=c["max_raw_kl"],executed_KL_max=c["max_executed_kl"],
            projected=c["projected_states"],states=c["sampled_states"],
            projected_percent=100*c["projected_states"]/c["sampled_states"],
            alpha_at_max_raw=(.5/c["max_raw_kl"])**.5 if c["max_raw_kl"]>.5 else 1))
structure=[];loss=[]
for n in ("v2","v3"):
    d=remote["logs"][n]
    assert d["batch_count"]==10550
    for r in d["STRUCTURE.jsonl"]:
        if r["label"]!="stage_end":continue
        structure.append(dict(method=n,**r))
    for s in range(6,10):
        ep=[r for r in d["epochs.json"] if r["task"]==s]
        bs=[r for r in d["batch_summary"] if r["stage"]==s]
        loss.append(dict(method=n,stage=s,epoch0_loss=ep[0]["loss"],epoch9_loss=ep[-1]["loss"],
            task6_present_updates=sum(r["task6_present_batches"] for r in bs),
            grad_clipping_certain_batches_lower_bound=sum(r["batches"] for r in bs if r["grad_norm"]["min"]>100),
            max_grad_norm=max(r["grad_norm"]["max"] for r in bs),
            projected=sum(r.get("projected_states",0) for r in bs) if n=="v3" else None,
            states=sum(r.get("states",0) for r in bs) if n=="v3" else None))
config_diffs=[]
def flatten(d,p=""):
    if isinstance(d,dict):
        out={}
        for k,v in d.items():out.update(flatten(v,p+"."+str(k)))
        return out
    return {p:d}
configs={}
for n,files in remote["configs"].items():
    assert "resolved_actual.json" in files,(n,files.keys())
    configs[n]=flatten(files["resolved_actual.json"])
for k in sorted(set.union(*(set(v) for v in configs.values()))):
    vals={n:d.get(k,"<missing>") for n,d in configs.items()}
    if len({json.dumps(v,sort_keys=True) for v in vals.values()})>1:config_diffs.append(dict(field=k,values=vals))
decomp={}
for name in ("v2","v3"):
    decomp[name]={str(t):str(sum(rates[name][s,t]-rates["RA"][s,t] for s in range(t,10))/(10-t)/4) for t in range(6,10)}
    assert sum(F(x) for x in decomp[name].values())==ms[name]["AUC"]-ms["RA"]["AUC"]
analysis=dict(status="PASS_READONLY",metrics={n:pack(m) for n,m in ms.items()},auc_contribution_fraction=decomp,
    pairs=paired,triples=triples,task6_history=task6history,projection=proj,structure=structure,loss=loss,
    config_differences=config_diffs,rng=remote["rng"],input_hashes=inputs,new_model_forwards=0,
    new_optimizer_calls=0,new_environment_steps=0,limitations=[
        "Projection and residual logs are cell/dataset/batch aggregates, not per-episode action traces.",
        "Later-stage structure maxima mix tasks; task6 branch norms are parameter norms, not action shifts.",
        "No unprojected-v3 arm, so v2-v3 does not isolate projection causality."
    ])
for s in range(6,10):
    a,b=[next(r for r in structure if r["method"]==n and r["stage"]==s) for n in ("v2","v3")]
    assert a["public_hash"]==b["public_hash"] and a["T0_hash"]==b["T0_hash"]
    assert all(x["vector_steps"]=={"RA":600,"v2":600,"v3":600} for x in triples if x["stage"]==s and x["task"]==6)
    assert next(r for r in remote["rng"] if r["stage"]==s and r["task"]==6)["equal_all"]
analysis["task6_all_stage_rng_equal_and_600_step_vectors"]=True
for p,h in inputs.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
analysis["local_input_hashes_after_equal"]=True
(R/"ANALYSIS.json").write_text(json.dumps(analysis,indent=2)+"\n")
(R/"TASK6_EPISODE_HISTORY.json").write_text(json.dumps(task6history,indent=2)+"\n")
print(json.dumps({k:analysis[k] for k in ("status","auc_contribution_fraction","config_differences","projection","loss")},indent=2))
print("TASK6 PAIRS",json.dumps([r for r in paired if r["task"]==6],indent=2))
print("TASK6 STRUCTURE",json.dumps([{k:r.get(k) for k in ("method","stage","delta_max","raw_kl_max","projected_samples","samples","branch_norms","cross_contribution_and_mean")} for r in structure],indent=2))
print("RNG",[(r["stage"],r["task"],r["equal_all"]) for r in remote["rng"]])
