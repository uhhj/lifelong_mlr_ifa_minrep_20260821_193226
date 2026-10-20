"""Consolidate existing research artifacts, never run training or environments."""
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict
import json, hashlib, statistics, csv, datetime

R=Path(__file__).resolve().parent
ROOT=R.parents[1]
EX=ROOT/"execution"
(R/"sources").mkdir(exist_ok=True)
sources=[];source_cache={};runs=[];special=[];scope=[];comp=[];checks=[]
METRICS=["AUC","FWT","NBT","OldFinal","FinalAll"]
def load(rel):
    p=EX/rel
    if rel in source_cache:return source_cache[rel][1],source_cache[rel][0]
    raw=p.read_bytes();data=json.loads(raw.decode("utf-8-sig"))
    sid=f"S{len(sources)+1:03d}"
    local=f"sources/{sid}.json";(R/local).write_bytes(raw)
    sources.append(dict(id=sid,path=str(p.resolve()),relative=rel,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),copy=local))
    source_cache[rel]=(sid,data)
    return data,sid
def matrix_from_rows(rows,key):
    m=[[None]*4 for _ in range(4)]
    for x in rows:m[x["stage"]-6][x["task"]-6]=x[key]
    return m
def metrics(m):
    r={(s+6,t+6):F(str(m[s][t]))/100 for s in range(4) for t in range(s+1)}
    assert all(0<=v<=1 and (v*20).denominator==1 for v in r.values())
    return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
        FWT=sum(r[t,t] for t in range(6,10))/4,
        NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
        OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def add(suite,method,seed,matrix,sid,reported=None,version="首次",base="",kind="完整训练",epochs=400,steps=None,notes="",train_seconds=None,eval_seconds=None):
    result=metrics(matrix)
    for k,v in (reported or {}).items():
        if k not in METRICS:continue
        if isinstance(v,dict) and "fraction" in v:assert result[k]==F(v["fraction"]),(suite,method,seed,k,result[k],v)
        elif isinstance(v,str):assert result[k]==F(v)
        elif isinstance(v,dict) and "percent" in v:assert abs(float(result[k]*100)-v["percent"])<1e-8
    rid=f"R{len(runs)+1:03d}"
    runs.append(dict(id=rid,suite=suite,method=method,seed=seed,version=version,kind=kind,
        status="COMPLETED",episodes=200,vector_attempts=20,epochs=epochs,
        optimizer_steps=steps if steps is not None else (105500 if suite=="OBJECT" else 76100),
        checkpoints=4,baseline_id=base,matrix=matrix,metrics={k:float(v*100) for k,v in result.items()},
        exact={k:str(v) for k,v in result.items()},source=sid,notes=notes,
        train_minutes=None if train_seconds is None else train_seconds/60,
        eval_minutes=None if eval_seconds is None else eval_seconds/60,
        protocol="state=True；ep99；初态0–19；每cell20；阶段global动作RNG"))
    checks.append(dict(run_id=rid,source=sid,matrix_cells=10,denominator=20,reported_metrics_matched=True))
    return rid
def find(suite,method,seed,version="首次"):
    rr=[x["id"] for x in runs if (x["suite"],x["method"],x["seed"],x["version"])==(suite,method,seed,version)]
    assert len(rr)==1,(suite,method,seed,version,rr)
    return rr[0]
def timing(x,phase):
    return next((v["seconds"] for v in x.get("timings",[]) if v["phase"]==phase),None)
def scalars(method,suite,seed,obj,sid,prefix="",note=""):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if any(w in k.lower() for w in ("sha256","hashes","file","path","command","stdout","stderr","processes")):continue
            scalars(method,suite,seed,v,sid,f"{prefix}.{k}" if prefix else k,note)
    elif isinstance(obj,list):
        for i,v in enumerate(obj):scalars(method,suite,seed,v,sid,f"{prefix}[{i}]",note)
    else:
        special.append(dict(method=method,suite=suite,seed=seed,metric=prefix,
            numeric=obj if isinstance(obj,(int,float)) and not isinstance(obj,bool) else None,
            text="" if isinstance(obj,(int,float)) and not isinstance(obj,bool) else str(obj),
            scope=note,source=sid))

# Original five seeds and the complete seed43 rerun remain distinct observations.
x,sid=load("20261017_five_seed_summary/RESULTS.json")
names={"random":"Random＋IFA","coreset":"Coreset＋IFA","event_coreset":"Event＋Coreset＋IFA"}
for row in x["first_runs"]+x["seed43_repeat"]:
    ver="首次" if row["run"]=="first" else "重跑repeat1"
    add("OBJECT",names[row["method"]],row["seed"],row["matrix"],sid,row["metrics"],version=ver,
        notes="seed43重跑不替换首次；历史硬件差异及global RNG混杂")
for r in runs:
    if r["method"]!="Random＋IFA":
        r["baseline_id"]=find("OBJECT","Random＋IFA" if r["method"]=="Coreset＋IFA" else "Coreset＋IFA",r["seed"],r["version"])

x,sid=load("20261014_ifaoff_original_protocol/final_evidence/COMPARISON.json")
load("20261014_ifaoff_original_protocol/final_evidence/LOCAL_FINAL_RECOMPUTATION.json")
for row in x["matrices"]:
    if row["condition"]!="off":continue
    reported={}
    for v in x["rows"]:
        if v["seed"]==row["seed"]:reported[v["metric"]]={"percent":v["values_percent"]["off"][row["arm"]]}
    add("OBJECT","Random（IFA关闭）" if row["arm"]=="random" else "Coreset（IFA关闭）",
        row["seed"],row["matrix"],sid,reported,
        base=find("OBJECT",names[row["arm"]],row["seed"]),notes="重训两组；只关闭IFA，保留原正式200口径")

for seed in (42,43,44):
    x,sid=load(f"20261017_coreset_goal_three_seed/final_evidence/seed{seed}/SUMMARY.json")
    for arm,row in x["arms"].items():
        add("GOAL",names[arm],seed,row["matrix"],sid,row["metrics"],
            base=find("GOAL","Random＋IFA",seed) if arm=="coreset" else "",
            train_seconds=row.get("training_seconds"),notes="新论文描述对齐共同实现；旧GOAL state_v2不混入")
load("20261017_coreset_goal_three_seed/RUN_STATUS.json")

x,sid=load("20261019_ra_coreset_goal_seed41/progress_20261019_222601/queue/AGGREGATE.json")
load("20261019_ra_coreset_goal_seed41/RUN_STATUS.json")
for arm in ("random","coreset","ra_coreset"):
    rows=[{**z,"percent":100*z["successes"]//z["trials"]} for z in x["matrix"] if z["arm"]==arm]
    add("GOAL",names.get(arm,"RA-Coreset＋IFA"),41,matrix_from_rows(rows,"percent"),sid,x["metrics"][arm],
        base=find("GOAL","Random＋IFA" if arm=="coreset" else "Coreset＋IFA",41) if arm!="random" else "")

# Original zero-response version, with the user's later reporting comparator.
x,sid=load("20261017_zr_coreset_seeds40_41/final_evidence/queue/FIVE_SEED_SUMMARY.json")
load("20261017_zr_coreset_seeds40_41/LOCAL_FINAL_ACCEPTANCE.json")
for row in x["primary_rows"]:
    seed=row["seed"];t=x["tables"][str(seed)]["new"]
    m=[[None]*4 for _ in range(4)]
    for key,v in t.items():
        s,u=map(int,key.split(","));m[s-6][u-6]=v
    add("OBJECT","Coreset＋IFA＋ZR-v1",seed,m,sid,row["new"],
        base=find("OBJECT","Coreset＋IFA",seed,"重跑repeat1" if seed==43 else "首次"),
        notes="共享网络仍训练；43主比较事后改用repeat1，训练绑定未改")

for rel,label,key in [
    ("20261018_sc_coreset_goal_seed42/final_evidence/AUDIT.json","SC-v1＋IFA","sc_coreset"),
    ("20261018_scv2_goal_seed42/LOCAL_FINAL_ACCEPTANCE.json","SC-v2＋IFA","SC-v2"),
    ("20261018_ra_coreset_goal_seed42/LOCAL_FINAL_ACCEPTANCE.json","RA-Coreset＋IFA","RA-Coreset")]:
    x,sid=load(rel)
    add("GOAL",label,42,matrix_from_rows(x["matrix"],key),sid,x["metrics"][key],
        base=find("GOAL","Coreset＋IFA",42),train_seconds=timing(x,"train"),eval_seconds=timing(x,"evaluate"),
        notes="SC-v1/v2均未通过扩展门槛" if label.startswith("SC") else "半配额Random锚定")
    scalars(label,"GOAL",42,x.get("storage",x.get("selection",{})),sid,"selection","已有离线入库诊断；非成功率")

x,sid=load("20261019_ra_coreset_goal_extension/progress_20261019_160822/LOCAL_FINAL_ACCEPTANCE.json")
for seed in (43,44):
    row=x["per_seed"][str(seed)]
    add("GOAL","RA-Coreset＋IFA",seed,matrix_from_rows(row["matrix"],"ra_coreset"),sid,row["metrics"]["ra_coreset"],
        base=find("GOAL","Coreset＋IFA",seed),train_seconds=timing(row,"train"),eval_seconds=timing(row,"evaluate"))
    scalars("RA-Coreset＋IFA","GOAL",seed,row["storage"],sid,"storage","入库审计")

x,sid=load("20261018_zrv2_seed43/completion_20261018/LOCAL_FINAL_ACCEPTANCE.json")
add("OBJECT","Coreset＋IFA＋ZR-v2",43,matrix_from_rows(x["matrix"],"new_percent"),sid,x["comparison"]["new"],
    base=find("OBJECT","Coreset＋IFA",43,"重跑repeat1"),kind="冻结父模型后校准",epochs=40,steps=10550,
    notes="父模型repeat1；AUC上升但原NBT门槛失败；未扩seed")

x,sid=load("20261019_ra_object_conditional/completion_20261020/LOCAL_FINAL_ACCEPTANCE.json")
for seed in (41,42,43):
    row=x["seed_results"][str(seed)]
    for arm,label in (("ra_coreset","RA-Coreset＋IFA"),("ra_coreset_ifa_zrv2","RA-Coreset＋IFA＋ZR-v2")):
        z=row[arm]
        add("OBJECT",label,seed,z["matrix"],sid,z["metrics"],
            base=find("OBJECT","Random＋IFA",seed,"重跑repeat1" if seed==43 else "首次") if arm=="ra_coreset" else find("OBJECT","RA-Coreset＋IFA",seed),
            kind="完整训练" if arm=="ra_coreset" else "冻结父模型后校准",
            epochs=400 if arm=="ra_coreset" else 40,steps=105500 if arm=="ra_coreset" else 10550,
            train_seconds=x["epoch_seconds"][str(seed)]["ra_committed_epochs" if arm=="ra_coreset" else "zr_calibration_epochs"],
            notes="耗时为epoch累计；RA41工程恢复额外重放157物理步，逻辑预算不重复；第二关失败未跑GOAL组合")
    for arm in ("random","coreset"):
        rr=next(v for v in runs if v["id"]==find("OBJECT",names[arm],seed,"重跑repeat1" if seed==43 else "首次"))
        assert rr["matrix"]==row[arm]["matrix"]

x,sid=load("20261020_zrv3_object_seed41/progress_20261020_144750/LOCAL_ACCEPTANCE.json")
z=x["variants"]["v3"]
add("OBJECT","RA-Coreset＋IFA＋ZR-v3",41,z["matrix"],sid,z["metrics"],
    base=find("OBJECT","RA-Coreset＋IFA",41),kind="冻结父模型后校准",epochs=40,steps=10550,
    notes="epsilon0.5实际KL投影；门槛未通过，无投影v3-A未执行")
scalars("RA-Coreset＋IFA＋ZR-v3","OBJECT",41,{k:v for k,v in z.items() if k not in ("metrics","matrix","parent_delta","v2_delta")},sid,"","已有在线/缓存诊断")

# Prototype and offline-only methods do not receive fabricated AUC/FWT/NBT.
for rel,method,seed,note in [
    ("20261017_tcc_zero_training/SUMMARY.json","TCC",None,"零训练诊断；未进入正式实验"),
    ("20261017_update_coordination_diagnostic/LOCAL_ACCEPTANCE.json","新旧任务更新协调",None,"功能副本诊断，非正式更新；无闭环"),
    ("20261017_update_coordination_diagnostic/RESULTS.json","新旧任务更新协调",None,"现有checkpoint诊断"),
    ("20261017_zero_response_residual/FINAL_ACCEPTANCE.json","ZR门控原型",42,"200次原型更新；0闭环"),
    ("20261020_sazr_object_seed41/FINAL_SUMMARY.json","SAZR（RA父模型Task6）",41,"单任务原型；验证拒绝；正式使用父策略"),
    ("20261020_gripper_repair_stageA/FINAL_AUDIT.json","夹爪修复Stage A",41,"离线训练/留出验证；0闭环，B/C未运行")]:
    x,sid=load(rel)
    if method=="新旧任务更新协调" and "probes" in x:
        x={k:v for k,v in x.items() if k not in ("probes","cells")}
    scalars(method,"OBJECT",seed,x,sid,"",note)
    scope.append(dict(method=method,suite="OBJECT",seeds=str(seed) if seed else "40–44",
        category="原型/离线诊断",status=x.get("status",x.get("decision","已有结果")),notes=note,source=sid))

x,sid=load("20261014_coreset_validation/CANCELLATION.json")
scalars("独立初态追加复评（取消）","OBJECT",None,x,sid,"","取消的9000条计划，不与原协议合并")
scope.append(dict(method="独立初态追加复评",suite="OBJECT",seeds="42–44计划",category="取消/不完整",
    status="CANCELLED",notes="保留已有11条正式+20条A/A；不生成完整指标",source=sid))
scope.extend([
    dict(method="ZR-v3-A无投影",suite="OBJECT",seeds="41",category="条件分支",status="NOT_STARTED",notes="v3门槛失败",source=""),
    dict(method="RA-Coreset＋ZR-v2＋IFA / Coreset＋ZR-v2＋IFA",suite="GOAL",seeds="41–44条件范围",category="条件分支",status="NOT_STARTED",notes="OBJECT第二关失败，未启动GOAL组合",source=""),
    dict(method="SC-v1/SC-v2扩展",suite="GOAL",seeds="43、44",category="条件分支",status="NOT_STARTED",notes="seed42门槛失败",source=""),
    dict(method="E1/E1-U/E1-O/R1、GOAL旧state_v2/E2/E3",suite="OBJECT/GOAL",seeds="历史",category="本轮范围外",status="EXCLUDED",notes="发生在本轮Coreset路线之前；旧state/评估协议不同，未并入排名",source=""),
])

# Compare only matching actual runs; keep post-result comparator switches explicit.
index={x["id"]:x for x in runs}
for r in runs:
    if not r["baseline_id"]:continue
    b=index[r["baseline_id"]]
    comp.append(dict(run_id=r["id"],suite=r["suite"],method=r["method"],seed=r["seed"],
        version=r["version"],baseline_id=b["id"],baseline=b["method"]+" / "+b["version"],
        **{k:r["metrics"][k]-b["metrics"][k] for k in METRICS},note=r["notes"]))
    if r["method"]=="Coreset＋IFA＋ZR-v1" and r["seed"]==43:
        b=index[find("OBJECT","Coreset＋IFA",43)]
        comp.append(dict(run_id=r["id"],suite=r["suite"],method=r["method"],seed=43,
            version="首次43敏感性口径",baseline_id=b["id"],baseline=b["method"]+" / 首次",
            **{k:r["metrics"][k]-b["metrics"][k] for k in METRICS},note="辅助口径，不替换主比较"))

groups=defaultdict(list)
for r in runs:groups[r["suite"],r["method"],r["version"]].append(r)
summaries=[]
for (suite,method,version),rs in groups.items():
    summaries.append(dict(suite=suite,method=method,version=version,n=len(rs),
        seeds=",".join(str(x["seed"]) for x in sorted(rs,key=lambda x:x["seed"])),
        run_ids=[x["id"] for x in rs],episodes=sum(x["episodes"] for x in rs),
        **{k:statistics.mean(x["metrics"][k] for x in rs) for k in METRICS},
        **{k+"_SD":statistics.stdev(x["metrics"][k] for x in rs) if len(rs)>1 else None for k in METRICS}))
    scope.append(dict(method=method,suite=suite,seeds=summaries[-1]["seeds"],category="完整四阶段闭环",
        status="COMPLETED",notes=version+"；每个seed200条，均值只含表列seed",source=",".join(sorted(set(x["source"] for x in rs)))))

assert len({(r["suite"],r["method"],r["seed"],r["version"]) for r in runs})==len(runs)
for s in sources:assert hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest()==s["sha256"]
data=dict(created=datetime.datetime.now().astimezone().isoformat(),runs=runs,summaries=summaries,
    comparisons=comp,special=special,scope=scope,sources=sources,
    acceptance=dict(status="PASS",runs=len(runs),matrix_cells=len(runs)*10,
        source_files=len(sources),all_source_hashes_unchanged=True,reported_metric_checks=checks,
        no_training_or_rollouts=True,sd_ddof=1))
(R/"ALL_RESULTS.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
for name,rows in [("single_runs",[{k:v for k,v in r.items() if k not in ("matrix","metrics","exact")} | r["metrics"] for r in runs]),
                  ("comparisons",comp),("special_metrics",special)]:
    with (R/(name+".csv")).open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/"RUN_STATUS.json").write_text(json.dumps(dict(status="DATA_RECOMPUTED_WORKBOOK_PENDING",runs=len(runs)))+"\n")
print(json.dumps(dict(runs=len(runs),groups=len(summaries),special_metrics=len(special),
    sources=len(sources),completed_formal_records=len(runs)*200),indent=2))
