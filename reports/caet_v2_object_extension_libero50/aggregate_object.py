"""Aggregate only three fully accepted metadata packages; no server/model access."""
import argparse
import hashlib
import json
import statistics
from fractions import Fraction as F
from pathlib import Path

def js(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def matrix(root,seed):
    rows=[json.loads(line) for p in sorted((root/'ra_coreset/evaluation').glob('trials*.jsonl'))
          for line in p.read_text(encoding='utf8').splitlines() if line.strip()]
    assert len(rows)==200
    by={(r['stage'],r['task'],r['trial']):r for r in rows}
    assert len(by)==200 and all(r['seed']==seed and type(r['success']) is bool for r in rows)
    return {(s,t):F(sum(by[s,t,i]['success'] for i in range(20)),20)
            for s in range(6,10) for t in range(6,s+1)}
def metrics(r):
    return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
        FWT=sum(r[t,t] for t in range(6,10))/4,
        NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
        OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def summary(values):
    return dict(mean_fraction=str(sum(values)/len(values)),mean_percent=float(100*sum(values)/len(values)),
                sample_sd_pp=statistics.stdev(float(100*x) for x in values),ddof=1,n=len(values))
ap=argparse.ArgumentParser()
for seed in (41,42,43):ap.add_argument('--seed'+str(seed),required=True,type=Path)
ap.add_argument('--output',required=True,type=Path)
args=ap.parse_args()
assert not args.output.exists(), 'Preserve previous reports'
per_seed={}; matrices={}; inputs={};costs={}
for seed in (41,42,43):
    p=getattr(args,'seed'+str(seed)).resolve()
    a=js(p/'LOCAL_FINAL_AUDIT.json');manifest=js(p/'PACKAGE_MANIFEST.json')
    assert a['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS' and a['seed']==seed
    assert a['method']=='CAET-v2' and a['suite']=='OBJECT'
    assert a['package_manifest_sha256']==sha(p/'PACKAGE_MANIFEST.json')
    for name,item in manifest['entries'].items():
        f=(p/name).resolve();assert p in f.parents
        assert f.stat().st_size==item['bytes'] and sha(f)==item['sha256']
    assert a['main_steps']==105500 and a['formal']['records']==200
    assert a['formal']['completed_vectors']==a['formal']['started_vectors']==20
    assert a['formal']['completed_cells']==10 and a['formal']['technical_exceptions']==0
    per_seed[seed]={};matrices[seed]={}
    for method,folder,key in [('v2','current','metrics'),('ra','baseline','ra'),('v1','v1','caet_v1')]:
        r=matrix(p/folder,seed);m=metrics(r)
        assert all(F(a[key][k]['fraction'])==v for k,v in m.items())
        per_seed[seed][method]=m
        matrices[seed][method]=[dict(stage=s,task=t,rate=str(r[s,t])) for s,t in r]
    c=js(p/'TASK_COST_AUDIT.json');assert c['status']=='TASK_AND_COST_RECOMPUTATION_PASS'
    costs[seed]=c
    inputs[seed]=dict(package=str(p),manifest_sha256=sha(p/'PACKAGE_MANIFEST.json'),
        audit_sha256=sha(p/'LOCAL_FINAL_AUDIT.json'),cost_sha256=sha(p/'TASK_COST_AUDIT.json'))
names=('AUC','FWT','NBT','OldFinal','FinalAll')
means={m:{k:summary([per_seed[s][m][k] for s in (41,42,43)]) for k in names} for m in ('ra','v1','v2')}
deltas={m:{k:summary([per_seed[s]['v2'][k]-per_seed[s][m][k] for s in (41,42,43)]) for k in names} for m in ('ra','v1')}
packed={s:{m:{k:str(v) for k,v in values.items()} for m,values in methods.items()} for s,methods in per_seed.items()}
result=dict(status='THREE_SEED_LOCAL_RECOMPUTATION_PASS',seeds=[41,42,43],method='CAET-v2',suite='OBJECT',
    per_seed=packed,means=means,paired_deltas=deltas,matrices=matrices,inputs=inputs,costs=costs,
    main_updates=316500,formal_episodes=600,formal_vectors=60,preflight_updates=24,
    statistical_significance_claim=False,automatic_extension_authorized=False)
args.output.mkdir(parents=True)
(args.output/'AGGREGATE_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
lines=['# CAET-v2 OBJECT 三seed结果','',
    '仅汇总全部独立验收通过的seed41/42/43，等权均值±样本标准差（ddof=1）。',
    '这是看过seed42后的探索性扩展，不宣称统计显著。CAET-v1保留post-action输入的科学问题。','',
    '|指标|RA|CAET-v1|CAET-v2|v2−RA 配对差|','|---|---:|---:|---:|---:|']
for k in names:
    values=[means[m][k] for m in ('ra','v1','v2')]+[deltas['ra'][k]]
    lines.append('|'+k+'|'+'|'.join(f"{x['mean_percent']:.6f} ± {x['sample_sd_pp']:.6f}" for x in values)+'|')
lines+=['','FWT沿用训练后对角线平均；NBT越低越好。旧任务最终成功率、各任务负结果及全部矩阵保留在审计JSON。',
    '辅助下一动作目标与主BC相差一个动作索引，global评估随机流未逐轨迹隔离；不能仅凭聚合指标分离迁移、保护或时序修正的因果作用。',
    '主更新316500、正式600条/60vector、隔离预检24次。效果拟合额外更新与各阶段曝光独立记录，负墙钟耗时保持不可用。']
(args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=result['status'],means=means,paired_deltas=deltas),indent=2))
