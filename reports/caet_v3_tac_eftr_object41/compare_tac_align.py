"""Exact comparison of accepted seed41 runs; no model loading or remote work."""
import datetime
import hashlib
import json
import sys
from fractions import Fraction as F
from pathlib import Path

root = Path(__file__).resolve().parent
tac, align, output = [Path(x).resolve() for x in sys.argv[1:4]]

def js(p):
    return json.loads(p.read_text(encoding='utf8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def metrics(r):
    return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,
                FWT=sum(r[t,t] for t in range(6,10))/4,
                NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
                OldFinal=sum(r[9,t] for t in range(6,9))/3,
                FinalAll=sum(r[9,t] for t in range(6,10))/4)

def episodes(p):
    rows=[json.loads(line) for f in sorted(p.glob('trials*.jsonl')) for line in f.read_text().splitlines()]
    by={(r['stage'],r['task'],r['trial']):r for r in rows}
    expected={(s,t,i) for s in range(6,10) for t in range(6,s+1) for i in range(20)}
    assert len(rows)==200 and set(by)==expected
    assert all(r['seed']==41 and type(r['success']) is bool for r in rows)
    rates={(s,t):F(sum(by[s,t,i]['success'] for i in range(20)),20) for s in range(6,10) for t in range(6,s+1)}
    return rates,by

rates={}; records={}; evidence={}; costs={}
for label,p in [('TAC',tac),('ALIGN',align)]:
    a=js(p/'LOCAL_FINAL_AUDIT.json')
    q=js(p/'QUEUE_TASK_COST_AUDIT.json')
    assert a['status']=='COMPLETED_LOCAL_INDEPENDENT_PASS' and q['status']=='QUEUE_AND_TASK_COST_AUDIT_PASS'
    assert q['local_final_sha256']==sha(p/'LOCAL_FINAL_AUDIT.json')
    manifest=js(p/'PACKAGE_MANIFEST.json')
    assert a['package_manifest_sha256']==sha(p/'PACKAGE_MANIFEST.json')
    # Verify the episode inputs used by this comparison against their accepted manifest.
    for prefix in ['current','baseline','v1']:
        for f in (p/prefix/'ra_coreset/evaluation').glob('trials*.jsonl'):
            e=manifest['entries'][f.relative_to(p).as_posix()]
            assert sha(f)==e['sha256'] and f.stat().st_size==e['bytes']
    rates[label],records[label]=episodes(p/'current/ra_coreset/evaluation')
    assert all(metrics(rates[label])[k]==F(v['fraction']) for k,v in a['metrics'].items())
    assert {(x['stage'],x['task']):F(x['current']) for x in a['matrix']}==rates[label]
    evidence[label]=dict(package=str(p),audit_sha256=sha(p/'LOCAL_FINAL_AUDIT.json'),queue_cost_sha256=sha(p/'QUEUE_TASK_COST_AUDIT.json'))
    costs[label]=dict(main_steps=a['main_steps'],main_epoch_minutes=a['main_training_seconds']/60,
                      formal_vector_minutes=q['formal_vector_minutes'],counts=a['counts'])
    for baseline,key in [('baseline','RA'),('v1','CAET_v2')]:
        r,b=episodes(p/baseline/'ra_coreset/evaluation')
        if key in rates:assert rates[key]==r and records[key]==b
        rates[key]=r;records[key]=b
for name,by in records.items():
    for key,r in by.items():
        ref=records['RA'][key]
        assert r['initial_state_sha256']==ref['initial_state_sha256'] and r['modal_sha256']==ref['modal_sha256']

def pack(m):
    return {k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()}

ms={k:metrics(r) for k,r in rates.items()}
comparisons={name:pack({k:ms[x][k]-ms[y][k] for k in ms[x]}) for name,x,y in
             [('TAC_minus_ALIGN','TAC','ALIGN'),('ALIGN_minus_RA','ALIGN','RA'),('TAC_minus_RA','TAC','RA')]}
matrix=[dict(stage=s,task=t,**{k:str(r[s,t]) for k,r in rates.items()}) for s,t in rates['RA']]
result=dict(status='ACCEPTED_RUNS_COMPARISON_PASS',observed_at=datetime.datetime.now().astimezone().isoformat(),
            seed=41,suite='OBJECT',metrics={k:pack(v) for k,v in ms.items()},comparisons=comparisons,
            matrix=matrix,costs=costs,evidence=evidence,EFTR='EARLY_TERMINATED_BY_USER_EXCLUDED',
            interpretation='TAC-ALIGN estimates the combined effect-regression and cross-task-contrast addition in this single seed; ALIGN-RA compares the BC temporal target and its legal mask. No significance claim, equal-compute claim, or closed-loop protection guarantee.')
output.mkdir(exist_ok=False)
(output/'COMPARISON.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['# OBJECT seed41：TAC 与 ALIGN 最终归因对照','','TAC、ALIGN均已完整独立验收；EFTR为用户提前终止，不纳入完整性能比较。单seed探索性，不作显著性结论。',
       '', '|指标（%）|RA|旧CAET-v2|ALIGN|TAC|TAC−ALIGN pp|ALIGN−RA pp|','|---|---:|---:|---:|---:|---:|---:|']
for k in ms['RA']:
    vals=[ms[name][k] for name in ['RA','CAET_v2','ALIGN','TAC']]+[ms['TAC'][k]-ms['ALIGN'][k],ms['ALIGN'][k]-ms['RA'][k]]
    lines.append('|'+k+'|'+'|'.join(f'{float(v*100):.6f}' for v in vals)+'|')
lines+=['','|学习阶段|任务|RA|ALIGN|TAC|TAC−ALIGN pp|','|---|---|---:|---:|---:|---:|']
for s,t in rates['RA']:
    vals=[rates[name][s,t] for name in ['RA','ALIGN','TAC']]+[rates['TAC'][s,t]-rates['ALIGN'][s,t]]
    lines.append(f'|{s}|{t}|'+'|'.join(f'{float(v*100):.2f}' for v in vals)+'|')
lines+=['','TAC−ALIGN对应效果回归与跨任务对比的联合增量，不能分别归因于其中一项。ALIGN−RA同时改变BC动作时间目标及合法mask，并非模型或输入窗口变化。',
        'NBT为有符号数值，越低越好，但受刚学完锚点影响；必须与最终旧任务成功率、各任务轨迹同时解释。AUC平均整个学习轨迹，不等于最终成功率。',
        '主训练均105500步；TAC含额外头更新与伙伴前向，成本不相等。']
for name,c in costs.items():lines.append(f"{name}：主epoch {c['main_epoch_minutes']:.3f}分钟，正式vector {c['formal_vector_minutes']:.3f}分钟；其他准备与验收另计。")
(output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False,indent=2))
