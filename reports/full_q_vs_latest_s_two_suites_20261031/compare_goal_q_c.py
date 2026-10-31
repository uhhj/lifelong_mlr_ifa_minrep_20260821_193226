"""Read-only comparison of accepted GOAL Q and budget-extended C; no runs."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,datetime,statistics
R=Path(__file__).resolve().parent;C=R.parent/'20261030_equal_optimizer_budget_cs_two_suites'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
qa=js(R/'GOAL_ACCEPTANCE.json');ca=js(C/'GOAL_ACCEPTANCE.json');qd=Path(qa['final_path']);cd=Path(ca['final_directory'])
assert qa['status']=='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE' and ca['status']=='ACCEPTED_GOAL_THREE_C_EQUAL_OPTIMIZER_BUDGET'
assert sha(qd/'LOCAL_FINAL_AUDIT.json')==qa['audit_sha256'] and sha(qd/'EVIDENCE.json')==qa['evidence_sha256']
assert sha(cd/'LOCAL_FINAL_AUDIT.json')==ca['audit_sha256'] and sha(cd/'PRIMARY_S_COMPARISON.json')==ca['comparison_sha256']
qau=js(qd/'LOCAL_FINAL_AUDIT.json');cau=js(cd/'LOCAL_FINAL_AUDIT.json');cc=js(cd/'PRIMARY_S_COMPARISON.json')
assert cau['source_sha256']==sha(cd/'EVIDENCE.json')
qv=js(qd/'EVIDENCE.json');cv=js(cd/'EVIDENCE.json')
def metrics(rows):
 r={(s,t):F(sum(x['success'] for x in rows if (x['stage'],x['task'])==(s,t)),20) for s in range(6,10) for t in range(6,s+1)}
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
cases={};cells=[]
for seed in (41,42,43):
 case=f'GOAL_{seed}';qr=json.loads(qv['light'][case+'/OUTCOMES.json']['text']);cr=json.loads(cv['light'][case+'/OUTCOMES.json']['text']);assert len(qr)==len(cr)==200
 key=lambda x:(x['stage'],x['task'],x['initial_index']);cm={key(x):x for x in cr};qm={key(x):x for x in qr};assert cm.keys()==qm.keys() and len(cm)==200
 pairs=[]
 for k,q in sorted(qm.items()):
  c=cm[k]
  for field in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed','attempt']:assert q[field]==c[field],(case,k,field)
  pairs.append(dict(stage=k[0],task=k[1],initial_index=k[2],Q=q['success'],C=c['success'],initial_sha256=q['initial_sha256'],noise_sha256=q['noise_sha256']))
 qmet=metrics(qr);cmet=metrics(cr)
 for name in qmet:
  assert str(qmet[name])==qau['cases'][case]['metrics'][name]['fraction']
  assert abs(float(cmet[name]*100)-cc['cases'][case]['C_metrics'][name])<1e-10
 cases[case]=dict(Q={k:float(v*100) for k,v in qmet.items()},C={k:float(v*100) for k,v in cmet.items()},Q_minus_C_pp={k:float((qmet[k]-cmet[k])*100) for k in qmet},paired_outcomes=pairs)
 for stage in range(6,10):
  for task in range(6,stage+1):
   rr=[x for x in pairs if (x['stage'],x['task'])==(stage,task)];assert len(rr)==20
   cells.append(dict(case=case,stage=stage,task=task,Q=5*sum(x['Q'] for x in rr),C=5*sum(x['C'] for x in rr),Q_success_C_fail=[x['initial_index'] for x in rr if x['Q'] and not x['C']],C_success_Q_fail=[x['initial_index'] for x in rr if x['C'] and not x['Q']]))
means={k:dict(Q=statistics.mean(x['Q'][k] for x in cases.values()),C=statistics.mean(x['C'][k] for x in cases.values()),Q_minus_C_pp=statistics.mean(x['Q_minus_C_pp'][k] for x in cases.values())) for k in qmet}
out=dict(status='PASS_GOAL_ACCEPTED_Q_AND_BUDGET_C_600_PAIRED_RECOMPUTATION',observed=datetime.datetime.now().astimezone().isoformat(),sources={arm:dict(path=str(d),evidence_sha256=sha(d/'EVIDENCE.json'),audit_sha256=sha(d/'LOCAL_FINAL_AUDIT.json')) for arm,d in [('Q',qd),('C',cd)]},means=means,cases=cases,cells=cells,paired_episodes=600,optimizer_calls_added=0,environment_calls_added=0,costs=dict(Q_production=228300,C_production=426600,Q_training_collection=120,C_training_collection=0),limits=['Budget-extended C, not historical 100-epoch C','Different update budget, schedules, training interactions and own success cache; no isolated retention causal claim','Three reused development seeds; no significance or independent confirmation','OBJECT Q not yet accepted; do not extrapolate'])
(R/'GOAL_Q_VS_BUDGET_C.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['# GOAL：已接受Q与增加预算后的C','', '三个seed41/42/43，600回合初态/noise/env_seed/attempt逐项配对，原接受证据SHA核验，指标按原定义用有理数重新计算。无新增训练或环境调用。','', '|指标（%）|Q|增加预算C|Q−C百分点|','|---|---:|---:|---:|']
lines+=['|%s|%.4f|%.4f|%+.4f|'%(k,x['Q'],x['C'],x['Q_minus_C_pp']) for k,x in means.items()]
lines+=['','NBT低好，其余高好。最终均值Q只高0.4167pp，AUC略低0.2083pp；描述上整体接近，不能声称统计等价。Q的NBT低3.1481pp、旧任务最终高1.1111pp，但FWT低1.25pp。NBT基于各自初学水平，Q较低的新任务初学成绩也会影响相对下降，不能仅凭NBT证明保持机制更强。','', '|seed|Q最终|C最终|Q−C百分点|','|---|---:|---:|---:|']
lines+=['|%s|%.2f|%.2f|%+.2f|'%(k,x['Q']['FinalAll'],x['C']['FinalAll'],x['Q_minus_C_pp']['FinalAll']) for k,x in cases.items()]
lines+=['','seed41/42的Q分别低1.25/3.75pp，seed43高6.25pp；小幅均值优势不是跨seed一致。','', '三seed Q/C生产优化器调用228300/426600，Q少46.48%；Q有120次额外训练采集，C没有，且C后续任务200轮并延长学习率调度、Q为100轮。不能把这个差值单独归因于支持度保持项，也不是等训练信息/等计算成本比较。仅GOAL已有完整Q接受结果，OBJECT继续运行。既有开发集重复使用和预训练来源未闭合限制保持。']
(R/'GOAL_Q_VS_BUDGET_C.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],means=means,finals={k:[x['Q']['FinalAll'],x['C']['FinalAll']] for k,x in cases.items()}),indent=2))
