"""Independent raw-record recomputation, original evaluation budget, no execution."""
import json,statistics,hashlib
from fractions import Fraction as F
from pathlib import Path
B=Path('/data/mlr_ifa/experiments');Q=Path('/data/mlr_ifa/analyses/Coreset_IFAoff_original_protocol_20261014')
def read(p):return json.loads(p.read_text())
def metrics(r):
 return {'AUC':sum(sum(r[i,j] for i in range(j,10))/(10-j) for j in range(6,10))/4,
 'FWT':sum(r[j,j] for j in range(6,10))/4,
 'NBT':sum(sum(r[j,j]-r[i,j] for i in range(j+1,10))/(9-j) for j in range(6,9))/3}
rows=[];matrices=[];histories={}
for s in (42,43,44):
 data={}
 for cond,name in [('on',f'Coreset_paper_OBJECT_seed{s}_20261013'),('off',f'Coreset_IFAoff_OBJECT_seed{s}_20261014')]:
  root=B/name;summary=read(root/'SUMMARY.json');accept=read(root/'INDEPENDENT_ACCEPTANCE.json');assert accept['status']=='PASS'
  if cond=='off':assert read(root/'TRAINING_ACCEPTANCE.json')['status']=='PASS'
  data[cond]={}
  for arm in ('random','coreset'):
   r={};actual=[]
   for stage in range(6,10):
    for t in range(6,stage+1):
     rr=[json.loads(x) for x in (root/arm/'evaluation'/f'trials_stage{stage}_task{t}.jsonl').read_text().splitlines()]
     assert len(rr)==20 and {x['trial'] for x in rr}==set(range(20))
     for x in rr:assert x['seed']==s and x['epoch']==99 and x['initial_state_index']==x['trial'] and type(x['success']) is bool
     r[stage,t]=F(sum(x['success'] for x in rr),20);actual.extend(rr)
   m=metrics(r);data[cond][arm]=m;histories[s,cond,arm]=actual
   for k,v in m.items():assert str(v)==summary['arms'][arm]['metrics'][k]['fraction']
   matrices.append({'seed':s,'condition':cond,'arm':arm,'matrix':[[float(r[i,j]*100) if (i,j) in r else None for j in range(6,10)] for i in range(6,10)]})
 for arm in ('random','coreset'):
  on={(x['stage'],x['task'],x['trial']):x for x in histories[s,'on',arm]}
  for x in histories[s,'off',arm]:assert x['initial_state_sha256']==on[x['stage'],x['task'],x['trial']]['initial_state_sha256']
 for k in ('AUC','FWT','NBT'):
  vals={cond:{a:float(v[k]*100) for a,v in arms.items()} for cond,arms in data.items()}
  delta={c:data[c]['coreset'][k]-data[c]['random'][k] for c in ('on','off')}
  rows.append({'seed':s,'metric':k,'values_percent':vals,'on_C_minus_R_pp':float(100*delta['on']),
   'off_C_minus_R_pp':float(100*delta['off']),'interaction_off_minus_on_pp':float(100*(delta['off']-delta['on']))})
stats={}
for k in ('AUC','FWT','NBT'):
 selected=[r for r in rows if r['metric']==k];stats[k]={}
 for c in ('on','off'):
  for arm in ('random','coreset'):
   x=[r['values_percent'][c][arm] for r in selected];stats[k][c+'_'+arm]={'mean':statistics.mean(x),'sample_sd':statistics.stdev(x)}
 for key in ('on_C_minus_R_pp','off_C_minus_R_pp','interaction_off_minus_on_pp'):
  x=[r[key] for r in selected];stats[k][key]={'mean':statistics.mean(x),'sample_sd':statistics.stdev(x)}
result={'status':'AWAITING_QUEUE_EXIT_AUDIT','new_episodes':1200,'old_control_episodes':1200,'new_epochs':2400,'new_steps':633000,
 'rows':rows,'matrices':matrices,'statistics':stats,'limits':'Exploratory three seeds; original shared-global action RNG retained. No trajectory-RNG independence or universal/significant causal conclusion. NBT positive means more forgetting.'}
with (Q/'COMPARISON.json').open('x') as f:json.dump(result,f,indent=2)
report=['# IFA关闭与原三seed对照','', '原初态0–19，每cell20次，新增1200条；未重跑旧对照。','',
 '| seed | 指标 | 原Random | 原Coreset | IFA关Random | IFA关Coreset | 原C−R | 关IFA C−R | 交互（关−开） |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 v=r['values_percent'];nums=[v['on']['random'],v['on']['coreset'],v['off']['random'],v['off']['coreset'],r['on_C_minus_R_pp'],r['off_C_minus_R_pp'],r['interaction_off_minus_on_pp']]
 report.append(f"| {r['seed']} | {r['metric']} | "+' | '.join(f'{x:.6f}' for x in nums)+' |')
for m in matrices:
 report.extend(['',f"## seed{m['seed']} {m['condition']} {m['arm']}",'','|阶段|Task6|Task7|Task8|Task9|','|---|---:|---:|---:|---:|'])
 for i,row in enumerate(m['matrix'],6):report.append('| '+str(i)+' | '+' | '.join('—' if v is None else f'{v:.2f}' for v in row)+' |')
report.extend(['','三seed均值与样本SD(ddof=1)见COMPARISON.json；训练seed仅3个，原全局动作RNG未隔离，不能将结果解释为普遍机制或显著性证据。','预训练状态模态元数据未知、硬件及调度补全等原协议边界仍保留。','状态：等待独立队列退出验收。'])
(Q/'REPORT.md').write_text('\n'.join(report)+'\n')
print(json.dumps({'new_records':1200,'status':result['status'],'statistics':stats},indent=2))
