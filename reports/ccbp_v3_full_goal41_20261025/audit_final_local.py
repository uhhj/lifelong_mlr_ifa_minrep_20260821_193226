"""Full local independent matrix/scalar acceptance; never launches training or envs."""
import json,sys,hashlib,subprocess,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
assert v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and v['no_residual'] and v['environment_calls']==v['optimizer_updates']==0
subprocess.run([sys.executable,str(R/'audit_progress.py'),str(D)],check=True)
audit=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'));assert audit['status']=='PASS_PROGRESS_NOT_FINAL'
assert all(x['epochs']==400 and x['steps']==76100 for x in audit['counts'].values()) and audit['isolated_optimizer_calls']==60 and audit['formal_episodes']==600
subprocess.run([sys.executable,str(R/'audit_evaluation_progress.py'),str(D)],check=True)
ev=json.loads((D/'EVALUATION_PROGRESS_AUDIT.json').read_text(encoding='utf8'));assert ev['completed_vectors']==60 and ev['closed_cells']==30 and not ev['ongoing']
light={n:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in v['light'].items()};assert light['controller.exit']=='0' and light['evaluate.exit']=='0'
for arm in 'CRG':
 assert light['train_'+arm+'.exit']=='0' and light['TRAIN_'+arm+'.json']['gradient_autograd_calls']==(60 if arm=='G' else 48)
 grads=light['GRADIENTS_'+arm+'.json'];assert {(x['stage'],x['epoch']) for x in grads}=={(t,e) for t in range(7,10) for e in [0,9,49,99]}
 assert sum(x['extra_autograd_calls'] for x in grads)==(60 if arm=='G' else 48)
 for x in grads:
  assert all(np.isfinite(x[k]['l2']) and x[k]['l2']>=0 and np.isfinite(x[k]['loss']) for k in (['bc','ifa_weighted','exec_weighted','route_weighted']+(['anchor_weighted'] if arm=='G' else [])))
 assert all(v['strict'][arm][str(t)]['strict_keys']==100 for t in range(6,10)) and v['strict'][arm]['recovery']['final_hardlink']
assert v['task6_between_current_arms_exact']
assert set(v['historical_task6'])==set('CRG') and all(x['equality_gate'] is False for x in v['historical_task6'].values())
if 'REPAIR_CONTINUATION.json' in light:
 assert light['RESTORE_ADMISSION.json']['all_fields_exact'] and light['RESTORE_ADMISSION.json']['completed_epochs_repeated']==0
records=light['OUTCOMES.json'];matrix={};metrics={};pairs={};neg=[]
def metric(r):return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
by={}
for arm in 'CRG':
 r={}
 for stage in range(6,10):
  for task in range(6,stage+1):
   cell=sorted([x for x in records if (x['arm'],x['stage'],x['task'])==(arm,stage,task)],key=lambda x:x['initial_index']);assert [x['initial_index'] for x in cell]==list(range(20))
   by[arm,stage,task]=cell;r[stage,task]=F(sum(x['success'] for x in cell),20)
 matrix[arm]={'%d_%d'%k:dict(successes=int(p*20),episodes=20,percent=float(p*100)) for k,p in r.items()}
 metrics[arm]={k:dict(fraction=str(x),percent=float(x*100)) for k,x in metric(r).items()}
for a,b in [('G','C'),('G','R'),('R','C')]:
 cells={}
 for stage in range(6,10):
  for task in range(6,stage+1):
   aa=by[a,stage,task];bb=by[b,stage,task]
   assert all((x['initial_sha256'],x['noise_sha256'])==(y['initial_sha256'],y['noise_sha256']) for x,y in zip(aa,bb))
   sf=[x['initial_index'] for x,y in zip(aa,bb) if y['success'] and not x['success']];fs=[x['initial_index'] for x,y in zip(aa,bb) if not y['success'] and x['success']]
   delta=(len(fs)-len(sf))*5;cells['%d_%d'%(stage,task)]=dict(delta_pp=delta,reference_success_to_candidate_failure=sf,reference_failure_to_candidate_success=fs)
   if delta<0:neg.append(dict(comparison=a+'-'+b,stage=stage,task=task,delta_pp=delta))
 pairs[a+'-'+b]=cells
diagnostics={}
for name,meta in v['diagnostic_files'].items():
 p=D/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256'];rows=json.loads(p.read_text(encoding='utf8'));assert len(rows)==meta['rows'];arm,stage=p.stem.split('_');stage=int(stage)
 assert arm in 'CRG' and len(rows)=={7:477,8:899,9:1329}[stage]
 scored=[]
 for row in rows:
  c=np.asarray(row['student_component_group_costs'],dtype=float).sum(-1)
  d=np.asarray(row['student_component_anchor_group_distances'],dtype=float).sum(-1)
  tc=np.asarray(row['teacher_component_group_costs'],dtype=float).sum(-1)[row['teacher_mode']]
  z=np.asarray(row['student_logits'],dtype=float);k=int(z.argmax());assert k==row['student_mode']
  w=row['quality_weight'];tau=v['teacher_audits'][p.stem]['tau'];assert abs(w-tau/(tau+row['frozen_teacher_cost']))<1e-7
  dot=(d+tc-c)/2;alpha=np.clip(dot/max(tc,1e-300),0,1)
  anchor=d-2*alpha*dot+alpha**2*tc
  assert np.allclose(anchor,row['directional_distances'],atol=1e-9,rtol=1e-9)
  assert np.allclose(alpha,row['projection_fractions'],atol=1e-9,rtol=1e-9)
  assert np.all(anchor>=-1e-9) and np.all(anchor<=c+1e-8)
  score=(c+w*d)/(1+w) if arm=='R' else c
  choices=np.flatnonzero(score==score.min())
  best=int(choices[np.argmax(z[choices])]) if arm=='R' else int(score.argmin())
  hinge=max(0.,.1+float(np.max(np.delete(z,best)))-float(z[best]))
  ex=float(score[k]);keep=float(w*anchor[k]) if arm=='G' else 0.
  assert np.isfinite([ex,keep,hinge,*c,*d,*anchor]).all()
  scored.append(dict(index=row['index'],task=row['task'],expert_exec=float(c[k]),expert_best=float(c.min()),teacher_distance_exec=float(d[k]),directional_exec=float(anchor[k]),projection_fraction=float(alpha[k]),score_exec=ex,score_best=float(score.min()),regret=float(c[k]-c.min()),margin_route=hinge,weighted_anchor=keep,auxiliary_loss=ex+keep+.1*hinge))
 diagnostics[p.stem]=dict(count=len(scored),mean_expert_exec=float(np.mean([r['expert_exec'] for r in scored])),p90_expert_exec=float(np.percentile([r['expert_exec'] for r in scored],90)),mean_regret=float(np.mean([r['regret'] for r in scored])),rows=scored)
 (D/('RECOMPUTED_'+p.stem+'.json')).write_text(json.dumps(diagnostics[p.stem],indent=2),encoding='utf8')
assert len(diagnostics)==9 and v['diagnostic_calls']==516 and v['diagnostic_windows']==16230
support=(metrics['G']['OldFinal']['percent']>metrics['C']['OldFinal']['percent'] and matrix['G']['9_8']['successes']>=matrix['C']['9_8']['successes'] and matrix['G']['9_9']['successes']>=matrix['C']['9_9']['successes'])
side=[t for t in [6,7,8] if matrix['G']['9_'+str(t)]['successes']<matrix['C']['9_'+str(t)]['successes']]
out=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',scientific_main_exploratory_criterion=support,other_old_task_declines_vs_C=side,no_automatic_expansion=True,matrix=matrix,metrics=metrics,paired=pairs,all_negative_cells=neg,training=dict(epochs=1200,updates=228300,own_isolated_optimizer_updates=60,formal_extra_autograd_calls=156,admission_extra_autograd_calls=39,epoch_seconds={a:light['TRAIN_'+a+'.json']['summed_epoch_seconds'] for a in 'CRG'},teacher_forward_batches=sum(x['forward_batches'] for n,x in light.items() if n.startswith('TEACHER_') and n.endswith('.json'))),evaluation=dict(episodes=600,vectors=60,cells=30,paired_noise=True,initial_bytes_remote_verified=True),hardware_policy='Current-card C/R/G comparisons; historical cross-card equality descriptive only',historical_task6=v['historical_task6'],final_cpu=dict(strict_CPs=12,diagnostic_calls=v['diagnostic_calls'],diagnostic_windows=v['diagnostic_windows'],diagnostic_seconds=v['diagnostic_seconds']),evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),limits=['seed41 and previously explored initial states; descriptive not confirmatory','proxy costs are not closed-loop guarantees','R-to-G changes geometry, scaling and routing score; cannot attribute only to geometry; rolling teachers are endogenous','paired-noise protocol differs from historical globalRNG evaluation; historic ALIGN/RA are external references'])
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8')
lines=['# CCBP-v3 GOAL seed41 完整序列：最终结果','', '独立验收完成；科学支持条件：'+str(support)+'。不自动扩seed。','', '|组别|AUC|FWT|NBT（低好）|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for a in 'CRG':lines.append('|'+a+'|'+'|'.join('%.4f'%metrics[a][k]['percent'] for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+'|')
lines+=['','全部任务矩阵、配对SF/FS、负单元、旧任务副作用及独立离线重算见LOCAL_FINAL_AUDIT与RECOMPUTED文件。离线改善不能代替闭环改善；科学结论应根据G-C、G-R、R-C各项实际差值解释，不能把验收PASS写为方法有效。','', '本轮1200epoch228300主更新，60隔离准入更新、正式156额外分项autograd及准入39次另列；12CP、600回合60vector30cell。当前单seed重复探索，不声称显著性、硬保护或唯一失败机制。']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps({k:out[k] for k in ['status','scientific_main_exploratory_criterion','other_old_task_declines_vs_C','metrics']},indent=2))
