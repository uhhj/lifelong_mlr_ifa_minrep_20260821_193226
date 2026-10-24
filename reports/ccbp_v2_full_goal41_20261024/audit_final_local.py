"""Full local independent matrix/scalar acceptance; never launches training or envs."""
import json,sys,hashlib,subprocess,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text(encoding='utf8'))
assert v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and v['no_residual'] and v['environment_calls']==v['optimizer_updates']==0
subprocess.run([sys.executable,str(R/'audit_progress.py'),str(D)],check=True)
audit=json.loads((D/'LOCAL_AUDIT.json').read_text(encoding='utf8'));assert audit['status']=='PASS_PROGRESS_NOT_FINAL'
assert all(x['epochs']==400 and x['steps']==76100 for x in audit['counts'].values()) and audit['isolated_optimizer_calls']==70 and audit['formal_episodes']==800
light={n:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in v['light'].items()};assert light['controller.exit']=='0' and light['evaluate.exit']=='0'
for arm in 'BCNV':
 assert light['train_'+arm+'.exit']=='0' and light['TRAIN_'+arm+'.json']['gradient_autograd_calls']==48
 grads=light['GRADIENTS_'+arm+'.json'];assert {(x['stage'],x['epoch']) for x in grads}=={(t,e) for t in range(7,10) for e in [0,9,49,99]}
 assert sum(x['extra_autograd_calls'] for x in grads)==48
 for x in grads:
  assert all(np.isfinite(x[k]['l2']) and x[k]['l2']>=0 and np.isfinite(x[k]['loss']) for k in ['bc','ifa_weighted','exec_weighted','route_weighted'])
 assert all(v['strict'][arm][str(t)]['strict_keys']==100 for t in range(6,10)) and v['strict'][arm]['recovery']['final_hardlink']
assert v['task6_equal_to_ALIGN_and_between_arms']
if 'REPAIR_CONTINUATION.json' in light:
 assert light['RESTORE_ADMISSION.json']['all_fields_exact'] and light['RESTORE_ADMISSION.json']['completed_epochs_repeated']==0
records=light['OUTCOMES.json'];matrix={};metrics={};pairs={};neg=[]
def metric(r):return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
by={}
for arm in 'BCNV':
 r={}
 for stage in range(6,10):
  for task in range(6,stage+1):
   cell=sorted([x for x in records if (x['arm'],x['stage'],x['task'])==(arm,stage,task)],key=lambda x:x['initial_index']);assert [x['initial_index'] for x in cell]==list(range(20))
   by[arm,stage,task]=cell;r[stage,task]=F(sum(x['success'] for x in cell),20)
 matrix[arm]={'%d_%d'%k:dict(successes=int(p*20),episodes=20,percent=float(p*100)) for k,p in r.items()}
 metrics[arm]={k:dict(fraction=str(x),percent=float(x*100)) for k,x in metric(r).items()}
for a,b in [('V','C'),('V','B'),('V','N'),('N','C'),('B','C')]:
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
 expected={7:477,8:899,9:1329}[stage];assert len(rows)==expected
 scored=[]
 for row in rows:
  c=np.asarray(row['student_component_group_costs'],dtype=float).sum(-1);d=np.asarray(row['student_component_anchor_group_distances'],dtype=float).sum(-1);logits=np.asarray(row['student_logits']);assert np.isfinite(c).all() and np.isfinite(d).all() and np.isfinite(logits).all()
  assert row['student_mode']==int(logits.argmax());w=row['quality_weight'];tau=v['teacher_audits'][p.stem]['tau'];assert abs(w-tau/(tau+row['frozen_teacher_cost']))<1e-7
  score=(c+w*d)/(1+w) if arm=='V' else d if arm=='B' else c
  candidates=np.flatnonzero(score==score.min());best=int(candidates[np.argmax(logits[candidates])]);route=float(np.sum((score-score[best])*np.logaddexp(0,logits-logits[best]))/4);k=row['student_mode']
  scored.append(dict(index=row['index'],task=row['task'],expert_exec=float(c[k]),expert_best=float(c.min()),anchor_exec=float(d[k]),score_exec=float(score[k]),score_best=float(score.min()),regret=float(score[k]-score.min()),costgap_route=route,route_is_trained_objective=arm in 'NV'))
 diagnostics[p.stem]=dict(count=len(scored),mean_expert_exec=float(np.mean([r['expert_exec'] for r in scored])),p90_expert_exec=float(np.percentile([r['expert_exec'] for r in scored],90)),mean_regret=float(np.mean([r['regret'] for r in scored])),rows=scored)
 (D/('RECOMPUTED_'+p.stem+'.json')).write_text(json.dumps(diagnostics[p.stem],indent=2),encoding='utf8')
assert len(diagnostics)==12 and v['diagnostic_calls']==688 and v['diagnostic_windows']==21640
support=matrix['V']['9_8']['successes']>matrix['C']['9_8']['successes'] and matrix['V']['9_9']['successes']>=matrix['C']['9_9']['successes']
side=[t for t in [6,7] if matrix['V']['9_'+str(t)]['successes']<matrix['C']['9_'+str(t)]['successes']]
out=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',scientific_main_exploratory_criterion=support,other_old_task_declines_vs_C=side,no_automatic_expansion=True,matrix=matrix,metrics=metrics,paired=pairs,all_negative_cells=neg,training=dict(epochs=1600,updates=304400,own_isolated_optimizer_updates=70,formal_extra_autograd_calls=192,admission_extra_autograd_calls=48,epoch_seconds={a:light['TRAIN_'+a+'.json']['summed_epoch_seconds'] for a in 'BCNV'},teacher_forward_batches=sum(x['forward_batches'] for n,x in light.items() if n.startswith('TEACHER_') and n.endswith('.json'))),evaluation=dict(episodes=800,vectors=80,cells=40,paired_noise=True,initial_bytes_remote_verified=True),final_cpu=dict(strict_CPs=16,diagnostic_calls=v['diagnostic_calls'],diagnostic_windows=v['diagnostic_windows'],diagnostic_seconds=v['diagnostic_seconds']),evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),limits=['seed41 and previously explored initial states; descriptive not confirmatory','proxy costs are not closed-loop guarantees','no uniform-anchor ablation; cannot isolate quality weighting against fixed weights','paired-noise protocol differs from historical globalRNG evaluation; historic ALIGN/RA are external references'])
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8')
lines=['# CCBP-v2 GOAL seed41 完整序列：最终结果','', '独立验收完成；科学支持条件：'+str(support)+'。不自动扩seed。','', '|组别|AUC|FWT|NBT（低好）|OldFinal|FinalAll|','|---|---:|---:|---:|---:|---:|']
for a in 'BCNV':lines.append('|'+a+'|'+'|'.join('%.4f'%metrics[a][k]['percent'] for k in ['AUC','FWT','NBT','OldFinal','FinalAll'])+'|')
lines+=['','全部任务矩阵、配对SF/FS、负单元、旧任务副作用及独立离线重算见LOCAL_FINAL_AUDIT与RECOMPUTED文件。离线改善不能代替闭环改善；科学结论应根据V-C、V-N各项实际差值解释，不能把验收PASS写为方法有效。','', '本轮1600epoch304400主更新，70隔离准入更新、正式192额外分项autograd及准入48次另列；16CP、800回合80vector40cell。当前单seed重复探索，不声称显著性、硬保护或唯一失败机制。']
(D/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(json.dumps({k:out[k] for k in ['status','scientific_main_exploratory_criterion','other_old_task_declines_vs_C','metrics']},indent=2))
