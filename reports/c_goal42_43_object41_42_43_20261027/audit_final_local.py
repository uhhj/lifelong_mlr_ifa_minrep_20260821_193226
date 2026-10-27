"""Actual final acceptance only: recompute all records, costs, counts and matrices."""
import json,sys,hashlib,subprocess
from pathlib import Path
from fractions import Fraction as F
import numpy as np
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();v=js(D/'EVIDENCE.json')
assert v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and not v['processes']
subprocess.run([sys.executable,'-X','utf8',str(R/'audit_progress.py'),str(D)],check=True)
a=js(D/'LOCAL_AUDIT.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['formal_episodes']==1000 and a['isolated_optimizer_calls']==95
assert sum(x['epochs'] for x in a['counts'].values())==2000 and sum(x['steps'] for x in a['counts'].values())==468700
L={n:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in v['light'].items()};assert L['controller.exit']=='0'
def metric(r):return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
matrices={};metrics={};costs={};diag_summary={};pairs={};neg=[];total_calls=total_windows=0
for case,remote in v['cases'].items():
 parent=js(R/case/'PARENT_BINDING.json');prefix=case+'/';stages=parent['stages'];steps=sum(st['batches']*100 for st in stages.values())
 assert remote['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and remote['no_residual'] and remote['environment_calls']==remote['optimizer_updates']==0 and remote['episodes']==200
 assert remote['contract_sha256']==v['children'][case]
 assert a['counts'][case]['epochs']==400 and a['counts'][case]['steps']==steps
 assert a['evaluations'][case]['vectors']==20 and a['evaluations'][case]['cells']==10 and not a['evaluations'][case]['ongoing'] and a['evaluations'][case]['record_event_advance']==0
 assert L[prefix+'train_C.exit']==L[prefix+'evaluate.exit']=='0'
 strict=remote['strict']['C'];assert all(strict[str(t)]['strict_keys']==100 and strict[str(t)]['parameters']==21633611 for t in range(6,10))
 assert strict['recovery']['final_hardlink'] and strict['recovery']['history_exact']
 assert len(remote['teacher_audits'])==3 and remote['task6_between_current_arms_exact']
 grads=L[prefix+'GRADIENTS_C.json'];assert {(x['stage'],x['epoch']) for x in grads}=={(t,e) for t in range(7,10) for e in [0,9,49,99]}
 assert sum(x['extra_autograd_calls'] for x in grads)==48==L[prefix+'TRAIN_C.json']['gradient_autograd_calls']
 for g in grads:assert all(np.isfinite(g[k]['l2']) and g[k]['l2']>=0 and np.isfinite(g[k]['loss']) for k in ['bc','ifa_weighted','exec_weighted','route_weighted'])
 expected_calls=2*sum((stages[str(t)]['old_valid']+31)//32 for t in range(7,10));expected_windows=2*sum(stages[str(t)]['old_valid'] for t in range(7,10))
 assert remote['diagnostic_calls']==expected_calls and remote['diagnostic_windows']==expected_windows;total_calls+=expected_calls;total_windows+=expected_windows
 costs[case]=dict(main_updates=steps,epochs=400,isolated_updates=19,formal_extra_autograd=48,admission_extra_autograd=12,epoch_seconds=L[prefix+'TRAIN_C.json']['summed_epoch_seconds'],teacher_forward_batches=sum(L[prefix+'TEACHER_C_%d.json'%t]['forward_batches'] for t in range(7,10)),cpu_diagnostic_seconds=remote['diagnostic_seconds'],evaluation_vector_seconds=sum(a['evaluations'][case]['vector_seconds']))
 rec=L[prefix+'OUTCOMES.json'];r={};by={}
 for stage in range(6,10):
  for task in range(6,stage+1):
   cell=sorted([x for x in rec if (x['stage'],x['task'])==(stage,task)],key=lambda x:x['initial_index']);assert [x['initial_index'] for x in cell]==list(range(20));by[stage,task]=cell;r[stage,task]=F(sum(x['success'] for x in cell),20)
 matrices[case]={'%d_%d'%k:dict(successes=int(p*20),episodes=20,percent=float(p*100)) for k,p in r.items()};metrics[case]={k:dict(fraction=str(x),percent=float(x*100)) for k,x in metric(r).items()}
 pp={}
 for task in range(6,9):
  for stage in range(task+1,10):
   aa=by[task,task];bb=by[stage,task];assert all((x['initial_sha256'],x['noise_sha256'])==(y['initial_sha256'],y['noise_sha256']) for x,y in zip(aa,bb))
   sf=[x['initial_index'] for x,y in zip(aa,bb) if x['success'] and not y['success']];fs=[x['initial_index'] for x,y in zip(aa,bb) if not x['success'] and y['success']]
   delta=5*(len(fs)-len(sf));pp['%d_%d'%(stage,task)]=dict(delta_pp=delta,SF=sf,FS=fs)
   if delta<0:neg.append(dict(case=case,stage=stage,task=task,delta_pp=delta))
 pairs[case]=pp
 for stage in range(7,10):
  name=case+'_C_'+str(stage);filename='diagnostics/'+name+'.json';meta={k.replace('\\','/'):x for k,x in v['diagnostic_files'].items()}[filename];p=D/filename;assert sha(p)==meta['sha256'];rows=js(p)
  expected={i:x for i,x in enumerate(stages[str(stage)]['rows']) if x['task']<stage and x['bc_valid']};assert len(rows)==meta['rows']==len(expected) and {x['index'] for x in rows}==set(expected)
  scored=[]
  for row in rows:
   assert row['task']==expected[row['index']]['task'] and row['sample_sha256']==expected[row['index']]['sha256']
   c=np.asarray(row['student_component_group_costs']).sum(-1);d=np.asarray(row['student_component_anchor_group_distances']).sum(-1);z=np.asarray(row['student_logits']);k=int(z.argmax());best=int(c.argmin());assert k==row['student_mode']
   route=max(0.,.1+float(np.max(np.delete(z,best)))-float(z[best]));assert np.isfinite([*c,*d,*z,route]).all()
   scored.append(dict(index=row['index'],task=row['task'],sample_sha256=row['sample_sha256'],expert_exec=float(c[k]),expert_best=float(c.min()),regret=float(c[k]-c.min()),margin_route=route,auxiliary_loss=float(c[k])+.1*route,teacher_distance_exec=float(d[k]),not_expert_min=k!=best))
  summaries={str(t):dict(count=len(rr),expert_exec=float(np.mean([x['expert_exec'] for x in rr])),regret=float(np.mean([x['regret'] for x in rr])),margin_route=float(np.mean([x['margin_route'] for x in rr])),not_expert_min=sum(x['not_expert_min'] for x in rr)) for t in range(6,stage) for rr in [[x for x in scored if x['task']==t]] if rr}
  (D/('RECOMPUTED_'+name+'.json')).write_text(json.dumps(dict(rows=scored,by_task=summaries),indent=2),encoding='utf8');diag_summary[name]=summaries
out=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',matrix=matrices,metrics=metrics,paired_within_C=pairs,all_negative_cells=neg,offline=diag_summary,costs=costs,training=dict(epochs=2000,updates=468700,isolated_updates=95,formal_extra_autograd=240,admission_extra_autograd=60,training_environment_episodes=0),evaluation=dict(episodes=1000,vectors=100,cells=50,actual_initial_bytes_verified=True),final_cpu=dict(strict_CPs=20,complete_recoveries=5,teacher_caches=15,diagnostic_calls=total_calls,diagnostic_windows=total_windows),source_sha256=sha(D/'EVIDENCE.json'),no_automatic_extension=True,limits=['Existing development seeds and initial states; no significance claim','C uses no training interaction; S has 40 additional collection episodes per seed','Past hardware results descriptive; actual CP/initial/noise bindings required before paired comparisons'])
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(dict(status=out['status'],metrics=metrics),indent=2))
