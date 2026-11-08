"""Actual final acceptance for finite S42/43; never invokes training or environments."""
import json,sys,hashlib,subprocess,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=js(D/'EVIDENCE.json');assert v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT'
subprocess.run([sys.executable,'-X','utf8',str(R/'audit_progress.py'),str(D)],check=True)
a=js(D/'LOCAL_AUDIT.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['epochs']==1200 and a['accepted_updates']==316500 and a['own_isolated_optimizer_calls']==273 and a['own_extra_autograd_calls']==123 and a['physical_training_optimizer_calls']==585900 and a['training_collection_completed_episodes']==120 and a['formal_episodes']==600
assert v['light']['controller.exit']['text']=='0' and not v['processes']
assert a['environment_mutex_audit']==dict(nonoverlap=True,acquired={'collect':12,'evaluate':3},released={'collect':12,'evaluate':3},active=None)
results={}
for key in ['Q1_41', 'Q1_42', 'Q1_43']:
 seed=int(key.split('_')[1]);r=dict(v['cases'][key],light={n[len(key+'/'):]:x for n,x in v['light'].items() if n.startswith(key+'/')});p=js(R/key/'PARENT_BINDING.json');progress=a['counts'][key]
 child=D/('final_seed'+key);child.mkdir(exist_ok=True);light=child/'light';light.mkdir(exist_ok=True)
 for n,x in r['light'].items():
  fp=light/n;fp.parent.mkdir(parents=True,exist_ok=True);fp.write_text(x['text'],encoding='utf8')
 (child/'EVIDENCE.json').write_text(json.dumps(dict(observed=v['observed'])),encoding='utf8')
 (child/'LOCAL_AUDIT.json').write_text(json.dumps(dict(failure=None)),encoding='utf8')
 subprocess.run([sys.executable,'-X','utf8',str(R/'audit_seed_evaluation.py'),str(child),str(seed)],check=True)
 ev=js(child/'EVALUATION_PROGRESS_AUDIT.json')
 assert r['no_residual'] and r['initial_state_bytes_verified'] and r['single_S_only'] and r['environment_calls']==r['optimizer_updates']==0
 assert r['contract_sha256']==v['children'][key] and ev['episodes']==200 and ev['completed_vectors']==20 and ev['closed_cells']==10 and not ev['ongoing'] and not ev['record_event_advance']
 L={n:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in r['light'].items()}
 assert L['train_S.exit']==L['evaluate.exit']=='0' and L['TRAIN_S.json']['gradient_autograd_calls']==36
 events=[json.loads(x) for x in L['COLLECTION_EVENTS.jsonl'].splitlines()];starts={};ends={}
 for x in events:
  assert x['kind'] in ['start','complete'];target=starts if x['kind']=='start' else ends;k=(x['arm'],x['task']);assert k not in target;target[k]=x
 assert set(starts)==set(ends)=={('S',t) for t in range(6,10)}
 for k,x in ends.items():assert x['time']>=starts[k]['time'] and x['attempts']==10
 gg=L['GRADIENTS_S.json'];assert {(x['stage'],x['epoch']) for x in gg}=={(t,ep) for t in range(7,10) for ep in [0,9,49,99]} and sum(x['extra_autograd_calls'] for x in gg)==36
 for x in gg:
  assert x['extra_autograd_calls']==3 and np.isfinite([x['bc_new'],x['exec_new'],x['retain_weighted']]).all()
  assert all(math.isfinite(y) and y>=0 for y in x['norms'].values())
  for k in ['retain_vs_bc','retain_vs_exec']:
   assert math.isfinite(x[k]['dot']) and (x[k]['cosine'] is None or math.isfinite(x[k]['cosine']) and abs(x[k]['cosine'])<=1.000001)
 assert r['strict']['S']['recovery']['CPU_optimizer_scheduler_load_exact']
 assert r['strict']['S']['recovery']['final_hardlink'] and r['strict']['S']['recovery']['history_exact']
 for t in range(6,10):
  m=L[f'MEMORY_S_{t}.json'];mm=r['memory_audits'][f'S_{t}'];cp=r['strict']['S'][str(t)]
  assert cp['strict_keys']==100 and mm['cache_sha256']==m['cache_sha256'] and mm['source_sha256']==m['source_sha256']==cp['sha256'] and mm['count']==m['count'] and mm['sampling_exact'] and mm['scale_exact']
 sh=L['SHORT_HISTORY_ADMISSION_S.json'];assert sh['extra_autograd_calls']==8 and sh['optimizer_calls']==sh['environments']==0 and sh['all_state_restored']
 records=L['OUTCOMES.json'];matrix={};rr={}
 for s in range(6,10):
  for t in range(6,s+1):
   rows=sorted([x for x in records if x['stage']==s and x['task']==t],key=lambda x:x['initial_index']);assert [x['initial_index'] for x in rows]==list(range(20))
   value=F(sum(x['success'] for x in rows),20);rr[s,t]=value;matrix[f'{s}_{t}']=dict(successes=int(value*20),episodes=20,percent=float(value*100))
 metrics=dict(AUC=sum(sum(rr[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(rr[t,t] for t in range(6,10))/4,NBT=sum(sum(rr[t,t]-rr[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(rr[9,t] for t in range(6,9))/3,FinalAll=sum(rr[9,t] for t in range(6,10))/4)
 summaries={};total_expert=0;total_retain=0
 for name,meta in v['diagnostic_files'].items():
  fp=D/name
  if not fp.stem.startswith(key+'_'):continue
  assert sha(fp)==meta['sha256'];rows=js(fp);assert len(rows)==meta['rows'];kind,arm,st=fp.stem[len(key)+1:].split('_');stage=int(st);assert arm=='S' and kind in ['expert','retain']
  expected=p['stages'][st]['old_valid'] if kind=='expert' else sum(L[f'MEMORY_S_{t}.json']['count'] for t in range(6,stage));assert len(rows)==expected
  if kind=='expert':total_expert+=len(rows)
  else:total_retain+=len(rows)
  scored=[];identities=set()
  for row in rows:
   identity=(row['index'],row['task']);assert identity not in identities;identities.add(identity)
   groups=np.asarray(row['component_group_costs' if kind=='expert' else 'component_group_distances'],float);z=np.asarray(row['logits'],float)
   assert groups.shape==(5,3) and z.shape==(5,) and np.isfinite(groups).all() and np.all(groups>=0) and np.isfinite(z).all()
   costs=groups.sum(-1);k=int(z.argmax());best=int(costs.argmin());assert k==row['student_mode'];route=max(0.,.1+float(np.max(np.delete(z,best)))-float(z[best]))
   scored.append(dict(index=row['index'],task=row['task'],execution_cost=float(costs[k]),best_cost=float(costs.min()),regret=float(costs[k]-costs.min()),margin_route=route))
  stats={}
  for t in sorted({x['task'] for x in scored}):
   rows_t=[x for x in scored if x['task']==t];stats[str(t)]=dict(count=len(rows_t),nonbest_count=sum(x['regret']>1e-12 for x in rows_t),statistics={k:dict(mean=float(np.mean([x[k] for x in rows_t])),median=float(np.median([x[k] for x in rows_t])),p90=float(np.percentile([x[k] for x in rows_t],90)),maximum=max(x[k] for x in rows_t)) for k in ['execution_cost','best_cost','regret','margin_route']})
  summaries[fp.stem]=stats;(D/('RECOMPUTED_'+fp.stem+'.json')).write_text(json.dumps(dict(rows=scored,by_task=stats),indent=2),encoding='utf8')
 assert len(summaries)==6 and total_expert==sum(p['stages'][str(t)]['old_valid'] for t in range(7,10))
 cache_windows=sum(x['count'] for x in r['memory_audits'].values());expert_calls=sum(math.ceil(p['stages'][str(t)]['old_valid']/32) for t in range(7,10))
 assert r['diagnostic_windows']==total_expert+total_retain+cache_windows and r['diagnostic_calls']==expert_calls+total_retain+cache_windows
 assert r['gate_audit']['status']=='PASS_INDEPENDENT_RECOMPUTATION' and r['gate_audit']['blocks']==9100 and r['gate_audit']['physical_training_optimizer_calls']==195300
 results[key]=dict(task6_baseline_all_tensors_exact=r['task6_baseline_all_tensors_exact'],gate_audit=r['gate_audit'],matrix=matrix,metrics={k:dict(fraction=str(x),percent=float(x*100)) for k,x in metrics.items()},offline=summaries,epoch_seconds=L['TRAIN_S.json']['summed_epoch_seconds'],collection_cost={str(t):{k:L[f'MEMORY_S_{t}.json'][k] for k in ['count','collection_successes','training_attempts','online_frontend_calls','online_policy_calls','seconds','terminal_task_unused_for_future_training']} for t in range(6,10)},final_CPU=dict(checkpoints=4,recoveries=1,caches=4,calls=r['diagnostic_calls'],windows=r['diagnostic_windows'],seconds=r['diagnostic_seconds']),gradient_records=gg)
out=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',cases=results,epochs=1200,main_updates=316500,physical_training_optimizer_calls=585900,isolated_optimizer_updates=273,formal_extra_autograd=108,admission_extra_autograd=123,training_collection_episodes=120,formal_episodes=600,vectors=60,cells=30,contract=v['contract_sha256'],evidence_sha256=sha(D/'EVIDENCE.json'),limits=['Only OBJECT S q=1 seed41/42/43; lambda_R=0.1; accepted calibrated S read-only. No outcome-based selection.','No retries, refills, outcome filtering of runs, or automatic further seeds.','Historical methods are read-only references; extra acquisition and double candidates prohibit equal-interaction or equal-compute causal claims.','Success filtering does not prove every retained action optimal; cached-feature replay is not raw-camera or physical replay.','Development states and previous method selection remain limitations.'])
out['environment_mutex_audit']=a['environment_mutex_audit']
out['limits'].append('Fresh OBJECT3090 runs reuse historical0.1; initial tensor/RNG and paired episodes exact, actual Task6 tensor parity recorded, not hardware-controlled single-factor causality. Prior engineering evidence reused; each stage own admission required.')
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=out['status'],metrics={k:x['metrics'] for k,x in results.items()}),indent=2))
