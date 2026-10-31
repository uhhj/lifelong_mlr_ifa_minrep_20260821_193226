"""Actual final acceptance for finite S42/43; never invokes training or environments."""
import json,sys,hashlib,subprocess,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=js(D/'EVIDENCE.json');assert v['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT'
subprocess.run([sys.executable,'-X','utf8',str(R/'audit_progress.py'),str(D)],check=True)
a=js(D/'LOCAL_AUDIT.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL'
assert v['scope']=='GOAL_ONLY_OBJECT_CONTINUES' and v['goal_hashes_before']==v['goal_hashes_after']
assert v['goal_acceptance_boundary']['episodes']==600 and not any('/GOAL_' in p['cmd'] for p in v['processes']+v['post_audit_processes'])
for case in ['GOAL_41','GOAL_42','GOAL_43']:
 c=a['counts'][case]
 assert (c['epochs'],c['accepted_updates'],c['physical_optimizer_calls'],c['own_isolated_optimizer_calls'],c['own_extra_autograd_calls'],c['training_collection_completed_episodes'],c['formal_episodes'])==(400,76100,76100,91,41,40,200)
assert v['optimizer_updates']==v['environment_calls']==0
results={}
for case in ['GOAL_41', 'GOAL_42', 'GOAL_43']:
 seed=int(case.split('_')[1]);key=case;r=dict(v['cases'][case],light={n[len(case)+1:]:x for n,x in v['light'].items() if n.startswith(case+'/')});p=js(R/case/'PARENT_BINDING.json');progress=a['counts'][case]
 child=D/('final_case_'+key);child.mkdir(exist_ok=True);light=child/'light';light.mkdir(exist_ok=True)
 for n,x in r['light'].items():
  fp=light/n;fp.parent.mkdir(parents=True,exist_ok=True);fp.write_text(x['text'],encoding='utf8')
 (child/'EVIDENCE.json').write_text(json.dumps(dict(observed=v['observed'])),encoding='utf8')
 (child/'LOCAL_AUDIT.json').write_text(json.dumps(dict(failure=None)),encoding='utf8')
 subprocess.run([sys.executable,'-X','utf8',str(R/'audit_seed_evaluation.py'),str(child),str(seed)],check=True)
 ev=js(child/'EVALUATION_PROGRESS_AUDIT.json')
 assert r['no_residual'] and r['initial_state_bytes_verified'] and r['single_Q_only'] and r['environment_calls']==r['optimizer_updates']==0
 assert r['contract_sha256']==v['children'][key] and ev['episodes']==200 and ev['completed_vectors']==20 and ev['closed_cells']==10 and not ev['ongoing'] and not ev['record_event_advance']
 L={n:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in r['light'].items()}
 assert L['train_Q.exit']==L['evaluate.exit']=='0' and L['TRAIN_Q.json']['gradient_autograd_calls']==36
 events=[json.loads(x) for x in L['COLLECTION_EVENTS.jsonl'].splitlines()];starts={};ends={}
 for x in events:
  assert x['kind'] in ['start','complete'];target=starts if x['kind']=='start' else ends;k=(x['arm'],x['task']);assert k not in target;target[k]=x
 assert set(starts)==set(ends)=={('Q',t) for t in range(6,10)}
 for k,x in ends.items():assert x['time']>=starts[k]['time'] and x['attempts']==10
 gg=L['GRADIENTS_Q.json'];assert {(x['stage'],x['epoch']) for x in gg}=={(t,ep) for t in range(7,10) for ep in [0,9,49,99]} and sum(x['extra_autograd_calls'] for x in gg)==36
 for x in gg:
  assert x['extra_autograd_calls']==3 and np.isfinite([x['bc_new'],x['exec_new'],x['retain_weighted']]).all()
  assert all(math.isfinite(y) and y>=0 for y in x['norms'].values())
  for k in ['retain_vs_bc','retain_vs_exec']:
   assert math.isfinite(x[k]['dot']) and (x[k]['cosine'] is None or math.isfinite(x[k]['cosine']) and abs(x[k]['cosine'])<=1.000001)
 assert r['strict']['Q']['recovery']['final_hardlink'] and r['strict']['Q']['recovery']['history_exact']
 for t in range(6,10):
  m=L[f'MEMORY_Q_{t}.json'];mm=r['memory_audits'][f'Q_{t}'];cp=r['strict']['Q'][str(t)]
  assert cp['strict_keys']==100 and mm['cache_sha256']==m['cache_sha256'] and mm['source_sha256']==m['source_sha256']==cp['sha256'] and mm['count']==m['count'] and mm['sampling_exact'] and mm['scale_exact']
 sh=L['SHORT_HISTORY_ADMISSION_Q.json'];assert sh['extra_autograd_calls']==8 and sh['optimizer_calls']==sh['environments']==0 and sh['all_state_restored']
 records=L['OUTCOMES.json'];matrix={};rr={}
 for s in range(6,10):
  for t in range(6,s+1):
   rows=sorted([x for x in records if x['stage']==s and x['task']==t],key=lambda x:x['initial_index']);assert [x['initial_index'] for x in rows]==list(range(20))
   value=F(sum(x['success'] for x in rows),20);rr[s,t]=value;matrix[f'{s}_{t}']=dict(successes=int(value*20),episodes=20,percent=float(value*100))
 metrics=dict(AUC=sum(sum(rr[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(rr[t,t] for t in range(6,10))/4,NBT=sum(sum(rr[t,t]-rr[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(rr[9,t] for t in range(6,9))/3,FinalAll=sum(rr[9,t] for t in range(6,10))/4)
 summaries={};total_expert=0;total_retain=0
 for name,meta in v['diagnostic_files'].items():
  fp=D/name
  if not fp.stem.startswith(case+'_'):continue
  assert sha(fp)==meta['sha256'];rows=js(fp);assert len(rows)==meta['rows'];kind,arm,st=fp.stem[len(case)+1:].split('_');stage=int(st);assert arm=='Q' and kind in ['expert','retain']
  expected=p['stages'][st]['old_valid'] if kind=='expert' else sum(L[f'MEMORY_Q_{t}.json']['count'] for t in range(6,stage));assert len(rows)==expected
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
 assert r['production_audit']['status']=='PASS_INDEPENDENT_Q_NO_GATE_RECOMPUTATION' and r['production_audit']['blocks']==0 and r['production_audit']['physical_training_optimizer_calls']==sum(x['batches']*100 for x in p['stages'].values())
 results[key]=dict(production_audit=r['production_audit'],matrix=matrix,metrics={k:dict(fraction=str(x),percent=float(x*100)) for k,x in metrics.items()},offline=summaries,epoch_seconds=L['TRAIN_Q.json']['summed_epoch_seconds'],collection_cost={str(t):{k:L[f'MEMORY_Q_{t}.json'][k] for k in ['count','collection_successes','training_attempts','online_frontend_calls','online_policy_calls','seconds','terminal_task_unused_for_future_training']} for t in range(6,10)},final_CPU=dict(checkpoints=4,recoveries=1,caches=4,calls=r['diagnostic_calls'],windows=r['diagnostic_windows'],seconds=r['diagnostic_seconds']),gradient_records=gg)
out=dict(status='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE',scope='GOAL_ONLY_OBJECT_PENDING',cases=results,epochs=1200,main_updates=228300,physical_training_optimizer_calls=228300,isolated_optimizer_updates=273,formal_extra_autograd=108,admission_extra_autograd=123,training_collection_episodes=120,formal_episodes=600,vectors=60,cells=30,contract=v['contract_sha256'],evidence_sha256=sha(D/'EVIDENCE.json'),limits=['Only six new Q; accepted S is read-only; canceled mechanism is not resumed.','No retries, refills, outcome filtering of runs, or automatic further seeds.','Q and S match accepted epochs and collection counts, not cache contents or physical optimizer/FLOPs/wall-clock costs.','Success filtering does not prove every retained action optimal; cached-feature replay is not raw-camera or physical replay.','Development states and previous method selection remain limitations.'])
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=out['status'],metrics={k:x['metrics'] for k,x in results.items()}),indent=2))
