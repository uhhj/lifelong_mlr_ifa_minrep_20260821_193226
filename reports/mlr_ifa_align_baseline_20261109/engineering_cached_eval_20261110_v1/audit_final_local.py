"""Exact final counts and all outcomes; no training, rollouts or GPU calls."""
from pathlib import Path
import json,sys,subprocess,hashlib
from fractions import Fraction as F
R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1]);js=lambda p:json.loads(p.read_text(encoding='utf8'))
e=js(D/'EVIDENCE.json');assert e['status']=='PASS_REMOTE_CPU_PENDING_LOCAL_AUDIT' and not e['processes']
subprocess.run([sys.executable,str(R/'audit_progress.py'),str(D)],check=True)
a=js(D/'LOCAL_AUDIT.json');assert (a['epochs'],a['production_updates'],a['formal_episodes'])==(2400,544800,1200)
assert a['environment_mutex']==dict(completed=1,active=None)
subprocess.run([sys.executable,str(R/'engineering_cached_eval_20261110_v1/audit_progress.py'),str(D)],check=True)
eng=js(D/'CACHED_ENGINEERING_AUDIT.json');assert eng['combined_environment_mutex']==dict(completed=6,active=None)
assert js(D/'ENGINEERING_AUDIT.json')['rng_boundary']['status']=='PASS_EXACT_PRE_CELL_RNG'
legacy=eng['handoff']['legacy_completed_cases']
assert set(eng['cache_admissions'])==set(a['counts'])-set(legacy) and all(v==10 for v in eng['cache_admissions'].values())
L={n:v['text'] for n,v in e['light'].items()};assert L['coordinator/controller.exit'].strip()=='1' and L['cached_engineering/controller.exit'].strip()=='0' and 'engineering/controller.exit' not in L
cases={}
for case,c in a['counts'].items():
 assert c['epochs']==400 and c['formal_episodes']==200 and c['admissions']==4
 owner='engineering/' if case in legacy else 'cached_engineering/'
 assert L[case+'/train.exit'].strip()==L[owner+case+'_evaluate.exit'].strip()=='0'
 counters=json.loads(L[case+'/OPTIMIZER_CALLS.json'])
 assert counters['production_started']==counters['production_completed']==c['production_updates']
 assert counters['admission_started']==counters['admission_completed']==20
 cpu=e['cpu'][case];assert cpu['status']=='PASS_CPU_CASE_RECONSTRUCTION' and len(cpu['checkpoints'])==4 and len(cpu['initial_states_and_rng'])==10
 records=[];matrix={};attempts=[]
 for n,text in L.items():
  if n.startswith(case+'/random/evaluation/trials_'):records += [json.loads(x) for x in text.splitlines() if x]
  if n.startswith(case+'/random/evaluation/attempts_'):attempts += [json.loads(x) for x in text.splitlines() if x]
 assert sum(x['event']=='vector_attempt_start' for x in attempts)==sum(x['event']=='vector_attempt_complete' for x in attempts)==20
 assert sum(x['event']=='environment_closed' for x in attempts)==10 and not any(x['event']=='technical_exception' for x in attempts)
 for s in range(6,10):
  for t in range(6,s+1):
   rows=sorted([x for x in records if (x['stage'],x['task'])==(s,t)],key=lambda x:x['initial_state_index']);assert [x['initial_state_index'] for x in rows]==list(range(20))
   hashes=cpu['initial_states_and_rng'][str(s)+'_'+str(t)]['initial_hashes'];assert all(x['initial_state_sha256']==hashes[x['initial_state_index']] and x['checkpoint_sha256']==cpu['checkpoints'][str(s)]['sha256'] for x in rows)
   for group in [0,1]:
    done=[x for x in attempts if x['event']=='vector_attempt_complete' and x['stage']==s and x['task']==t and x['group']==group];assert len(done)==1
    assert done[0]['env_seed']==int(case[-2:])+group and done[0]['successes']==[x['success'] for x in rows[group*10:(group+1)*10]] and 1<=done[0]['steps']<=600
   matrix[s,t]=F(sum(x['success'] for x in rows),20)
 m=matrix;metrics=dict(AUC=sum(sum(m[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(m[t,t] for t in range(6,10))/4,NBT=sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(m[9,t] for t in range(6,9))/3,NewFinal=m[9,9],FinalAll=sum(m[9,t] for t in range(6,10))/4)
 cases[case]=dict(metrics={k:dict(fraction=str(v),percent=float(v*100)) for k,v in metrics.items()},matrix={str(s)+'_'+str(t):float(v*100) for (s,t),v in m.items()},records=records)
out=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',evaluation_cache_boundary=dict(legacy_full_window_cases=legacy,cached_cases=eng['cached_cases'],atol=1e-4,rtol=1e-4,bitwise_equivalence_claim=False,unchanged_global_Normal_sampling=True),cases=cases,epochs=2400,production_optimizer=544800,isolated_admission_optimizer=120,training_environments=0,formal_episodes=1200,checkpoints_cpu=24,recoveries_cpu=6,noise_scheme='Historical global action RNG reset per stage; saved cell RNG inspected, not per-trajectory common-noise proof',evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest())
(D/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(out['status'])
