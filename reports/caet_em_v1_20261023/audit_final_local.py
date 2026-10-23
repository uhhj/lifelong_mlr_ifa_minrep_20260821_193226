"""Independent local complete matrix/count/identity recomputation; no private payloads."""
import sys,json,hashlib,time
from pathlib import Path
from fractions import Fraction
P=Path(sys.argv[1]);F=Path(sys.argv[2]);d=json.loads((P/'EVIDENCE.json').read_text());proof=json.loads((F/'EVIDENCE.json').read_text());assert proof['status']=='PASS' and d['controller_exit']=='0' and not d['own_processes'] and not d['hash_bad'];assert proof['contract']==d['contract_sha256'];assert all(v=='0' for v in d['phase_exits'].values())
download=json.loads((P/'DOWNLOAD.json').read_text());assert all(v['sha256']==v['at_snapshot']['sha256'] for v in download.values())
for name,m in download.items():assert hashlib.sha256((P/'metadata'/name).read_bytes()).hexdigest()==m['sha256']
def js(p):return json.loads(p.read_text())
def metrics(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
arms={};steps={6:157,7:227,8:309,9:362}
for arm in ('B0','B1','B2'):
 root=P/'metadata'/arm;matrix={};epoch_count=updates=0
 for t in range(6,10):
  epochs=js(root/'epochs'/('task%d.json'%t));rows=js(root/'dataset'/('task%d.json'%t))['rows'];assert len(epochs)==100
  av=sum(sum(r['action_mask']) for r in rows);ef=sum(sum(r['effect_mask']) for r in rows)
  for j,e in enumerate(epochs):
   assert e['epoch']==j and e['batches']==steps[t] and e['action_valid']==av and e['action_invalid']==4*len(rows)-av
   if arm!='B0':assert e['effect_valid']==ef
  epoch_count+=len(epochs);updates+=sum(e['batches'] for e in epochs)
  for task in range(6,t+1):
   trials=[json.loads(x) for x in (root/'evaluation'/('trials_stage%d_task%d.jsonl'%(t,task))).read_text().splitlines()];events=[json.loads(x) for x in (root/'evaluation'/('attempts_stage%d_task%d.jsonl'%(t,task))).read_text().splitlines()]
   assert len(trials)==20 and [r['initial_state_index'] for r in trials]==list(range(20));init=proof['initial'][str(task)]
   for i,r in enumerate(trials):assert r['checkpoint_sha256']==proof['checkpoints'][arm][str(t)]['sha256'] and r['initial_file_sha256']==init['file_sha256'] and r['initial_state_sha256']==init['initial_hashes'][i] and r['seed']==41 and r['stage']==t and r['task']==task and 1<=r['vector_group_steps']<=600
   starts=[x for x in events if x['event']=='vector_attempt_start'];ends=[x for x in events if x['event']=='vector_attempt_complete'];assert len(starts)==len(ends)==2 and not any(x['event']=='technical_exception' for x in events)
   for g in (0,1):assert starts[g]['env_seed']==ends[g]['env_seed']==41+g and starts[g]['initial_indices']==list(range(g*10,g*10+10)) and ends[g]['successes']==[r['success'] for r in trials[g*10:g*10+10]]
   assert sum(x['event']=='environment_closed' for x in events)==1;matrix[t,task]=Fraction(sum(r['success'] for r in trials),20)
 assert epoch_count==400 and updates==105500;result=metrics(matrix);server=js(root/'METRICS.json')['metrics'];assert all(Fraction(server[k]['fraction'])==v for k,v in result.items())
 arms[arm]=dict(metrics={k:dict(fraction=str(v),percent=float(v*100)) for k,v in result.items()},success_counts={str(k):int(v*20) for k,v in matrix.items()},main_epochs=epoch_count,main_updates=updates,episodes=200)
diff={a+'-'+b:{k:arms[a]['metrics'][k]['percent']-arms[b]['metrics'][k]['percent'] for k in arms[a]['metrics']} for a,b in [('B1','B0'),('B2','B1'),('B2','B0')]}
out=P.parent/('final_CAET_EM_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir();report=dict(status='PASS_COMPLETE_LOCAL_ACCEPTANCE',arms=arms,paired_differences_pp=diff,contract=d['contract_sha256'],snapshot=P.name,CPU_final=F.name,exploratory_single_seed=True,automatic_extension=False,original_failed_A_preserved=True)
(out/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(report,indent=2));print(out);print(json.dumps(report,indent=2))
