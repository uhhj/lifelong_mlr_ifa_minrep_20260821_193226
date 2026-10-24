"""Local scalar/record audit only; no policy forward, environment or server writes."""
import json,sys,hashlib,statistics
from pathlib import Path
import numpy as np
D=Path(sys.argv[1]);v=json.loads((D/'EVIDENCE.json').read_text());light={}
for n,x in v['light'].items():
 assert hashlib.sha256(x['text'].encode()).hexdigest()==x['sha256']
 if n.endswith('.json'):light[n]=json.loads(x['text'])
initial=light['INITIAL_A.json'];epoch_seconds={}
for arm in 'ABCD':
 assert light['INITIAL_'+arm+'.json']==initial
 train=light['TRAIN_'+arm+'.json'];rows=light['EPOCHS_'+arm+'.json']
 assert train['epochs']==len(rows)==100 and train['steps']==33700
 assert abs(sum(r['seconds'] for r in rows)-train['summed_epoch_seconds'])<1e-7
 assert v['light']['train_'+arm+'.exit']['text']=='0'
 assert light['ADMISSION_'+arm+'.json']['status']=='PASS'
 epoch_seconds[arm]=train['summed_epoch_seconds']
records=light.get('OUTCOMES.json',[]);seen=set();identities={};cells={}
for r in records:
 arm,t,i=r['arm'],r['task'],r['initial_index'];key=(arm,t,i)
 assert key not in seen and arm in 'ABCD' and t in (8,9) and i in range(20);seen.add(key)
 assert r['env_seed']==41+i//10 and r['attempt']==1 and 1<=r['steps']<=600
 assert r['termination']==('success' if r['success'] else 'timeout600') and (r['success'] or r['steps']==600)
 assert r['cp_sha256']==light['TRAIN_'+arm+'.json']['cp_sha256']
 noise=np.random.RandomState(20261024+41000+t*100+i).standard_normal((600,7)).astype(np.float32)
 assert hashlib.sha256(noise.tobytes()).hexdigest()==r['noise_sha256']
 identity=(r['initial_file_sha256'],r['initial_sha256'],r['noise_sha256'],r['env_seed'])
 if (t,i) in identities:assert identities[t,i]==identity
 identities[t,i]=identity;cells.setdefault((arm,t),[]).append(i)
events=[json.loads(x) for x in v['light']['EVAL_EVENTS.jsonl']['text'].splitlines()]
starts={};done=[]
for e in events:
 if e['kind']=='start':assert e['run'] not in starts;starts[e['run']]=e
 else:
  assert e['kind']=='complete' and e['run'] in starts and e['run'] not in [x['run'] for x in done]
  assert e['seconds']>0;done.append(e);assert e['episodes']==len(done)*10
  arm,task,group=e['run'].split('_');t=int(task[1:]);g=int(group[1:])
  assert all((arm,t,i) in seen for i in range(g*10,(g+1)*10))
assert len(records)==10*len(done)
mean=statistics.mean(e['seconds'] for e in done) if done else None
active=[e for n,e in starts.items() if n not in {x['run'] for x in done}];assert len(active)<=1
elapsed=v['observed']-active[0]['time'] if active else 0
remaining=max(0,(16-len(done))*mean-elapsed) if mean is not None else None
out=dict(status='PASS_PROGRESS_NOT_FINAL',observation=v['observed'],episodes=len(records),complete_vectors=len(done),active_vector=active[0]['run'] if active else None,complete_cells=sum(len(ids)==20 for ids in cells.values()),unique_attempts=True,noise_regenerated_sha=True,initial_hash_records_paired=True,initial_file_bytes_and_cpu_cp_final_pending=True,training_epochs=400,main_updates=134800,own_admission_updates=13,epoch_seconds=epoch_seconds,total_epoch_minutes=sum(epoch_seconds.values())/60,vector_mean_seconds=mean,active_elapsed_seconds=elapsed,remaining_linear_seconds=remaining,performance_not_computed=True)
(D/'EVALUATION_PROGRESS_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
