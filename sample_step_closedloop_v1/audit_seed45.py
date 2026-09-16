import hashlib,statistics as st
import numpy as np
from .common import *
def run():
 verify();p=read(OUT/'PROTOCOL_FROZEN.json');p['cells']=[c for c in p['cells'] if c['environment_seed']==45];p['points']=p['points'][:1];counts={};paired={};n=0
 for c in p['cells']:
  dest=OUT/'evaluation'/c['point']/f"task{c['task']}"/c['arm'];meta=read(dest/'COMPLETE.json');tt=read(dest/'TRIALS.json');rr=read(dest/'ROUNDS.json')
  assert len(tt)==50 and [x['trial'] for x in tt]==list(range(50)) and len(rr)==5
  for fn,h in meta['files'].items():assert sha(dest/fn)==h
  total=0
  for i,r in enumerate(rr):
   with np.load(dest/f'round{i}.npz') as z:
    done=z['done'];steps=z['steps'];actions=z['actions'];selected=steps>0;success=done[selected].any(axis=0)
    assert success.tolist()==r['success']==[bool(x['success']) for x in tt[i*10:(i+1)*10]]
    ah=hashlib.sha256();dh=hashlib.sha256()
    for a,d,dt in zip(actions,done,r['action_dtypes']):ah.update(np.asarray(a,dtype=dt).tobytes());dh.update(d.tobytes())
    assert ah.hexdigest()==r['action_sha256'] and dh.hexdigest()==r['done_sha256']
    total+=int(success.sum())
  assert total==meta['successes'];counts[c['environment_seed'],c['task'],c['arm']]=total;paired[c['environment_seed'],c['task'],c['arm']]=(rr,meta);n+=50
 for s in [45]:
  for t in [6,7,8]:
   for a in ARMS[1:]:
    r0,m0=paired[s,t,'MLR'];r1,m1=paired[s,t,a];assert m0['frozen_frontend_hash']==m1['frozen_frontend_hash']
    for x,y in zip(r0,r1):
     for k in ['init_sha256','initial_obs_sha256','policy_seed','policy_rng_before']:assert x[k]==y[k],(s,t,a,k)
 for name in p['points']:
  for a in ARMS:
   d=OUT/'training'/name/a;m=read(d/'COMPLETE.json');ev=rows(d/'events.jsonl');assert [r['horizon'] for r in ev]==list(range(1,301));assert sha(d/'events.jsonl')==m['event_sha256'];assert sha(m['checkpoint']['path'])==m['checkpoint']['sha256']
   if a!='MLR':assert all(r['native_moments_preserved'] and r['kkt']<=1e-8 and r['projection_residual']<=5e-6 for r in ev)
 result=read(OUT/'SUMMARY_SEED45.json')
 for row in result['seeds']:
  s,a=row['seed'],row['arm'];expected=st.mean(counts[s,t,a]/50*100 for t in [6,7,8]);assert abs(expected-row['seen'])<1e-10
 assert n==450
 write(OUT/'INDEPENDENT_AUDIT_SEED45.json',dict(status='PASS',trials=n,all_raw_done_and_action_hashes_recomputed=True,paired_initial_inputs_and_frontends=True,all_training_steps_and_checkpoints_verified=True,success_recomputed_without_analyzer=True))
 print('INDEPENDENT_AUDIT_PASS',n,flush=True)
if __name__=='__main__':run()
