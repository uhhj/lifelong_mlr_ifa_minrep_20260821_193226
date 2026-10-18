"""Follow-up read-only phase/gripper cross-tabs and existing RNG tensor hashes.
Never calls a policy, optimizer or environment; no raw RNG payload published.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
import collections,hashlib,json,pickle
from pathlib import Path
import numpy as np
import torch,h5py
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
d=json.loads((P/'DIAGNOSTICS.json').read_text());hashes={str(P/'DIAGNOSTICS.json'):sha(P/'DIAGNOSTICS.json')};out={'cells':[],'rng':[],'source_hashes':{}}
for c in d['cells']:
 root=Path(c['root']);tasks=json.loads((root/'TASKS.json').read_text());hdf=tasks['6']['paths'][0]
 hashes[hdf]=sha(hdf);assert hashes[hdf]==d['inputs_sha256'][hdf]
 with h5py.File(hdf,'r') as f:acts={w['candidates'][0]['demo']:f['data'][w['candidates'][0]['demo']]['actions'][:] for w in c['window_bindings']}
 profiles={}
 for method in ('population','random','coreset'):
  ww=[w for w in c['window_bindings'] if w['unique'] and (method=='population' or w['selected'][method])]
  counter=collections.Counter();demo=collections.Counter();covered=collections.defaultdict(set)
  for w in ww:
   x=w['candidates'][0];q=min(3,int(x['target']/max(1,x['length']-1)*4));a=acts[x['demo']][x['target'],-1];sign='positive' if a>0 else 'negative' if a<0 else 'zero'
   counter[(q,sign,x['valid']<8)]+=1;demo[x['demo']]+=1
   covered[x['demo']].update(range(x['start'],min(x['start']+8,x['length'])))
  profiles[method]={'cross_tabs':[dict(quartile=q,gripper=sign,padded=pad,n=n) for (q,sign,pad),n in sorted(counter.items())],
    'demo_counts':dict(demo),'temporal_support_coverage':{demo:len(covered[demo])/len(a) for demo,a in acts.items()},
    'uncovered_raw_time_count':{demo:len(a)-len(covered[demo]) for demo,a in acts.items()}}
 out['cells'].append(dict(suite=c['suite'],seed=c['seed'],profiles=profiles))
 for arm in ('random','coreset'):
  for stage in range(6,10):
   p=root/arm/'evaluation'/('rng_before_stage%d_task6.pth'%stage)
   assert p.exists(),p
   hashes[str(p)]=sha(p);rng=torch.load(p,map_location='cpu')
   state_hash={}
   for k,v in rng.items():
    if torch.is_tensor(v):b=v.numpy().tobytes()
    elif k=='cuda':b=b''.join(x.cpu().numpy().tobytes() for x in v)
    else:b=pickle.dumps(v,protocol=4)
    state_hash[k]=hashlib.sha256(b).hexdigest()
   out['rng'].append(dict(suite=c['suite'],seed=c['seed'],arm=arm,stage=stage,source=str(p),tensor_or_state_hashes=state_hash))
for seed in (42,43,44):
 s=Path('/root/mlr_ifa/experiment_workspaces/Coreset_paper_GOAL_three_seed_20261017/seed%d/source'%seed)
 for n in ('coreset_exp/evaluate.py','coreset_exp/eval_metric.py','libero/lifelong/models/policy_head.py'):
  p=s/n;hashes[str(p)]=sha(p);out['source_hashes'][str(p)]=hashes[str(p)]
assert all(sha(p)==v for p,v in hashes.items())
out.update(status='PASS_READONLY',hashes_before_after_equal=True,inputs_sha256=hashes)
assert not (P/'JOINT_RNG.json').exists()
(P/'JOINT_RNG.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(status=out['status'],rng_snapshots=len(out['rng']),cells=len(out['cells']))))
