from pathlib import Path
from fractions import Fraction as F
import json,hashlib
Q=Path('/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019')
parents=json.loads((Q/'PARENTS.json').read_text());out={}
for seed,v in parents.items():
 p=Path(v['root']);rows={};files={};results={}
 for arm in ('random','coreset'):
  rates={}
  for s in range(6,10):
   for t in range(6,s+1):
    f=p/arm/'evaluation'/('trials_stage%d_task%d.jsonl'%(s,t));rr=[json.loads(x) for x in f.read_text().splitlines()]
    assert len(rr)==20 and {x['trial'] for x in rr}==set(range(20))
    assert all(x['seed']==int(seed) and x['arm']==arm and x['stage']==s and x['task']==t and type(x['success']) is bool for x in rr)
    rates[s,t]=F(sum(x['success'] for x in rr),20);files[str(f)]=hashlib.sha256(f.read_bytes()).hexdigest()
  m=dict(AUC=sum(sum(rates[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(rates[t,t] for t in range(6,10))/4,NBT=sum(sum(rates[t,t]-rates[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(rates[9,t] for t in range(6,9))/3,FinalAll=sum(rates[9,t] for t in range(6,10))/4)
  summary=json.loads((p/'SUMMARY.json').read_text())
  assert all(F(summary['arms'][arm]['metrics'][k]['fraction'])==m[k] for k in ['AUC','FWT','NBT'])
  results[arm]={k:{'fraction':str(v),'percent':float(v*100)} for k,v in m.items()}
 out[seed]={'parent':str(p),'metrics':results,'input_episode_sha256':files,'records_checked':400}
Path('/tmp/ra_object_baseline_audit.json').write_text(json.dumps({'status':'PASS_READONLY_EPISODE_RECOMPUTE','seeds':out,'new_rollouts':0},indent=2))
print({s:a['metrics']['random']['AUC'] for s,a in out.items()})

