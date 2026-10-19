"""Independent stdlib-only episode arithmetic and preauthorized gates."""
import json,hashlib,statistics
from pathlib import Path
from fractions import Fraction as F
def metrics(folder,seed,arm):
 rates={};rows={};hashes={}
 files=list(Path(folder).glob('trials*.jsonl'));assert len(files)==10
 for s in range(6,10):
  for t in range(6,s+1):
   f=Path(folder)/('trials_stage%d_task%d.jsonl'%(s,t));raw=f.read_bytes();hashes[f.name]=hashlib.sha256(raw).hexdigest()
   v=[json.loads(l) for l in raw.decode().splitlines() if l.strip()]
   assert len(v)==20 and {r['trial'] for r in v}==set(range(20))
   assert all((r['seed'],r['arm'],r['stage'],r['task'])==(seed,arm,s,t) and type(r['success']) is bool for r in v)
   assert all(r['trial']==r['initial_state_index'] for r in v)
   rates[s,t]=F(sum(r['success'] for r in v),20)
   for r in v:rows[s,t,r['trial']]=r
 m=dict(AUC=sum(sum(rates[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(rates[t,t] for t in range(6,10))/4,NBT=sum(sum(rates[t,t]-rates[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(rates[9,t] for t in range(6,9))/3,FinalAll=sum(rates[9,t] for t in range(6,10))/4)
 return m,rates,rows,hashes
def gate(deltas,engineering):
 assert len(deltas)==3
 if engineering is not True:raise RuntimeError('Engineering failure cannot select a performance fallback')
 assert all(isinstance(d,F) for d in deltas)
 mean=sum(deltas)/3;count=sum(d>0 for d in deltas)
 return dict(passed=mean>0 and count>=2,mean_auc_delta_fraction=str(mean),mean_auc_delta_pp=float(mean*100),positive_seeds=count,all_engineering_pass=True)
def pack(m):return {k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()}
def stats(ms):return {k:dict(mean_fraction=str(sum(m[k] for m in ms)/len(ms)),mean_percent=float(sum(m[k] for m in ms)/len(ms)*100),sample_SD_percent=statistics.stdev(float(m[k]*100) for m in ms)) for k in ms[0]}
