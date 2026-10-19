"""Independent outcome/attempt recomputation; no torch, model loads or environments."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,time
Q=Path(__file__).resolve().parent
B=Path('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019')
R=Path('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019')
def js(p):return json.loads(p.read_text())
def main():
 assert js(B/'INDEPENDENT_ACCEPTANCE.json')['status']=='PASS'
 assert js(R/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS'
 audit=js(R/'AUDIT.json');assert audit['status']=='PASS'
 result={};mat=[];total_steps=0;total_epochs=0
 for arm,O in [('random',B),('coreset',B),('ra_coreset',R)]:
  for exitname in ['pipeline','supervisor']:assert (O/(exitname+'.exit')).read_text().strip()=='0'
  epochs=js(O/arm/'epochs.json');assert len(epochs)==400 and {(x['task'],x['epoch']) for x in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
  steps=sum(x['batches'] for x in epochs);assert steps==76100;total_steps+=steps;total_epochs+=len(epochs)
  rates={};cnt=0
  for s in range(6,10):
   for t in range(6,s+1):
    rows=[json.loads(l) for l in (O/arm/'evaluation'/f'trials_stage{s}_task{t}.jsonl').read_text().splitlines()]
    assert len(rows)==20 and {r['trial'] for r in rows}==set(range(20))
    assert all(r['seed']==41 and r['stage']==s and r['task']==t and r['arm']==arm and type(r['success']) is bool for r in rows)
    rates[s,t]=F(sum(r['success'] for r in rows),20);cnt+=len(rows)
    mat.append(dict(arm=arm,stage=s,task=t,successes=sum(r['success'] for r in rows),trials=20))
  events=Counter(json.loads(l)['event'] for p in (O/arm/'evaluation').glob('attempts*.jsonl') for l in p.read_text().splitlines())
  assert events==Counter(vector_attempt_start=20,vector_attempt_complete=20,environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
  m=dict(AUC=sum(sum(rates[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(rates[t,t] for t in range(6,10))/4,NBT=sum(sum(rates[t,t]-rates[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(rates[9,t] for t in range(6,9))/3,FinalAll=sum(rates[9,t] for t in range(6,10))/4)
  assert all(F(audit['metrics'][arm][k]['fraction'])==v for k,v in m.items())
  result[arm]={k:dict(fraction=str(v),percent=float(100*v)) for k,v in m.items()}
 assert total_steps==228300 and total_epochs==1200
 out=dict(status='PASS_AWAITING_QUEUE_EXIT',seed=41,epochs=total_epochs,steps=total_steps,formal_records=600,vector_attempts=60,final_checkpoints=12,metrics=result,matrix=mat,time=time.time())
 (Q/'AGGREGATE.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
