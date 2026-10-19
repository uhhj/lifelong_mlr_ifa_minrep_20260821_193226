"""Exact independent three-seed gate; never launches the conditional next branch."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,time,statistics
Q=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text())
def metrics(root,arm,seed):
 rates={}
 for s in range(6,10):
  for t in range(6,s+1):
   rows=[json.loads(l) for l in (root/arm/'evaluation'/f'trials_stage{s}_task{t}.jsonl').read_text().splitlines()]
   assert len(rows)==20 and {r['trial'] for r in rows}==set(range(20))
   assert all(r['seed']==seed and r['arm']==arm and r['stage']==s and r['task']==t and type(r['success']) is bool for r in rows)
   rates[s,t]=F(sum(r['success'] for r in rows),20)
 m=dict(AUC=sum(sum(rates[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(rates[t,t] for t in range(6,10))/4,NBT=sum(sum(rates[t,t]-rates[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(rates[9,t] for t in range(6,9))/3,FinalAll=sum(rates[9,t] for t in range(6,10))/4)
 return m,[[float(100*rates[s,t]) if t<=s else None for t in range(6,10)] for s in range(6,10)]
def pack(m):return {k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()}
def main():
 parents=js(Q/'PARENTS.json');results={};deltas=[];acc={a:{k:[] for k in ['AUC','FWT','NBT','OldFinal','FinalAll']} for a in ['random','coreset','ra_coreset']}
 for seed in (41,42,43):
  O=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%seed);P=Path(parents[str(seed)]['root'])
  assert js(O/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS'
  audit=js(O/'AUDIT.json');assert audit['status']=='PASS' and len(audit['checkpoints'])==4
  assert all(x['loaded_keys']==100 for x in audit['checkpoints'])
  for n in ['pipeline','supervisor']:assert (O/(n+'.exit')).read_text().strip()=='0'
  ep=js(O/'ra_coreset/epochs.json');assert len(ep)==400 and sum(x['batches'] for x in ep)==105500 and {(x['task'],x['epoch']) for x in ep}=={(t,e) for t in range(6,10) for e in range(100)}
  events=Counter(json.loads(l)['event'] for p in (O/'ra_coreset/evaluation').glob('attempts*.jsonl') for l in p.read_text().splitlines())
  assert events==Counter(vector_attempt_start=20,vector_attempt_complete=20,environment_construct_start=10,environment_construct_complete=10,environment_closed=10)
  values={};matrices={}
  for arm,root in [('random',P),('coreset',P),('ra_coreset',O)]:
   m,matrix=metrics(root,arm,seed);values[arm]=pack(m);matrices[arm]=matrix
   assert all(F(audit['metrics'][arm][k]['fraction'])==v for k,v in m.items())
   for k,v in m.items():acc[arm][k].append(v)
  delta=F(values['ra_coreset']['AUC']['fraction'])-F(values['random']['AUC']['fraction']);deltas.append(delta)
  results[str(seed)]=dict(metrics=values,matrices=matrices,RA_minus_Random_AUC=pack({'delta':delta})['delta'],costs=js(O/'ra_coreset/TRAINING_COMPLETE.json')['seconds_this_invocation'])
 mean=sum(deltas)/3;positive=sum(x>0 for x in deltas);passed=mean>0 and positive>=2
 stats={a:{k:dict(mean_fraction=str(sum(v)/3),mean_percent=float(100*sum(v)/3),sample_SD_percent=statistics.stdev([float(x*100) for x in v])) for k,v in m.items()} for a,m in acc.items()}
 out=dict(status='PASS_AWAITING_EXIT_AND_LOCAL_AUDIT',stage=1,seeds=results,statistics=stats,epochs=1200,steps=316500,checkpoints=12,new_records=600,vector_attempts=60,gate=dict(comparator='Random_IFA',mean_auc_delta_fraction=str(mean),mean_auc_delta_pp=float(100*mean),positive_seeds=positive,all_engineering_pass=True,passed=passed),next_authorized_branch='OBJECT_RA_ZRv2_IFA_41_42_43' if passed else 'GOAL_Coreset_ZRv2_IFA_41_42_43_44',next_stage_launched=False,time=time.time())
 (Q/'STAGE1_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
