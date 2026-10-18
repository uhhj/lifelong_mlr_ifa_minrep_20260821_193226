"""Read-only exact five-seed reporting, retains both seed43 baseline choices."""
import copy,json,statistics,time
from pathlib import Path
from fractions import Fraction as F
Q=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def parent(seed):
 if seed==40:return Path('/data/mlr_ifa/experiments/Seeds40_41_three_arm_20261016/seed40/rc')
 if seed==41:return Path('/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41/rc')
 return Path(f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{seed}_20261013')
def load(p,seed):
 r={}
 for s in range(6,10):
  for t in range(6,s+1):
   rr=[json.loads(x) for x in (p/f'trials_stage{s}_task{t}.jsonl').read_text().splitlines()]
   assert len(rr)==20 and {x['initial_state_index'] for x in rr}==set(range(20))
   assert all((x['seed'],x['stage'],x['task'],x['epoch'])==(seed,s,t,99) and type(x['success']) is bool for x in rr)
   r[s,t]=F(sum(x['success'] for x in rr),20)
 return r
def metrics(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def summary(rows):
 return {k:{a:dict(mean=statistics.mean(v),sample_sd=statistics.stdev(v)) for a in ('baseline','new','delta_pp') for v in [[r[a][k] if a=='delta_pp' else float(F(r[a][k])*100) for r in rows]]} for k in ('AUC','FWT','NBT','OldFinal','FinalAll')}
def main():
 rows=[];tables={}
 for seed in range(40,45):
  p=parent(seed);o=Path(('/data' if seed==42 else '/root')+f'/mlr_ifa/experiments/ZR_coreset_OBJECT_seed{seed}_20261017')
  assert read(o/'RUN_STATUS.json')['status']=='COMPLETED'
  assert read(o/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS'
  assert read(o/'CHECKPOINT_ACCEPTANCE.json')['status']=='PASS'
  b=load(p/'coreset/evaluation',seed);n=load(o/'coreset_ifa_zr/evaluation',seed);bm=metrics(b);nm=metrics(n)
  saved=read(o/'METRIC_COMPARISON.json')
  for k in bm:assert F(saved['baseline'][k]['fraction'])==bm[k] and F(saved['new'][k]['fraction'])==nm[k]
  rows.append(dict(seed=seed,baseline={k:str(v) for k,v in bm.items()},new={k:str(v) for k,v in nm.items()},delta_pp={k:float(100*(nm[k]-v)) for k,v in bm.items()}))
  tables[seed]={'first_baseline':{f'{s},{t}':float(v*100) for (s,t),v in b.items()},'new':{f'{s},{t}':float(v*100) for (s,t),v in n.items()}}
 primary=copy.deepcopy(rows)
 repeat=load(Path('/data/mlr_ifa/experiments/Seed43_Random_Coreset_repeat_20261016_v1/coreset/evaluation'),43)
 bm=metrics(repeat);r=next(r for r in primary if r['seed']==43)
 r['baseline']={k:str(v) for k,v in bm.items()};r['delta_pp']={k:float(100*(F(r['new'][k])-v)) for k,v in bm.items()}
 tables[43]['repeat1_baseline']={f'{s},{t}':float(v*100) for (s,t),v in repeat.items()}
 out=dict(status='AGGREGATED_PENDING_EXIT_AND_LOCAL_AUDIT',new_extension_episodes=400,new_extension_epochs=800,new_extension_steps=211000,five_seed_new_episodes=1000,primary_rows=primary,primary_statistics=summary(primary),sensitivity_first43_rows=rows,sensitivity_first43_statistics=summary(rows),tables=tables,post_result_seed43_comparator_change=True,exploratory_extension_after_42_43_44_results=True,auto_extend=False,unix=time.time())
 (Q/'FIVE_SEED_SUMMARY.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
