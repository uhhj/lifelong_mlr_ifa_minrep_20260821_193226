"""Read-only exact three-seed recomputation from complete records."""
import json,statistics,time
from pathlib import Path
from fractions import Fraction as F
Q=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def load(p,s):
 c={}
 for stage in range(6,10):
  for task in range(6,stage+1):
   rows=[json.loads(l) for l in (p/f'trials_stage{stage}_task{task}.jsonl').read_text().splitlines()]
   assert len(rows)==20 and {r['initial_state_index'] for r in rows}==set(range(20))
   assert all((r['seed'],r['stage'],r['task'],r['epoch'])==(s,stage,task,99) and type(r['success']) is bool for r in rows)
   c[stage,task]=F(sum(r['success'] for r in rows),20)
 return c
def metrics(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def main():
 rows=[];tables={}
 for seed in (42,43,44):
  parent=Path(f'/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{seed}_20261013')
  out=Path(('/data' if seed==42 else '/root')+f'/mlr_ifa/experiments/ZR_coreset_OBJECT_seed{seed}_20261017')
  assert read(out/'RUN_STATUS.json')['status']=='COMPLETED'
  assert read(out/'PROCESS_EXIT_ACCEPTANCE.json')['status']=='PASS'
  assert read(out/'CHECKPOINT_ACCEPTANCE.json')['status']=='PASS'
  b=load(parent/'coreset/evaluation',seed);n=load(out/'coreset_ifa_zr/evaluation',seed);bm=metrics(b);nm=metrics(n)
  saved=read(out/'METRIC_COMPARISON.json')
  for k in bm:assert bm[k]==F(saved['baseline'][k]['fraction']) and nm[k]==F(saved['new'][k]['fraction'])
  rows.append(dict(seed=seed,baseline={k:str(v) for k,v in bm.items()},new={k:str(v) for k,v in nm.items()},delta_pp={k:float((nm[k]-bm[k])*100) for k in bm}))
  tables[seed]={'baseline':{f'{s},{t}':float(v*100) for (s,t),v in b.items()},'new':{f'{s},{t}':float(v*100) for (s,t),v in n.items()}}
 stats={}
 for key in ('AUC','FWT','NBT','OldFinal','FinalAll'):
  stats[key]={}
  for arm in ('baseline','new','delta_pp'):
   values=[r[arm][key] if arm=='delta_pp' else float(F(r[arm][key])*100) for r in rows]
   stats[key][arm]=dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values))
 result=dict(status='AGGREGATED_PENDING_QUEUE_EXIT_AND_LOCAL_AUDIT',new_extension_epochs=800,new_extension_steps=211000,new_extension_episodes=400,three_seed_new_episodes=600,baseline_episodes_readonly=600,rows=rows,tables=tables,statistics=stats,exploratory_extension_after_seed42=True,auto_extend=False,unix=time.time())
 (Q/'THREE_SEED_SUMMARY.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
