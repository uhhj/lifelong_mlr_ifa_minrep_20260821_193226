"""Independent exact outcome/accounting aggregation. No tensor loads or environments."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,statistics,time
Q=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def calc(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def pack(m):return {k:dict(fraction=str(v),percent=float(100*v)) for k,v in m.items()}
def main():
 report={};rawmetrics={};files={}
 for seed in (42,43,44):
  O=Path('/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed%d_%s'%(seed,'20261018' if seed==42 else '20261019'))
  P=Path('/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed%d_20261017'%seed)
  a=js(O/'AUDIT.json');proc=js(O/'PROCESS_EXIT_ACCEPTANCE.json');contract=js(O/'CONTRACT.json')
  assert a['status']==proc['status']=='PASS' and proc['no_remaining_processes']
  for name in ['train','evaluate','audit','pipeline','supervisor']:assert (O/(name+'.exit')).read_text().strip()=='0'
  ep=js(O/'ra_coreset/epochs.json');assert len(ep)==400 and {(x['task'],x['epoch']) for x in ep}=={(t,e) for t in range(6,10) for e in range(100)}
  assert sum(x['batches'] for x in ep)==ep[-1]['steps']==76100
  assert js(O/'TASK6_TENSOR_PARITY.json')['equal'] and js(O/'ra_coreset/INITIAL_STATE.json')==js(P/'coreset/INITIAL_STATE.json')
  assert len(a['checkpoints'])==4 and all(x['loaded_keys']==100 for x in a['checkpoints']);cp={x['task']:x['sha256'] for x in a['checkpoints']}
  binding=js(O/'PARENT_BINDING.json')
  for p,h in binding['protected'].items():assert sha(Path(p))==h,p
  payload={};sel={}
  for t in range(6,10):
   m=js(O/'ra_coreset/storage'/('task%d.json'%t));off=js(O/'offline_selection'/('task%d.json'%t));rep=m['selection']
   ids=[Path(x['source']).name for x in m['selected']];q=[499,445,452,985][t-6]
   assert ids==off['selected_ids']==rep['selected_ids'] and len(set(ids))==len(ids)==q
   assert rep['anchor_ids']==off['anchor_ids']==rep['random_draw_ids'][:q//2] and set(rep['anchor_ids'])<=set(ids)
   assert rep['swap_count']==len(rep['swaps'])==len(set(rep['anchor_ids'])-set(rep['original_ids']))
   assert m['before']==payload
   payload.update({x['destination']:x['sha256'] for x in m['selected']})
   assert js(O/'ra_coreset'/('buffer_after_task%d.json'%t))['hashes']==payload
   sel[t]={k:rep[k] for k in ['quota','anchor_count','swap_count','max_before','max_after','mean_before','mean_after']}
  rates={};records={};mm={}
  for arm,root in [('random',P),('coreset',P),('ra_coreset',O)]:
   r={};rr={};events=[]
   for s in range(6,10):
    for t in range(6,s+1):
     p=root/arm/'evaluation'/('trials_stage%d_task%d.jsonl'%(s,t));ts=rows(p);files[str(p)]=sha(p)
     assert len(ts)==20 and {x['trial'] for x in ts}==set(range(20))
     assert all(x['seed']==seed and x['stage']==s and x['task']==t and x['arm']==arm and type(x['success']) is bool for x in ts)
     r[s,t]=F(sum(x['success'] for x in ts),20)
     for x in ts:
      if arm=='ra_coreset':assert x['checkpoint_sha256']==cp[s] and x['loaded_downstream_keys']==100 and x['modal_sha256']==sha(O/'MODAL_CONTRACT.json')
      rr[s,t,x['trial']]=x
     events+=rows(root/arm/'evaluation'/('attempts_stage%d_task%d.jsonl'%(s,t)))
   assert Counter(x['event'] for x in events)==Counter(environment_construct_start=10,environment_construct_complete=10,vector_attempt_start=20,vector_attempt_complete=20,environment_closed=10)
   for s,t in r:
    for g in (0,1):
     e=[x for x in events if x['event']=='vector_attempt_complete' and x['stage']==s and x['task']==t and x['group']==g];assert len(e)==1 and e[0]['env_seed']==seed+g
     assert all(e[0]['successes'][i]==rr[s,t,10*g+i]['success'] for i in range(10))
   rates[arm]=r;records[arm]=rr;mm[arm]=calc(r)
   assert pack(mm[arm])==a['metrics'][arm]
  for k in records['ra_coreset']:assert len({records[arm][k]['initial_state_sha256'] for arm in records})==1
  rawmetrics[seed]=mm
  report[seed]=dict(metrics={arm:pack(m) for arm,m in mm.items()},delta_vs_coreset=pack({k:mm['ra_coreset'][k]-mm['coreset'][k] for k in mm['coreset']}),delta_vs_random=pack({k:mm['ra_coreset'][k]-mm['random'][k] for k in mm['random']}),matrix=[dict(stage=s,task=t,**{arm:float(100*r[s,t]) for arm,r in rates.items()}) for s,t in sorted(rates['ra_coreset'])],selection=sel,criteria=a['criteria'],timings=rows(O/'timings.jsonl'),checkpoints=a['checkpoints'])
 stats={}
 for arm in ['random','coreset','ra_coreset','RA_minus_Coreset','RA_minus_Random']:
  stats[arm]={}
  for k in ['AUC','FWT','NBT','OldFinal','FinalAll']:
   v=[m['ra_coreset'][k]-m['coreset' if arm=='RA_minus_Coreset' else 'random'][k] if arm.startswith('RA_minus') else m[arm][k] for m in rawmetrics.values()]
   stats[arm][k]=dict(mean_fraction=str(sum(v)/3),mean_percent=float(sum(v)/3*100),sample_sd_pp=statistics.stdev([float(x*100) for x in v]))
 out=dict(status='AGGREGATED_AWAITING_EXIT_AUDIT',time=time.time(),seeds=report,statistics=stats,epochs=1200,optimizer_steps=228300,RA_formal_records=600,RA_vector_attempts=60,checkpoints=12,input_outcome_hashes=files,ddof=1,exploratory_extension_after_seed42=True)
 (Q/'THREE_SEED_RESULTS.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(dict(status=out['status'],statistics=stats),indent=2))
if __name__=='__main__':main()
