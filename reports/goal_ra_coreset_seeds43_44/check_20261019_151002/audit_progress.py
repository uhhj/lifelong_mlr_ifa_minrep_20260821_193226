"""Independent read-only completed-seed audit plus partial-run accounting/ETA."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,sys,datetime,statistics
R=Path(__file__).resolve().parent;D=R/sys.argv[1]
P0=R.parent/'20261017_coreset_goal_three_seed/final_evidence'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def calc(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def pack(m):return {k:dict(fraction=str(v),percent=float(100*v)) for k,v in m.items()}
snap=js(D/'SNAPSHOT.json');completed={};partial={}
for seed in (43,44):
 E=D/('seed%d'%seed)/'evidence';pm=js(E/'PACKAGE_MANIFEST.json')
 for n,m in pm['files'].items():assert sha(E/n)==m['sha256'] and (E/n).stat().st_size==m['bytes'],n
 c=js(E/'CONTRACT.json');assert c['seed']==seed
 assert snap['seeds'][str(seed)]['frozen_hash_check']=='PASS'
 parity=js(E/'TASK6_TENSOR_PARITY.json') if (E/'TASK6_TENSOR_PARITY.json').exists() else None
 if parity:assert parity['equal'] and all(x==0 for x in parity['max_abs_diff'].values())
 ep=js(E/'ra_coreset/epochs.json');assert len(ep)==len({(x['task'],x['epoch']) for x in ep})
 assert sum(x['batches'] for x in ep)==ep[-1]['steps']
 payload={};stored={}
 for t in range(6,10):
  p=E/'ra_coreset/storage'/('task%d.json'%t)
  if not p.exists():continue
  m=js(p);off=js(E/'offline_selection'/('task%d.json'%t));rep=m['selection'];ids=[Path(x['source']).name for x in m['selected']]
  assert ids==rep['selected_ids']==off['selected_ids'] and len(set(ids))==len(ids)==m['K']
  assert rep['anchor_ids']==off['anchor_ids']==rep['random_draw_ids'][:m['K']//2] and set(rep['anchor_ids'])<=set(ids)
  assert rep['swap_count']==len(rep['swaps'])==len(set(rep['anchor_ids'])-set(rep['original_ids']))
  assert rep['swaps']==off['swaps']
  assert m['before']==payload;payload.update({x['destination']:x['sha256'] for x in m['selected']})
  after=E/'ra_coreset'/('buffer_after_task%d.json'%t)
  if after.exists():assert js(after)['hashes']==payload
  stored[t]=dict(quota=m['K'],swaps=rep['swap_count'],ordered_offline_match=True)
 if not (E/'AUDIT.json').exists():
  partial[seed]=dict(package_files=len(pm['files']),epochs=len(ep),steps=ep[-1]['steps'],task6_tensor_equal=True if parity else None,storage=stored,records=snap['seeds'][str(seed)]['records']);continue
 a=js(E/'AUDIT.json');proc=js(E/'PROCESS_EXIT_ACCEPTANCE.json');assert a['status']==proc['status']=='PASS'
 assert proc['no_remaining_processes'] and len(ep)==400 and ep[-1]['steps']==76100 and len(stored)==4 and len(payload)==2381
 for n in ['train','evaluate','audit','pipeline','supervisor']:assert (E/(n+'.exit')).read_text().strip()=='0'
 cp={x['task']:x['sha256'] for x in a['checkpoints']};assert len(cp)==4 and all(x['loaded_keys']==100 for x in a['checkpoints'])
 P=D/('seed%d'%seed)/'baseline_evidence';parent=js(E/'PARENT_BINDING.json')['parent']
 # Archive only exactly needed original outcome/RNG-fingerprint metadata, hash bound by contract.
 for arm in ['coreset','random']:
  source=P0/('seed%d'%seed)/arm
  srcs=list((source/'evaluation').glob('*.jsonl'))+([source/'INITIAL_STATE.json'] if arm=='coreset' and (source/'INITIAL_STATE.json').exists() else [])
  if not P.exists():assert srcs
  for src in srcs:
   rel=src.relative_to(P0/('seed%d'%seed));assert sha(src)==c['files'][parent+'/'+rel.as_posix()]
   dst=P/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(src.read_bytes())
 for p in P.rglob('*'):
  if p.is_file():assert sha(p)==c['files'][parent+'/'+p.relative_to(P).as_posix()]
 assert js(E/'ra_coreset/INITIAL_STATE.json')==js(P/'coreset/INITIAL_STATE.json')
 rates={};record={};metrics={}
 for arm,root in [('ra_coreset',E),('coreset',P),('random',P)]:
  r={};rr={};ev=[]
  for s in range(6,10):
   for t in range(6,s+1):
    ts=rows(root/arm/'evaluation'/('trials_stage%d_task%d.jsonl'%(s,t)))
    assert len(ts)==20 and {x['trial'] for x in ts}==set(range(20))
    assert all(x['seed']==seed and x['stage']==s and x['task']==t and x['arm']==arm and type(x['success']) is bool for x in ts)
    r[s,t]=F(sum(x['success'] for x in ts),20)
    for x in ts:
     if arm=='ra_coreset':assert x['checkpoint_sha256']==cp[s] and x['loaded_downstream_keys']==100 and x['modal_sha256']==sha(E/'MODAL_CONTRACT.json')
     rr[s,t,x['trial']]=x
    ev+=rows(root/arm/'evaluation'/('attempts_stage%d_task%d.jsonl'%(s,t)))
  assert Counter(x['event'] for x in ev)==Counter(environment_construct_start=10,environment_construct_complete=10,vector_attempt_start=20,vector_attempt_complete=20,environment_closed=10)
  for s,t in r:
   for g in [0,1]:
    es=[x for x in ev if x['event']=='vector_attempt_complete' and x['stage']==s and x['task']==t and x['group']==g]
    assert len(es)==1 and es[0]['env_seed']==seed+g and all(es[0]['successes'][i]==rr[s,t,10*g+i]['success'] for i in range(10))
  rates[arm]=r;record[arm]=rr;metrics[arm]=calc(r);assert pack(metrics[arm])==a['metrics'][arm]
 for k in record['ra_coreset']:assert len({record[arm][k]['initial_state_sha256'] for arm in record})==1
 completed[seed]=dict(status='PASS',epochs=400,steps=76100,records=200,vector_attempts=20,technical_exceptions=0,strict_checkpoint_loads=4,metrics={arm:pack(m) for arm,m in metrics.items()},delta_vs_coreset=pack({k:metrics['ra_coreset'][k]-metrics['coreset'][k] for k in metrics['coreset']}),delta_vs_random=pack({k:metrics['ra_coreset'][k]-metrics['random'][k] for k in metrics['random']}),matrix=[dict(stage=s,task=t,**{arm:float(100*r[s,t]) for arm,r in rates.items()}) for s,t in sorted(rates['ra_coreset'])],storage=stored,task6_tensor_equal=True,initial_rng_equal=True,all_exits=0,timings=rows(E/'timings.jsonl'))
eta={}
if 43 in completed and 44 in partial:
 base=js(D/'seed43/evidence/ra_coreset/epochs.json');ep44=js(D/'seed44/evidence/ra_coreset/epochs.json')
 done={(x['task'],x['epoch']) for x in ep44}
 recent=ep44[-20:];r43={(x['task'],x['epoch']):x for x in base}
 ratio=statistics.median(x['seconds']/r43[x['task'],x['epoch']]['seconds'] for x in recent)
 remaining=sum(x['seconds'] for x in base if (x['task'],x['epoch']) not in done)*ratio
 eval_seconds=next(x['seconds'] for x in completed[43]['timings'] if x['phase']=='evaluate')
 eta=dict(training_remaining_minutes=remaining/60,evaluation_reference_minutes=eval_seconds/60,training_recent_rate_ratio=ratio,total_remaining_range_minutes=[round(remaining*.85/60+eval_seconds*.9/60+2),round(remaining*1.2/60+eval_seconds*1.15/60+5)],basis='seed43 per-task epoch times scaled by seed44 last20 matching epochs; seed43 observed full evaluation wall time; overhead allowance')
out=dict(status='PARTIAL_READONLY_AUDIT_PASS',observation_time=snap['time'],completed=completed,partial=partial,eta=eta,new_training_or_evaluation_calls=0,inputs_and_packages_hash_verified=True)
(D/'PROGRESS_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out,indent=2))
