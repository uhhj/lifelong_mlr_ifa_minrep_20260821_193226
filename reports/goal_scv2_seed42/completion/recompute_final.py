"""Independent metadata-only final acceptance and exact comparison; no model/rollout calls."""
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import json,hashlib,csv
R=Path(__file__).resolve().parent;E=R/'final_evidence';P=R/'baseline_evidence';V=R/'scv1_evidence'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def witness(x,y):
 if isinstance(x,dict):
  assert x.keys()==y.keys()
  for k in x:witness(x[k],y[k])
 elif isinstance(x,list):
  assert len(x)==len(y)
  for a,b in zip(x,y):witness(a,b)
 elif isinstance(x,float):assert abs(x-y)<=1e-12
 else:assert x==y
manifest=js(E/'PACKAGE_MANIFEST.json')
for n,v in manifest['files'].items():
 p=E/n;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],n
contract=js(E/'CONTRACT.json');binding=js(E/'PARENT_BINDING.json')
for p in P.rglob('*'):
 if p.is_file():assert sha(p)==contract['files'][binding['parent']+'/'+p.relative_to(P).as_posix()],str(p)
for n,v in js(V/'PROVENANCE.json')['files'].items():assert sha(V/n)==v['sha256']
a=js(E/'AUDIT.json');proc=js(E/'PROCESS_EXIT_ACCEPTANCE.json');v1audit=js(V/'AUDIT.json')
assert a['status']==proc['status']==v1audit['status']=='PASS' and proc['no_remaining_processes']
for phase in ['train','evaluate','audit','pipeline','supervisor']:assert (E/(phase+'.exit')).read_text().strip()=='0'
assert js(E/'TASK6_TENSOR_PARITY.json')['equal']
assert all(x==0 for x in js(E/'TASK6_TENSOR_PARITY.json')['max_abs_diff'].values())
assert js(E/'sc_coreset/INITIAL_STATE.json')==js(P/'coreset/INITIAL_STATE.json')
epochs=js(E/'sc_coreset/epochs.json')
assert len(epochs)==400 and {(x['task'],x['epoch']) for x in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
assert sum(x['batches'] for x in epochs)==epochs[-1]['steps']==76100
assert len(a['checkpoints'])==4 and all(x['loaded_keys']==100 for x in a['checkpoints'])
cp={x['task']:x['sha256'] for x in a['checkpoints']};storage={};payload={}
for t in range(6,10):
 m=js(E/'sc_coreset/storage'/f'task{t}.json');rep=m['selection'];off=js(E/'offline_selection'/f'task{t}.json');b=js(E/'binding_metadata'/f'task{t}.json')['windows']
 ids=[Path(x['source']).name for x in m['selected']];q=[499,445,452,985][t-6]
 assert ids==rep['selected_ids']==off['selected_ids'] and len(ids)==len(set(ids))==m['K']==q
 assert rep['original_ids']==off['original_ids']
 for key in ['before','after','swaps']:witness(rep[key],off[key])
 before,after=rep['before'],rep['after']
 assert after['unique_times']>=before['unique_times']
 assert set(before['endpoint_demos'])<=set(after['endpoint_demos']) and set(before['demo_counts'])<=set(after['demo_counts'])
 assert after['radius']<=before['radius']+1e-12 and after['mean']<=before['mean']+1e-12
 assert F(after['E_numerator'],after['E_denominator'])<F(before['E_numerator'],before['E_denominator'])
 assert len(set(ids)-set(rep['original_ids']))==len(rep['swaps'])
 assert m['before']==payload
 payload.update({x['destination']:x['sha256'] for x in m['selected']})
 assert js(E/'sc_coreset'/f'buffer_after_task{t}.json')['hashes']==payload
 assert m['modal_sha256']==sha(E/'MODAL_CONTRACT.json') and m['contract_sha256']==sha(E/'CONTRACT.json')
 storage[t]=dict(quota=q,changed=len(rep['swaps']),before=before,after=after)
assert len(payload)==2381
def extract(root,arm,check_new=False):
 rates={};records={};events=[]
 for s in range(6,10):
  for t in range(6,s+1):
   rr=lines(root/arm/'evaluation'/f'trials_stage{s}_task{t}.jsonl')
   assert len(rr)==20 and {x['trial'] for x in rr}==set(range(20))
   assert all(x['stage']==s and x['task']==t and x['seed']==42 and x['arm']==arm and type(x['success']) is bool for x in rr)
   rates[s,t]=F(sum(x['success'] for x in rr),20)
   for x in rr:
    if check_new:assert x['checkpoint_sha256']==cp[s] and x['modal_sha256']==sha(E/'MODAL_CONTRACT.json') and x['loaded_downstream_keys']==100
    records[s,t,x['trial']]=x
   events+=lines(root/arm/'evaluation'/f'attempts_stage{s}_task{t}.jsonl')
 assert Counter(x['event'] for x in events)==Counter(environment_construct_start=10,environment_construct_complete=10,vector_attempt_start=20,vector_attempt_complete=20,environment_closed=10)
 for s,t in rates:
  for g in [0,1]:
   ee=[x for x in events if x['stage']==s and x['task']==t and x.get('group')==g and x['event']=='vector_attempt_complete'];assert len(ee)==1
   assert ee[0]['env_seed']==42+g and all(ee[0]['successes'][i]==records[s,t,10*g+i]['success'] for i in range(10))
 return rates,records
def metrics(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def pack(m):return {k:dict(fraction=str(v),percent=float(100*v)) for k,v in m.items()}
rates={};records={}
for name,root,arm in [('SC-v2',E,'sc_coreset'),('Coreset',P,'coreset'),('Random',P,'random'),('SC-v1',V,'sc_coreset')]:
 rates[name],records[name]=extract(root,arm,name=='SC-v2')
for k in records['SC-v2']:
 assert len({records[n][k]['initial_state_sha256'] for n in records})==1
 assert len({records[n][k]['modal_sha256'] for n in records})==1
mm={k:metrics(v) for k,v in rates.items()}
for name,arm in [('SC-v2','sc_coreset'),('Coreset','coreset'),('Random','random')]:
 for k,v in mm[name].items():assert v==F(a['metrics'][arm][k]['fraction'])
for k,v in mm['SC-v1'].items():assert v==F(v1audit['metrics']['sc_coreset'][k]['fraction'])
gate=dict(AUC=mm['SC-v2']['AUC']>F(63,80),OldFinal=mm['SC-v2']['OldFinal']>=F(41,60))
assert gate==a['criteria'] and all(gate.values())==a['metric_gate']==proc['metric_gate']
deltas={n:pack({k:mm['SC-v2'][k]-mm[n][k] for k in mm[n]}) for n in ['Coreset','SC-v1','Random']}
matrix=[dict(stage=s,task=t,**{name:float(100*r[s,t]) for name,r in rates.items()}) for s,t in sorted(rates['SC-v2'])]
contributions={name:{str(t):sum(rates['SC-v2'][s,t]-rates[name][s,t] for s in range(t,10))/(10-t)/4 for t in range(6,10)} for name in ['Coreset','SC-v1']}
for n,c in contributions.items():assert sum(c.values())==mm['SC-v2']['AUC']-mm[n]['AUC']
out=dict(status='PASS',engineering='COMPLETED',decision='NO_EXTENSION' if not all(gate.values()) else 'ELIGIBLE_FOR_EXTENSION',criteria=gate,
 epochs=400,optimizer_steps=76100,formal_records=200,vector_attempts=20,technical_exceptions=0,checkpoints=a['checkpoints'],
 package_files=len(manifest['files']),buffer_payload_files=2381,metrics={k:pack(v) for k,v in mm.items()},delta_pp=deltas,matrix=matrix,
 task_auc_contribution_pp={n:{k:float(100*v) for k,v in c.items()} for n,c in contributions.items()},storage=storage,
 input_hashes_pass=True,actual_checkpoint_load_proof='Server AUDIT: all four ep99 checkpoints strictly loaded 100 keys; no checkpoint downloaded',
 initial_state_equal=True,task6_tensor_equal=True,ifa_selected_pairs_equal={t:x['current']['info']['selected_pairs']==x['parent']['info']['selected_pairs'] for t,x in a['ifa'].items()},
 timings=lines(E/'timings.jsonl'),exits={n:0 for n in ['train','evaluate','audit','pipeline','supervisor']})
(R/'LOCAL_FINAL_ACCEPTANCE.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (R/'final_matrix.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
print(json.dumps({k:v for k,v in out.items() if k not in ['storage','checkpoints']},ensure_ascii=False,indent=2))
