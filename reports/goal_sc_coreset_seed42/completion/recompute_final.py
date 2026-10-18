"""Independent exact-metric/metadata acceptance. No tensor load or environment calls."""
import json,hashlib,csv,math
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
R=Path(__file__).resolve().parent;E=R/'final_evidence';P=R/'baseline_evidence'
def js(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
manifest=js(E/'PACKAGE_MANIFEST.json')
for n,h in manifest['files'].items():
 p=E/n;assert p.stat().st_size==h['bytes'] and sha(p)==h['sha256'],n
contract=js(E/'CONTRACT.json');binding=js(E/'PARENT_BINDING.json')
for p in P.rglob('*'):
 if p.is_file():assert sha(p)==contract['files'][binding['parent']+'/'+p.relative_to(P).as_posix()],str(p)
a=js(E/'AUDIT.json');exit_audit=js(E/'PROCESS_EXIT_ACCEPTANCE.json')
assert a['status']==exit_audit['status']=='PASS' and exit_audit['no_remaining_processes']
for phase in ['train','evaluate','audit','pipeline']:assert (E/(phase+'.exit')).read_text().strip()=='0'
assert js(E/'TASK6_TENSOR_PARITY.json')['equal']
assert all(x==0 for x in js(E/'TASK6_TENSOR_PARITY.json')['max_abs_diff'].values())
assert js(E/'sc_coreset/INITIAL_STATE.json')==js(P/'coreset/INITIAL_STATE.json')
ep=js(E/'sc_coreset/epochs.json')
assert {(x['task'],x['epoch']) for x in ep}=={(t,e) for t in range(6,10) for e in range(100)} and len(ep)==400
assert sum(x['batches'] for x in ep)==ep[-1]['steps']==76100
assert len(a['checkpoints'])==4 and all(x['loaded_keys']==100 for x in a['checkpoints'])
cp={x['task']:x['sha256'] for x in a['checkpoints']}
selection={};expected_payload={}
for t in range(6,10):
 m=js(E/'sc_coreset/storage'/f'task{t}.json');rep=m['selection'];b=js(E/'binding_metadata'/f'task{t}.json')['windows']
 pre=js(E/'selection_preflight'/f'task{t}.json')
 assert rep['original_ids']==pre['original_ids'] and rep['selected_ids']==pre['selected_ids']
 ids=[Path(x['source']).name for x in m['selected']];n=len(b);q=[499,445,452,985][t-6]
 assert ids==rep['selected_ids'] and len(ids)==len(set(ids))==m['K']==q
 pop=Counter(str(x['group']) for x in b.values());counts=Counter(str(b[i]['group']) for i in ids)
 assert dict(pop)==rep['population'] and dict(counts)==rep['final']==rep['target']
 for g,c in pop.items():assert q*c//n<=counts[g]<=(q*c+n-1)//n and abs(counts[g]*n-q*c)<n
 assert len(set(rep['original_ids'])-set(ids))==rep['minimum_swaps']==len(rep['swaps'])
 expected_payload.update({x['destination']:x['sha256'] for x in m['selected']})
 assert js(E/'sc_coreset'/f'buffer_after_task{t}.json')['hashes']==expected_payload
 assert m['modal_sha256']==sha(E/'MODAL_CONTRACT.json') and m['contract_sha256']==sha(E/'CONTRACT.json')
 selection[t]={k:rep[k] for k in ['minimum_swaps','radius_before','radius_after','demo_full','demo_original','demo_final','seconds']}

def extract(root,arm):
 rates={};records={};events=[]
 for s in range(6,10):
  for t in range(6,s+1):
   rr=lines(root/arm/'evaluation'/f'trials_stage{s}_task{t}.jsonl')
   assert len(rr)==20 and {x['trial'] for x in rr}==set(range(20))
   assert all(x['stage']==s and x['task']==t and x['seed']==42 and x['arm']==arm and type(x['success']) is bool for x in rr)
   rates[s,t]=F(sum(x['success'] for x in rr),20)
   for x in rr:
    if arm=='sc_coreset':assert x['checkpoint_sha256']==cp[s] and x['modal_sha256']==sha(E/'MODAL_CONTRACT.json') and x['loaded_downstream_keys']==100
    records[s,t,x['trial']]=x
   events+=lines(root/arm/'evaluation'/f'attempts_stage{s}_task{t}.jsonl')
 assert Counter(x['event'] for x in events)==Counter(environment_construct_start=10,environment_construct_complete=10,vector_attempt_start=20,vector_attempt_complete=20,environment_closed=10)
 for s in range(6,10):
  for t in range(6,s+1):
   for g in [0,1]:
    ee=[x for x in events if x['stage']==s and x['task']==t and x.get('group')==g and x['event']=='vector_attempt_complete'];assert len(ee)==1
    assert ee[0]['env_seed']==42+g and all(ee[0]['successes'][i]==records[s,t,10*g+i]['success'] for i in range(10))
 return rates,records
def metrics(r):
 return {'AUC':sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,'FWT':sum(r[t,t] for t in range(6,10))/4,'NBT':sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,'OldFinal':sum(r[9,t] for t in range(6,9))/3,'FinalAll':sum(r[9,t] for t in range(6,10))/4}
def pack(m):return {k:{'fraction':str(v),'percent':float(100*v)} for k,v in m.items()}
r,rr=extract(E,'sc_coreset');c,cc=extract(P,'coreset');z,zz=extract(P,'random')
for key,x in rr.items():assert x['initial_state_sha256']==cc[key]['initial_state_sha256']==zz[key]['initial_state_sha256']
mm={k:metrics(v) for k,v in [('sc_coreset',r),('coreset',c),('random',z)]}
for arm,m in mm.items():
 for k,v in m.items():assert v==F(a['metrics'][arm][k]['fraction'])
gate={'AUC':mm['sc_coreset']['AUC']>mm['coreset']['AUC'],'OldFinal':mm['sc_coreset']['OldFinal']>=mm['coreset']['OldFinal']}
assert gate==a['criteria'] and all(gate.values())==a['metric_gate']==exit_audit['metric_gate']
matrix=[dict(stage=s,task=t,random=float(100*z[s,t]),coreset=float(100*c[s,t]),sc_coreset=float(100*r[s,t])) for s,t in sorted(r)]
assert matrix==a['matrix']
delta={k:mm['sc_coreset'][k]-mm['coreset'][k] for k in mm['sc_coreset']}
contributions={t:(sum(r[s,t]-c[s,t] for s in range(t,10))/(10-t)/4) for t in range(6,10)}
assert sum(contributions.values())==delta['AUC']
out={'status':'PASS','engineering':'COMPLETED','metric_gate':all(gate.values()),'criteria':gate,'decision':'NO_EXTENSION' if not all(gate.values()) else 'ELIGIBLE_FOR_EXTENSION','epochs':400,'optimizer_steps':76100,'checkpoints':a['checkpoints'],'formal_records':200,'completed_vector_attempts':20,'technical_exceptions':0,'baseline_records_readonly':400,'input_package_files':len(manifest['files']),'initial_model_and_rng_identical':True,'task6_model_tensors_equal':True,'metrics':{k:pack(v) for k,v in mm.items()},'delta_vs_coreset':pack(delta),'matrix':matrix,'task_auc_contribution_pp':{str(t):float(100*v) for t,v in contributions.items()},'storage':selection,'timings':lines(E/'timings.jsonl'),'parent_evidence_provenance':'Existing GOAL seed42 final evidence; every file matches the preregistered CONTRACT hash.'}
(R/'LOCAL_FINAL_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n')
with (R/'final_matrix.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(matrix[0]));w.writeheader();w.writerows(matrix)
print(json.dumps({k:v for k,v in out.items() if k not in ['storage','checkpoints']},indent=2))
