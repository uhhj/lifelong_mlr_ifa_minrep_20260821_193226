"""Independent read-only acceptance and dual-comparator reporting; no ML imports."""
import hashlib,json,statistics,importlib.util
from fractions import Fraction as F
from pathlib import Path
R=Path(__file__).resolve().parent
E=R/'final_evidence'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def lines(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,seed):
 c={}; records={}
 for s in range(6,10):
  for t in range(6,s+1):
   rows=lines(p/f'trials_stage{s}_task{t}.jsonl')
   assert len(rows)==20 and {r['initial_state_index'] for r in rows}==set(range(20))
   for r in rows:
    assert (r['seed'],r['stage'],r['task'],r['epoch'])==(seed,s,t,99) and type(r['success']) is bool
    records[s,t,r['initial_state_index']]=r
   c[s,t]=F(sum(r['success'] for r in rows),20)
 return c,records
def metrics(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
manifest=read(R/'final_manifest.json')
assert all(sha(E/p)==v['sha256'] for p,v in manifest.items())
hashes=read(E/'hash_checks.json')
assert hashes['input_mismatches']==[] and hashes['orchestration_unchanged']
assert all(not v['source_mismatches'] and not v['config_mismatches'] for v in hashes['seeds'].values())
assert read(E/'queue/FINAL_QUEUE_ACCEPTANCE.json')['no_residual_process']
assert (E/'queue/queue.exit').read_text().strip()=='0'
# Verify prior lightweight evidence without invoking or rewriting prior audit outputs.
old_execution=R.parent/'object_zr_coreset_seeds42_43_44/execution'
prior=old_execution/'20261017_zr_coreset_extension'
old=read(prior/'FINAL_RESULTS.json')
assert read(prior/'LOCAL_FINAL_ACCEPTANCE.json')['status']=='COMPLETED'
for p,v in read(prior/'final_manifest.json').items():assert sha(prior/'final_evidence'/p)==v['sha256']
p42=old_execution/'20261017_zr_coreset_seed42'
new={42:load(p42/'final_evidence/run/coreset_ifa_zr/evaluation',42)}
first={42:load(p42/'final_evidence/baseline',42)}
for seed in (43,44):
 new[seed]=load(prior/f'final_evidence/seed{seed}/coreset_ifa_zr/evaluation',seed)
 first[seed]=load(prior/f'launch_evidence/seed{seed}/baseline',seed)
audits={};costs=dict(old['cost'])
for seed in (40,41):
 o=E/f'seed{seed}'; a=o/'coreset_ifa_zr'
 n,nr=load(a/'evaluation',seed); b,br=load(R/f'launch_evidence/seed{seed}/baseline',seed)
 new[seed]=(n,nr);first[seed]=(b,br)
 for k,r in nr.items():
  for f in ('initial_state_sha256','modal_sha256'):assert r[f]==br[k][f]
 epochs=read(a/'epochs.json');assert len(epochs)==400
 assert {(x['task'],x['epoch']) for x in epochs}=={(t,e) for t in range(6,10) for e in range(100)}
 total=0
 for x in epochs:
  assert x['batches']=={6:157,7:227,8:309,9:362}[x['task']]
  total+=x['batches'];assert total==x['steps']
 assert total==105500
 for s,t in n:
  attempts=lines(a/'evaluation'/f'attempts_stage{s}_task{t}.jsonl')
  assert not any(v['event']=='technical_exception' for v in attempts)
  starts=[v for v in attempts if v['event']=='vector_attempt_start'];ends=[v for v in attempts if v['event']=='vector_attempt_complete']
  assert len(starts)==len(ends)==2
  for g in (0,1):
   st=next(v for v in starts if v['group']==g);en=next(v for v in ends if v['group']==g)
   assert st['env_seed']==en['env_seed']==seed+g
   assert st['initial_indices']==list(range(10*g,10*g+10)) and st['unix']<=en['unix'] and en['steps']<=600
   for i,v in enumerate(en['successes']):assert v==nr[s,t,10*g+i]['success']
 checks=lines(o/'DIRECT_CHANNEL_ISOLATION.jsonl');assert len(checks)==44 and all(x['passed'] for x in checks)
 for task,size in {6:0,7:781,8:1521,9:2301}.items():
  cc=[x for x in checks if x['stage']==task]
  assert {x['epoch'] for x in cc}=={-1,9,19,29,39,49,59,69,79,89,99}
  end=next(x for x in cc if x['epoch']==99)
  assert end['full_legal_old_buffer'] and sum(x['n'] for x in end['rows'] if x['kind']=='legal_buffer')==size
 rr=[x for c in checks for x in c['rows']]
 assert all(x['passed'] and x['selected_exact'] for x in rr)
 assert max(x['old_gate'] for x in rr)<=1e-6 and max(x['contribution'] for x in rr)<=1e-6
 assert all(x['old_unchanged'] for x in lines(o/'FROZEN_BRANCH_IDENTITY.jsonl'))
 cp=read(o/'CHECKPOINT_ACCEPTANCE.json');assert cp['status']=='PASS' and len(cp['checkpoints'])==4
 for c in cp['checkpoints']:
  assert c['branches']==[str(t) for t in range(6,c['stage']+1)]
  assert all(v=='torch.float64' for v in c['gate_dtypes'].values())
  assert all(r['checkpoint_sha256']==c['sha256'] for k,r in nr.items() if k[0]==c['stage'])
 audit=read(o/'FINAL_AUDIT.json')
 for k in ('inputs_unchanged','buffer_members_order_payloads_equal','IFA_identity_equal','structural_checks_pass'):assert audit[k]
 proc=read(o/'PROCESS_EXIT_ACCEPTANCE.json');assert proc['status']=='PASS' and proc['remaining']==[]
 assert read(o/'RUN_STATUS.json')['status']=='COMPLETED'
 assert (o/'pipeline.exit').read_text().strip()==(o/'supervisor.exit').read_text().strip()=='0'
 saved=read(o/'METRIC_COMPARISON.json')
 for k,v in metrics(n).items():assert F(saved['new'][k]['fraction'])==v
 for k,v in metrics(b).items():assert F(saved['baseline'][k]['fraction'])==v
 costs[str(seed)]=read(o/'COST.json')
 audits[seed]={'epochs':400,'steps':105500,'checkpoints':4,'episodes':200,'vector_attempts':20,'structural_checks':44,'old_gate_max':max(x['old_gate'] for x in rr),'contribution_max':max(x['contribution'] for x in rr),'status':'PASS'}
primary=dict(first)
primary[43]=load(old_execution/'20261016_seed43_three_arm_repeat_v1/completion_evidence/RC/coreset/evaluation',43)
for k,r in primary[43][1].items():
 for f in ('initial_state_sha256','modal_sha256'):assert r[f]==new[43][1][k][f]
def summarize(baselines):
 rows=[]
 for s in range(40,45):
  bm=metrics(baselines[s][0]);nm=metrics(new[s][0])
  gates=dict(AUC=nm['AUC']>bm['AUC'],FWT=nm['FWT']>=bm['FWT'],NBT=nm['NBT']<bm['NBT'],OldFinal=nm['OldFinal']>=bm['OldFinal'])
  rows.append(dict(seed=s,baseline={k:str(v) for k,v in bm.items()},new={k:str(v) for k,v in nm.items()},delta_pp={k:float(100*(nm[k]-v)) for k,v in bm.items()},gates=gates,all_gates=all(gates.values())))
 stats={}
 for k in metrics(new[42][0]):
  stats[k]={}
  for arm in ('baseline','new','delta_pp'):
   v=[r[arm][k] if arm=='delta_pp' else float(F(r[arm][k])*100) for r in rows]
   stats[k][arm]={'mean':statistics.mean(v),'sample_sd':statistics.stdev(v)}
 return {'rows':rows,'statistics':stats}
summary={'primary_repeat43':summarize(primary),'sensitivity_first43':summarize(first)}
summary['matrices_percent']={str(s):{name:{f'{stage},{task}':float(v*100) for (stage,task),v in src[s][0].items()} for name,src in [('primary_baseline',primary),('first_baseline',first),('ZR',new)]} for s in range(40,45)}
summary['cost']=costs
summary['post_result_seed43_comparator_change']=True
(R/'FINAL_RESULTS.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
assert all(sha(E/p)==v['sha256'] for p,v in manifest.items())
out={'status':'COMPLETED','local_independent_acceptance':'PASS','seeds':[40,41,42,43,44],'total_epochs':2000,'total_steps':527500,'strict_final_checkpoints':20,'formal_episodes':1000,'vector_attempts':100,'technical_exception_attempts':0,'structural_checks':220,'extension_audits':audits,'source_config_input_checks':'PASS','unique_extension_inputs':hashes['unique_inputs'],'queue_exit':0,'input_files_unchanged':True,'no_residual_process':True,'new_rollouts_or_training_in_this_audit':False,'seed42_original_failure_preserved':True}
(R/'LOCAL_FINAL_ACCEPTANCE.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')

server=read(E/'queue/FIVE_SEED_SUMMARY.json')
for key,sk in [('primary_repeat43','primary_rows'),('sensitivity_first43','sensitivity_first43_rows')]:
 for row,remote in zip(summary[key]['rows'],server[sk]):
  assert row['seed']==remote['seed']
  for arm in ('baseline','new'):assert row[arm]==remote[arm]
assert all(sha(E/p)==v['sha256'] for p,v in manifest.items())
print(json.dumps({'acceptance':out,'rows':summary['primary_repeat43']['rows'],'statistics':summary['primary_repeat43']['statistics']},indent=2))
