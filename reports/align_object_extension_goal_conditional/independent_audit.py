"""Independent exact metadata ledger and episode recomputation, local or host."""
import hashlib,json,math
from fractions import Fraction as F
from pathlib import Path
from formal_audit import audit_formal
def js(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def metrics(r):return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def pack(m):return {k:dict(fraction=str(v),percent=float(v*100)) for k,v in m.items()}
def load_records(p,pattern):return [json.loads(line) for f in sorted(Path(p).glob(pattern)) for line in f.read_text().splitlines()]
def matrix(rows,seed):
 by={(r['stage'],r['task'],r['trial']):r for r in rows}
 assert len(rows)==len(by)==200 and all(r['seed']==seed and type(r['success']) is bool for r in rows)
 return {(s,t):F(sum(by[s,t,i]['success'] for i in range(20)),20) for s in range(6,10) for t in range(6,s+1)},by
def audit_run(out,parent,seed,suite):
 O,P=Path(out),Path(parent);budget=105500 if suite=='OBJECT' else 76100
 sizes=[1562,2261,3081,3620] if suite=='OBJECT' else [997,1388,1848,3365]
 quotas=[781,740,780,660] if suite=='OBJECT' else [499,445,452,985]
 audit=js(O/'AUDIT.json');exitproof=js(O/'PROCESS_EXIT_ACCEPTANCE.json');contract=js(O/'CONTRACT.json')
 assert audit['status']==exitproof['status']=='PASS' and audit['method']=='RA-TAC-Align' and contract['seed']==seed
 assert exitproof['audit_sha256']==sha(O/'AUDIT.json') and not exitproof['residual_workers']
 assert (O/'pipeline.exit').read_text().strip()==(O/'supervisor.exit').read_text().strip()=='0'
 assert contract['optimizer_steps']==budget and contract['bc_target']=='next_a_u_plus_1_masked'
 assert js(O/'ra_coreset/INITIAL_STATE.json')==js(P/'ra_coreset/INITIAL_STATE.json')
 gate=js(O/'GPU_PREFLIGHT.json');assert gate['status']=='PASS' and gate['optimizer_calls']==gate['main_optimizer_calls']==5 and gate['auxiliary_optimizer_calls']==gate['environment_steps']==0
 assert all(gate['proof'][k] for k in ('disabled_full_state_exact_RA','masked_valid_nll_exact','empty_valid_mask_differentiable_zero','active_serialized_full_state_restore_exact'))
 history=js(O/'ra_coreset/epochs.json');assert len(history)==400
 counts=dict(main_updates=0,bc_exposures=0,invalid_bc_exposures=0,optimizer_trials=0);step=0
 for t in range(6,10):
  md=js(O/'ra_coreset/dataset_manifests'/('task%d.json'%t));rows=md['rows'];n=sizes[t-6]
  assert len(rows)==n and [r['index'] for r in rows]==list(range(n))
  if contract['version']=='ALIGN_OBJECT_THREE_SEED_GOAL_CONDITIONAL_20261022':assert not md['positive'] and not md['negative']
  m=js(O/'ra_coreset/storage'/('task%d.json'%t));base=js(P/'ra_coreset/storage'/('task%d.json'%t))
  assert m['K']==base['K']==quotas[t-6] and m['selection']['selected_ids']==base['selection']['selected_ids']
  assert [(Path(x['source']).name,x['destination'],x['sha256']) for x in m['selected']]==[(Path(x['source']).name,x['destination'],x['sha256']) for x in base['selected']]
  assert js(O/'ra_coreset'/('ifa_task%d.json'%t))==js(P/'ra_coreset'/('ifa_task%d.json'%t))
  assert next(x for x in audit['checkpoints'] if x['task']==t)['strict_loaded_keys']==100
  for ep in range(100):
   h=history[(t-6)*100+ep];assert (h['task'],h['epoch'],h['batches'])==(t,ep,math.ceil(n/10))
   entries=js(O/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(t,ep)));assert len(entries)==h['batches']
   assert [s['step'] for s in entries]==list(range(step,step+h['batches']))
   assert sorted(i for s in entries for i in s['batch_ids'])==list(range(n))
   for s in entries:
    valid=sum(rows[i]['bc_valid'] for i in s['batch_ids'])
    assert s['variant']=='ALIGN' and s['bc_valid']==valid and s['bc_invalid']==len(s['batch_ids'])-valid and s['committed_updates']==s['optimizer_trials']==1
    assert not any(s.get(k,0) for k in ('effect_samples','cross_anchors','extra_partner_forwards','head_updates'))
    counts['main_updates']+=1;counts['bc_exposures']+=valid;counts['invalid_bc_exposures']+=len(s['batch_ids'])-valid;counts['optimizer_trials']+=1
   step+=h['batches'];assert h['steps']==step
 assert step==budget and audit['auxiliary']['counts']==counts
 E=O/'ra_coreset/evaluation';trials=load_records(E,'trials*.jsonl');attempts=load_records(E,'attempts*.jsonl')
 formal=audit_formal(dict(trials=trials,attempts=attempts,cells={f.name:js(f) for f in E.glob('cell*.json')}),seed)
 assert formal['records']==200 and formal['started_vectors']==formal['completed_vectors']==20 and formal['completed_cells']==10
 rates,by=matrix(trials,seed);br,bb=matrix(load_records(P/'ra_coreset/evaluation','trials*.jsonl'),seed)
 for key,row in by.items():
  assert row['initial_state_sha256']==bb[key]['initial_state_sha256'] and row['modal_sha256']==bb[key]['modal_sha256']
  assert row['checkpoint_sha256']==next(x for x in audit['checkpoints'] if x['task']==row['stage'])['sha256']
 current,base=metrics(rates),metrics(br)
 assert all(F(audit['metrics'][k]['fraction'])==v and F(audit['baseline'][k]['fraction'])==base[k] for k,v in current.items())
 return dict(status='INDEPENDENT_FULL_MATRIX_LEDGER_PASS',seed=seed,suite=suite,metrics=pack(current),baseline=pack(base),delta=pack({k:current[k]-base[k] for k in current}),formal=formal,counts=counts,
    matrix=[dict(stage=s,task=t,current=str(rates[s,t]),parent=str(br[s,t])) for s,t in rates],main_training_seconds=sum(x['seconds'] for x in history),
    audit_sha256=sha(O/'AUDIT.json'),acceptance_sha256=sha(O/'PROCESS_EXIT_ACCEPTANCE.json'),contract_sha256=sha(O/'CONTRACT.json'))
