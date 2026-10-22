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
def audit_run(out,parent,seed,suite='OBJECT'):
 O,P=Path(out),Path(parent);audit=js(O/'AUDIT.json');contract=js(O/'CONTRACT.json');accept=js(O/'PROCESS_EXIT_ACCEPTANCE.json');variant=contract['variant']
 assert variant in ('TACR','TACRST') and suite=='OBJECT' and contract['seed']==seed
 assert audit['status']==accept['status']=='PASS' and accept['audit_sha256']==sha(O/'AUDIT.json') and not accept['residual_workers']
 assert (O/'pipeline.exit').read_text().strip()==(O/'supervisor.exit').read_text().strip()=='0'
 assert contract['optimizer_steps']==105500 and contract['bc_target']=='next_a_u_plus_1_masked' and contract['deployed_branch_keys']==14
 assert js(O/'ra_coreset/INITIAL_STATE.json')==js(P/'ra_coreset/INITIAL_STATE.json')
 pre=js(O/'GPU_PREFLIGHT.json');assert pre['status']=='PASS' and pre['optimizer_calls']==8 and pre['main_optimizer_calls']==5 and pre['representation_optimizer_calls']==3 and pre['environment_steps']==0
 assert pre['proof']['active_serialized_full_state_restore_exact'] and pre['proof']['disabled_full_state_exact_RA'] and pre['proof']['deploy_adapter_same_fusion_exact']
 hist=js(O/'ra_coreset/epochs.json');assert len(hist)==400
 counts=dict(main_updates=0,representation_updates=0,bc_exposures=0,invalid_bc_exposures=0,effect_exposures=0,st_anchors=0,old_key_reads=0);step=0
 for t in range(6,10):
  md=js(O/'ra_coreset/dataset_manifests'/('task%d.json'%t));rows=md['rows'];n=[1562,2261,3081,3620][t-6];assert len(rows)==n
  assert [r['index'] for r in rows]==list(range(n))
  m=js(O/'ra_coreset/storage'/('task%d.json'%t));base=js(P/'ra_coreset/storage'/('task%d.json'%t));assert m['K']==base['K']==[781,740,780,660][t-6]
  assert m['selection']['selected_ids']==base['selection']['selected_ids']
  assert [(Path(x['source']).name,x['destination'],x['sha256']) for x in m['selected']]==[(Path(x['source']).name,x['destination'],x['sha256']) for x in base['selected']]
  assert js(O/'ra_coreset'/('ifa_task%d.json'%t))==js(P/'ra_coreset'/('ifa_task%d.json'%t))
  cp=next(x for x in audit['checkpoints'] if x['task']==t);dp=next(x for x in audit['deployments'] if x['task']==t)
  assert cp['strict_loaded_keys']==100 and dp['strict_loaded_keys']==14 and dp['student_keys']==20 and dp['training_deployment_exact']
  cursor=0;teacher={}
  if variant=='TACRST' and t>6:
   tm=js(O/'teacher_keys'/('stage%d.json'%t));pc=next(x for x in audit['checkpoints'] if x['task']==t-1);pd=next(x for x in audit['deployments'] if x['task']==t-1)
   assert tm['identity']['teacher_main_sha256']==pc['sha256'] and tm['identity']['teacher_aux_sha256']==pd['aux_sha256'] and tm['identity']['teacher_stage']==t-1
   assert tm['dimension']==128 and tm['teacher_optimizer_calls']==0 and tm['teacher_frozen']
   ids=[i for i,r in enumerate(rows) if r['task']<t and r['bc_valid']];assert ids==[x['index'] for x in tm['rows']] and len(ids)==tm['count']
   for x in tm['rows']:assert x['sample_sha256']==rows[x['index']]['sha256']
   teacher=dict(main=pc['sha256'],student=pd['aux_sha256'],keys=tm['sha256'])
  for ep in range(100):
   h=hist[(t-6)*100+ep];stats=js(O/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(t,ep)));assert (h['task'],h['epoch'],h['batches'])==(t,ep,math.ceil(n/10)) and len(stats)==h['batches']
   assert sorted(i for s in stats for i in s['batch_ids'])==list(range(n))
   for b,s in enumerate(stats):
    ix=s['batch_ids'];v=sum(rows[i]['bc_valid'] for i in ix);em=sum(rows[i]['effect_valid'] for i in ix)
    assert s['variant']==variant and s['step']==step and s['bc_valid']==v and s['bc_invalid']==len(ix)-v and s['effect_samples']==em
    assert s['committed_updates']==s['optimizer_trials']==s['representation_optimizer_calls']==s['head_updates']==1 and s['head_steps']==ep*len(stats)+b+1 and s['teacher_hash']==teacher
    anchors=[i for i in ix if md['positive'].get(str(i)) and md['negative'].get(str(i))] if variant=='TACRST' and t>6 else [];partners=[];assert s['cursor_before']==cursor
    for i in anchors:
     ps=md['positive'][str(i)];ns=md['negative'][str(i)];partners.append([ps[cursor%len(ps)]]+[ns[(cursor+k)%len(ns)] for k in range(min(4,len(ns)))]);cursor+=1
    assert s['anchor_ids']==anchors and s['partner_ids']==partners and s['cursor']==cursor and s['cross_anchors']==len(anchors) and s['old_key_reads']==sum(map(len,partners))
    assert 0<s['gate_mean']<=1 and 0<=s['residual_ratio_max']<=.200001
    if not anchors:assert s['st_loss']==0
    for k,val in dict(main_updates=1,representation_updates=1,bc_exposures=v,invalid_bc_exposures=len(ix)-v,effect_exposures=em,st_anchors=len(anchors),old_key_reads=sum(map(len,partners))).items():counts[k]+=val
    step+=1
   assert h['steps']==step
 assert step==105500 and audit['auxiliary']['counts']==counts
 E=O/'ra_coreset/evaluation';trials=load_records(E,'trials*.jsonl');formal=audit_formal(dict(trials=trials,attempts=load_records(E,'attempts*.jsonl'),cells={p.name:js(p) for p in E.glob('cell*.json')}),seed)
 assert formal['records']==200 and formal['completed_vectors']==20 and formal['completed_cells']==10
 r,by=matrix(trials,seed);br,bb=matrix(load_records(P/'ra_coreset/evaluation','trials*.jsonl'),seed)
 for k,row in by.items():
  cp=next(x for x in audit['checkpoints'] if x['task']==row['stage']);dp=next(x for x in audit['deployments'] if x['task']==row['stage'])
  assert row['initial_state_sha256']==bb[k]['initial_state_sha256'] and row['modal_sha256']==bb[k]['modal_sha256']
  assert row['checkpoint_sha256']==cp['sha256'] and row['deployment_residual_sha256']==dp['sha256'] and row['loaded_residual_keys']==14
 current,base=metrics(r),metrics(br);assert all(F(audit['metrics'][k]['fraction'])==v and F(audit['baseline'][k]['fraction'])==base[k] for k,v in current.items())
 return dict(status='INDEPENDENT_FULL_MATRIX_LEDGER_PASS',variant=variant,seed=seed,suite='OBJECT',metrics=pack(current),baseline=pack(base),delta=pack({k:current[k]-base[k] for k in current}),formal=formal,counts=counts,matrix=[dict(stage=s,task=t,current=str(r[s,t]),parent=str(br[s,t])) for s,t in r],main_training_seconds=sum(x['seconds'] for x in hist),audit_sha256=sha(O/'AUDIT.json'),acceptance_sha256=sha(O/'PROCESS_EXIT_ACCEPTANCE.json'),contract_sha256=sha(O/'CONTRACT.json'))
