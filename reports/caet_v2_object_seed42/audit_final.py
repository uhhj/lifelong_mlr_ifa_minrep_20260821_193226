"""Independent exact metrics, metadata ledger and constraint recomputation."""
import json,sys,hashlib,math
from pathlib import Path
from fractions import Fraction as F
from formal_audit import audit_formal
p=Path(sys.argv[1]);manifest=json.loads((p/'PACKAGE_MANIFEST.json').read_text())
def js(path):return json.loads(path.read_text(encoding='utf8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for name,r in manifest['entries'].items():assert sha(p/name)==r['sha256'] and (p/name).stat().st_size==r['bytes']
assert not manifest['residual_workers']
C=p/'current';B=p/'baseline';V=p/'v1'
assert (p/'launcher/recovery_environment01.exit').read_text().strip()=='0'
assert (p/'launcher/supervisor.exit').read_text().strip()=='1' # preserved import-only failure
assert js(p/'launcher/RECOVERY_ENVIRONMENT01.json')['prior_optimizer_calls']==0
assert (C/'pipeline.exit').read_text().strip()=='0'
a=js(C/'AUDIT.json');process=js(C/'PROCESS_EXIT_ACCEPTANCE.json')
assert a['status']==process['status']=='PASS' and process['audit_sha256']==sha(C/'AUDIT.json') and not process['residual_workers']
assert a['method']=='CAET-v2' and a['all_inputs_unchanged']
gate=js(C/'GPU_PREFLIGHT.json');assert gate['status']=='PASS' and gate['optimizer_calls']==8 and gate['environment_steps']==0
assert gate['zero_aux_native_update_exact'] and gate['full_state_epoch_recovery_exact']
assert js(C/'TASK6_TENSOR_PARITY.json')['equal'] and js(C/'ra_coreset/INITIAL_STATE.json')==js(B/'ra_coreset/INITIAL_STATE.json')
h=js(C/'ra_coreset/epochs.json');assert len(h)==400;steps=0;activation={}
for i,row in enumerate(h):
 t,e=6+i//100,i%100;assert (row['task'],row['epoch'])==(t,e)
 assert row['batches']=={6:157,7:227,8:309,9:362}[t];steps+=row['batches'];assert row['steps']==steps
 stats=js(C/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(t,e)));assert [x['step'] for x in stats]==list(range(steps-row['batches'],steps))
 constraints=js(C/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,e)))
 active=t>6 and js(C/'effects/models'/('task%d'%t)/'ACCEPTANCE.json')['status']=='PASS';assert constraints['active']==active
 if active:
  for key,value in constraints['values'].items():
   assert math.isfinite(value)
   assert constraints['mu_after'][key]==max(0.,min(1.,constraints['mu_before'][key]+.1*(value-constraints['bounds'][key]-.05)))
 for x in stats:
  assert x['active']==active
  if not active:assert x['effect_loss'] is None and not x['transfer_applied'] and not x['retention_groups']
  if x['transfer_dot'] is not None:assert x['transfer_supported']>0 and x['transfer_applied']==(x['transfer_dot']>=0)
 r=activation.setdefault(str(t),dict(active_stage=active,transfer=0,blocked=0,retention=0))
 r['transfer']+=sum(x['transfer_applied'] for x in stats);r['blocked']+=sum(x['transfer_dot'] is not None and x['transfer_dot']<0 for x in stats);r['retention']+=sum(x['retention_groups']>0 for x in stats)
assert steps==105500 and not activation['6']['active_stage'] and not activation['7']['active_stage']
for t in range(6,10):
 current=js(C/'ra_coreset/storage'/('task%d.json'%t));base=js(B/'ra_coreset/storage'/('task%d.json'%t))
 assert current['K']==base['K']=={6:781,7:740,8:780,9:660}[t]
 assert current['selection']['selected_ids']==base['selection']['selected_ids']
 assert [(Path(x['source']).name,x['destination'],x['sha256']) for x in current['selected']]==[(Path(x['source']).name,x['destination'],x['sha256']) for x in base['selected']]
 assert js(C/'ra_coreset'/('ifa_task%d.json'%t))==js(B/'ra_coreset'/('ifa_task%d.json'%t))
 cp=next(x for x in a['checkpoints'] if x['task']==t);assert cp['strict_loaded_keys']==100
 if t>6:
  fit=js(C/'effects/models'/('task%d'%t)/'ACCEPTANCE.json')
  if fit['status']=='PASS':
   assert fit['heldout_mae']<min(fit['training_mean_baseline_mae'],fit['permuted_action_mae']) and fit['action_gradient_finite_nonzero'] and len(fit['heldout_old_demos'])>=3
   refs=js(C/'ra_coreset/functional_references'/('task%d.json'%t))['groups']
   for k,r in refs.items():
    best=min(r['candidates'],key=lambda x:(x['nll'],x['stage']))
    assert best==dict(stage=r['reference_stage'],nll=r['bound']) and int(k.split('|')[0])<=r['reference_stage']<t
    assert len({x['demo'] for x in r['members']})>=3 and len({(x['demo'],x['u']) for x in r['members']})>=12

def load(root,pattern):return [json.loads(line) for f in sorted((root/'ra_coreset/evaluation').glob(pattern)) for line in f.read_text().splitlines()]
trials=load(C,'trials*.jsonl');events=load(C,'attempts*.jsonl');cells={f.name:js(f) for f in (C/'ra_coreset/evaluation').glob('cell*.json')}
formal=audit_formal(dict(trials=trials,attempts=events,cells=cells),42)
assert formal['records']==200 and formal['started_vectors']==formal['completed_vectors']==20 and formal['completed_cells']==10
def matrix(rows):
 assert len(rows)==200;by={(r['stage'],r['task'],r['trial']):r for r in rows};assert len(by)==200
 rates={}
 for s in range(6,10):
  for t in range(6,s+1):
   values=[by[s,t,i] for i in range(20)];assert all(r['seed']==42 and type(r['success']) is bool for r in values)
   rates[s,t]=F(sum(r['success'] for r in values),20)
 return rates,by
def metrics(r):return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def pack(m):return {k:dict(fraction=str(v),percent=float(100*v)) for k,v in m.items()}
rates,new=matrix(trials);base,old=matrix(load(B,'trials*.jsonl'));v1,_=matrix(load(V,'trials*.jsonl'))
for key,r in new.items():
 assert r['initial_state_sha256']==old[key]['initial_state_sha256'] and r['modal_sha256']==old[key]['modal_sha256']
 assert r['checkpoint_sha256']==next(c['sha256'] for c in a['checkpoints'] if c['task']==r['stage'])
m,b,v=metrics(rates),metrics(base),metrics(v1)
assert all(F(a['metrics'][k]['fraction'])==x and F(a['baseline'][k]['fraction'])==b[k] for k,x in m.items())
out=dict(status='COMPLETED_LOCAL_INDEPENDENT_PASS',method='CAET-v2',suite='OBJECT',seed=42,metrics=pack(m),ra=pack(b),caet_v1=pack(v),delta_ra=pack({k:m[k]-b[k] for k in m}),delta_v1=pack({k:m[k]-v[k] for k in m}),matrix=[dict(stage=s,task=t,v2=str(rates[s,t]),ra=str(base[s,t]),v1=str(v1[s,t])) for s,t in rates],activation=activation,formal=formal,main_steps=steps,preflight_updates=8,main_training_seconds=sum(r['seconds'] for r in h),package_manifest_sha256=sha(p/'PACKAGE_MANIFEST.json'),automatic_seed_expansion=False)
(p/'LOCAL_FINAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
report='# CAET-v2 OBJECT seed42 final acceptance\n\nIndependent metadata and exact episode recomputation PASS. Separate algorithm from v1.\n\n|Metric|RA %|CAET-v1 %|CAET-v2 %|v2−RA pp|\n|---|---:|---:|---:|---:|\n'
for k in m:report+=f'|{k}|{float(100*b[k]):.6f}|{float(100*v[k]):.6f}|{float(100*m[k]):.6f}|{float(100*(m[k]-b[k])):+.6f}|\n'
report+='\nSingle-seed exploratory result; support thresholds selected after observing v1/GOAL. Native BC remains one-action-index earlier than causal auxiliary labels, so this experiment alone cannot isolate timing correction from the proposed mechanism. Global evaluation action RNG is not per-trajectory isolated. Historical-best retention refers to offline next-action NLL on retained samples, not guaranteed closed-loop success. CAET-v1 retains its post-action-input scientific limitation. FWT is the original trained-diagonal metric; lower NBT is favorable under the original definition. Extra effect fitting and functional-reference scoring cost must be reported. No other seeds are automatically authorized.\n'
(p/'REPORT.md').write_text(report,encoding='utf8');print(json.dumps(out,indent=2))
