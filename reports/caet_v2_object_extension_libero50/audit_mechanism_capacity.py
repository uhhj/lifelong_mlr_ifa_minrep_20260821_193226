"""Independent numeric/ledger audit of read-only evidence; no GPU or rollout."""
import json,sys,math,hashlib
from pathlib import Path
P=Path(sys.argv[1]);M=Path(sys.argv[2]);S=json.loads((P/'SNAPSHOT.json').read_text(encoding='utf8'));D=json.loads((M/'SNAPSHOT.json').read_text(encoding='utf8'))
assert all(D[k]==0 for k in ('server_writes','model_forwards','optimizer_updates','environment_calls'))
assert D['controller_contract_checks'] and all(D['controller_contract_checks'].values())
summary={};timing=[]
for t,epochs in D['constraints'].items():
 previous=None;max_mu=0;violations=0;max_excess=-math.inf
 for i,(name,row) in enumerate(sorted(epochs.items())):
  assert name=='task%s_epoch%02d.json'%(t,i) and row['active'] and row['epsilon']==.05
  assert set(row['values'])==set(row['bounds'])==set(row['mu_before'])==set(row['mu_after'])
  if previous is not None:assert row['mu_before']==previous
  else:assert all(v==0 for v in row['mu_before'].values())
  for k,v in row['values'].items():
   assert math.isfinite(v)
   excess=v-row['bounds'][k]-.05
   assert row['mu_after'][k]==max(0.,min(1.,row['mu_before'][k]+.1*excess))
   max_excess=max(max_excess,excess);violations+=excess>0;max_mu=max(max_mu,row['mu_after'][k])
  previous=row['mu_after']
 fit=next(v for k,v in S['files'].items() if k.endswith('/effects/models/task%s/ACCEPTANCE.json'%t))
 assert fit['status']=='PASS' and fit['heldout_mae']<min(fit['training_mean_baseline_mae'],fit['permuted_action_mae']) and fit['action_gradient_finite_nonzero'] and len(fit['heldout_old_demos'])>=3
 h=D['effect_epochs'][t];assert len(h)==10 and sum(x['updates'] for x in h)==fit['effect_updates']
 if fit['seconds']<0:timing.append(dict(task=int(t),raw_seconds=fit['seconds'],status='INVALID_DURATION_PRESERVED',reason='Frozen fit.py uses wall-clock time.time; arithmetic/eligibility does not use this duration. No retraining or replacement value.'))
 summary[t]=dict(predictor='PASS',effect_updates=fit['effect_updates'],constraint_epochs=len(epochs),protected_groups=len(previous),max_mu=max_mu,violated_group_epochs=violations,max_observed_excess=max_excess,full_epoch_dual_updates_exact=True,mean_control_mae=fit['training_mean_baseline_mae'],permuted_control_mae=fit['permuted_action_mae'],heldout_mae=fit['heldout_mae'])
x=D['preflight_lossless_compression_probe'];assert x['sha256']==x['restored_sha256'] and x['bytes']==x['restored_bytes'] and x['writes']==0
out=dict(status='READ_ONLY_MECHANISM_AND_CAPACITY_AUDIT_PASS_WITH_TIMING_CAVEAT',tasks=summary,effect_updates=sum(x['effect_updates'] for x in summary.values()),timing_caveats=timing,orchestration_hashes=len(D['controller_contract_checks']),compression_plain_zlib_saving_bytes=x['bytes']-x['compressed_bytes'],not_closed_loop_result=True,no_server_changes=True)
(M/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
