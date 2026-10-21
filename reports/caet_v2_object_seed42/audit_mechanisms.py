import json,hashlib,sys,math
from pathlib import Path
p=Path(sys.argv[1]);receipt=json.loads((p/'MECHANISM_RECEIPT.json').read_text());result={}
for n,m in receipt['files'].items():
 raw=(p/n).read_bytes();assert hashlib.sha256(raw).hexdigest()==m['sha256'];d=json.loads(raw);a=d['admission'];assert a['status']=='PASS'
 assert a['heldout_mae']<a['training_mean_baseline_mae'] and a['heldout_mae']<a['permuted_action_mae']
 assert a['action_gradient_finite_nonzero'] and a['action_gradient_abs_sum']>0 and len(a['heldout_old_demos'])>=3 and a['heldout_supported']>0
 assert a['support_sha256']==d['support_sha256']
 totals=dict(updates=0,transfer_applied=0,transfer_blocked=0,retention_active=0,retention_nonzero_loss=0)
 previous=None;max_mu=0.;positive_mu_epochs=0
 for ep in d['epochs']:
  rows=ep['rows'];assert len(rows)==ep['batches'];assert [x['step'] for x in rows]==list(range(ep['steps']-ep['batches'],ep['steps']))
  c=ep['constraints'];assert c['active'] and c['epsilon']==.05
  assert previous is None and all(v==0 for v in c['mu_before'].values()) or previous==c['mu_before']
  for k,v in c['mu_after'].items():
   expected=max(0.,min(1.,c['mu_before'][k]+.1*(c['values'][k]-c['bounds'][k]-.05)));assert abs(v-expected)<1e-12
   max_mu=max(max_mu,v)
  positive_mu_epochs+=any(v>0 for v in c['mu_after'].values());previous=c['mu_after']
  for x in rows:
   assert x['active'];dot=x['transfer_dot'];assert x['transfer_applied']==(dot is not None and dot>=0)
   assert (dot is not None)==(x['transfer_supported']>0)
   assert all(math.isfinite(v) for k,v in x.items() if isinstance(v,float))
   assert x['retention_loss']>=0
   assert (x['retention_groups']>0)==(x['retention_loss']>0)
   totals['updates']+=1;totals['transfer_applied']+=x['transfer_applied'];totals['transfer_blocked']+=dot is not None and dot<0
   totals['retention_active']+=x['retention_groups']>0;totals['retention_nonzero_loss']+=x['retention_loss']>0
 result[n]=dict(epochs=len(d['epochs']),admission=a,counts=totals,positive_mu_epochs=positive_mu_epochs,max_mu=max_mu,dual_recomputation='PASS')
out=dict(status='MECHANISM_LEDGER_PASS',observed_unix=receipt['observed_unix'],tasks=result,not_closed_loop_success=True)
(p/'MECHANISM_LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2),encoding='utf8');print(json.dumps(out,indent=2))
