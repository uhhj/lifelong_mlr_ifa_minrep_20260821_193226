import json,math
from coreset_exp.common import O,sha
from .variant import VARIANT

def js(p):return json.loads(p.read_text())

def audit():
 if VARIANT=='EFTR':
  from .audit_eftr import audit as other
  return other()
 result={}
 for t in range(6,10):
  root=O/'effects/current'/('task%d'%t);proof=js(root/'AUDIT.json');g=js(root/'SUPPORT.json')
  assert proof['status']=='PASS' and sha(root/'labels.pth')==proof['labels_sha256']
  if t>6:
   expected=js(O/'GATE1_EVIDENCE.json')['stages'][str(t)]
   for k in ('edges','current_nodes','old_nodes','groups'):assert g[k]==expected[k]
  fit=None if t==6 else js(O/'effects/models'/('task%d'%t)/'ACCEPTANCE.json')
  active=t>6 and fit['status']=='PASS'
  if active:
   assert fit['heldout_mae']<min(fit['training_mean_baseline_mae'],fit['permuted_action_mae'])
   assert fit['action_gradient_finite_nonzero'] and len(fit['heldout_old_demos'])>=3
   assert sha(O/'effects/models'/('task%d'%t)/'final.pth')==fit['frozen_predictor_sha256']
   refs=js(O/'ra_coreset/functional_references'/('task%d.json'%t))['groups']
   for k,r in refs.items():
    assert r['reference_stage']<t and r['reference_stage']>=int(k.split('|')[0])
    best=min(r['candidates'],key=lambda x:(x['nll'],x['stage']));assert best['nll']==r['bound'] and best['stage']==r['reference_stage']
    assert len({x['demo'] for x in r['members']})>=3 and len({(x['demo'],x['u']) for x in r['members']})>=12
  transferred=blocked=protected=0
  for ep in range(100):
   row=js(O/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,ep)));assert row['active']==active
   if active:
    assert row['bounds']=={k:r['bound'] for k,r in refs.items()}
    for k,v in row['values'].items():
     assert math.isfinite(v);assert row['mu_after'][k]==max(0.,min(1.,row['mu_before'][k]+.1*(v-row['bounds'][k]-.05)))
   for x in js(O/'ra_coreset/auxiliary'/('task%d_epoch%02d.json'%(t,ep))):
    assert x['active']==active
    if not active:assert x['effect_loss'] is None and not x['transfer_applied'] and x['retention_groups']==0
    if x['transfer_dot'] is not None:
     assert x['transfer_supported']>0 and math.isfinite(x['transfer_dot'])
     assert x['transfer_applied']==(x['transfer_dot']>=0 and (VARIANT!='RSET' or x['selective_gate']['joint']>0))
    transferred+=int(x['transfer_applied']);blocked+=int(x['transfer_dot'] is not None and x['transfer_dot']<0);protected+=int(x['retention_groups']>0)
  result[str(t)]=dict(active=active,predictor=fit,transfer_applied_updates=transferred,transfer_blocked_updates=blocked,retention_active_updates=protected,causal_labels=proof['valid_next_labels'],support_sha256=sha(root/'SUPPORT.json'))
 return result
