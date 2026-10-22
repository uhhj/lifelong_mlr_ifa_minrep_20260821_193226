"""Independent local scalar/identity/calibration recomputation; no model or environment."""
import collections,hashlib,json,math,sys
from pathlib import Path
R=Path(__file__).resolve().parent
def quant(rows,key,q):
 data=[(x['metrics'][key],x['demo']) for x in rows if x['metrics'] is not None];counts=collections.Counter(d for v,d in data);acc=0
 for v,d in sorted(data):
  acc+=1/len(counts)/counts[d]
  if acc>=q-1e-12:return v
 return sorted(data)[-1][0]
def summary(rows):
 return dict(n=len(rows),local_accepted=sum(x['local']['accepted'] for x in rows),weight_sum=sum(x['local']['weight'] for x in rows),cross_local=sum(x['cross'] and x['local']['accepted'] for x in rows),cross_weight_sum=sum(x['local']['weight'] for x in rows if x['cross']),cross=sum(x['cross'] for x in rows),old_pass={b:{k:sum(x['old'][b]['passed'][k] for x in rows) for k in rows[0]['old'][b]['passed']} for b in rows[0]['old']} if rows else {})
def same(a,b):
 if isinstance(a,dict):return set(a)==set(b) and all(same(a[k],b[k]) for k in a)
 if isinstance(a,float):return math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
 return a==b
def main():
 p=Path(sys.argv[1]);raw=(p/'EVIDENCE.json').read_bytes();d=json.loads(raw);pre=json.loads((p/'PRECOMMIT.json').read_text());receipt=json.loads((p/'RECEIPT.json').read_text());assert receipt['sha256']==hashlib.sha256(raw).hexdigest()
 assert d['protocol_sha256']==pre['protocol_sha256'] and d['input_evidence_sha256']==pre['input_evidence_sha256']
 assert d['optimizer_calls']==d['environment_calls']==d['server_writes']==0 and not d['cuda_initialized']
 semantic=bool(d.get('semantic_rule'));metric_keys=('d','state','arm') if semantic else ('d','state','arm','gripper')
 assert d['predictor_gate']==('FROZEN_V2_STRATIFIED_AUDIT_ONLY_NEW_RSET_UNVALIDATED' if semantic else 'NOT_TESTED')
 source=(R/'gate1_RSET/EVIDENCE.json').read_bytes();assert hashlib.sha256(source).hexdigest()==d['input_evidence_sha256'];g=json.loads(source)
 for key,c in d['calibrations'].items():
  task,exc=key.split('/');assert c['excluded_demo'] not in c['training_demos'];assert g['tasks'][task]['holdout_demo'] not in c['training_demos']
  assert c['calibration_count']==len(c['calibration']);assert c['calibration_estimable']==sum(x['metrics'] is not None for x in c['calibration'])
  expected={k:quant(c['calibration'],k,.95 if k=='d' else .99) for k in metric_keys};expected['tau']=max(quant(c['calibration'],'d',.5),1e-6);assert c['bounds']==expected
 def check(rows,stage):
  seen=set()
  for x in rows:
   identity=x['identity'];key=(identity['task'],identity['demo'],identity['u']);assert key not in seen;seen.add(key);assert identity['task']==int(stage)
   cal=d['calibrations'][stage+'/'+str(x['local']['calibration_excluded_demo'])];assert identity['demo'] not in cal['training_demos'];m=x['local']['metrics'];b=cal['bounds']
   why=['INSUFFICIENT_DISTINCT_DEMOS'] if m is None else [k for k in metric_keys if m[k]>b[k]];assert x['local']['reasons']==why and x['local']['accepted']==(not why)
   if semantic and m is not None:assert m['gripper_class_matched'] and m['demos']>=2
   weight=math.exp(-m['d']**2/b['tau']**2) if not why else 0.;assert math.isclose(x['local']['weight'],weight,rel_tol=1e-12,abs_tol=1e-15)
   for old in x.get('old',{}).values():
    assert old['passed']=={k:v>=3 for k,v in old['demos'].items()};assert old['demos']['joint_top20']<=old['demos']['joint_full'];assert old['demos']['joint_full']<=min(old['demos']['state_full'],old['demos']['action_full'])
 for stage,s in d['stages'].items():
  check(s['expert'],stage);supported={g['stages'][stage]['current_nodes'][i]['sample'] for i,j in g['stages'][stage]['edges']};original={x['sample']:x for x in g['stages'][stage]['rset_zero_training']['queries']}
  for x in s['expert']:
   assert x['cross']==(x['identity']['sample'] in supported)
   if not x['holdout']:assert x['old']['all_task']['passed']['joint_top20']==original[x['identity']['sample']]['expert_action_supported']
  assert same(s['summary'],{name:summary([x for x in s['expert'] if cond(x)]) for name,cond in [('all_train',lambda x:not x['holdout']),('effect_supported_train',lambda x:x['cross']),('effect_holdout',lambda x:x['holdout'])]})
 for model in d['models']:
  assert model['strict_keys']==100 and model['unchanged'] and model['checkpoint_task'] in (model['stage']-1,model['stage']);assert len(model['sha256'])==64
  for a in model['actions'].values():check(a['rows'],str(model['stage']));assert same(a['summary'],summary(a['rows']))
 if semantic:
  src=(R/'gripper_semantics_20261022_125239/SOURCE_EVIDENCE.json').read_bytes();assert hashlib.sha256(src).hexdigest()==d['semantic_source_sha256']==pre['semantic_source_sha256'];sd=json.loads(src)
  assert len(sd['pure_native_formatter_tests'])==28 and all(x['exact'] for x in sd['pure_native_formatter_tests'])
  for stage,v in d['ood'].items():
   for stress in v['stress'].values():check(stress['rows'],stage);assert stress['accepted']==sum(x['local']['accepted'] for x in stress['rows']) and stress['n']==len(stress['rows'])
   assert len(v['gripper_same_class'])==3 and all(x['support_and_weight_exact'] for x in v['gripper_same_class'].values())
  assert len(d['predictors'])==2
  for predictor in d['predictors']:
   assert predictor['unchanged'] and predictor['all_parameters_frozen'] and predictor['optimizer_calls']==0
   assert abs(predictor['global_recomputed']['mae']-predictor['original_admission']['heldout_mae'])<1e-5
   for row in predictor['phases'].values():
    reasons=[];m=row['metrics']
    if row['windows']<12:reasons.append('FEWER_THAN_12_WINDOWS')
    if len(row['task_demos'])<2:reasons.append('FEWER_THAN_2_TASK_DEMOS')
    if row['changed_arm_permutations']<12:reasons.append('INSUFFICIENT_ARM_PERMUTATION_CHANGE')
    for k in ('global_mean_mae','phase_mean_mae','arm_permutation_mae'):
     if m['expert_mae']>=m[k]:reasons.append('NOT_BETTER_THAN_'+k)
    if row['arm_gradient_abs_sum']<=0:reasons.append('ZERO_ARM_GRADIENT')
    if any(x['mae']>=m['phase_mean_mae'] for x in row['semantic_variants'].values()):reasons.append('SEMANTIC_GRIPPER_VARIANTS_NOT_BETTER_THAN_PHASE_MEAN')
    assert row['reasons']==reasons and row['accepted']==(not reasons)
 status='SCALAR_IDENTITY_CALIBRATION_AUDIT_PASS' if d['policy_status']=='PASS' else 'PARTIAL_POLICY_FORWARD_INCOMPLETE'
 out=dict(status=status,evidence_sha256=receipt['sha256'],policy_status=d['policy_status'],calibrations=len(d['calibrations']),models=len(d['models']),zero_optimizer_environment=True,production_change=False,predictor_reliability_validated=False)
 (p/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
