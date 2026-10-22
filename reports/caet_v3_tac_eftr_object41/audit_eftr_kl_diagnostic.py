"""Independent arithmetic audit and lightweight publication packaging."""
import sys,json,math,hashlib,statistics,datetime
from pathlib import Path
p=Path(sys.argv[1]);raw=(p/'EVIDENCE.json').read_bytes();d=json.loads(raw)
assert d['status']=='ZERO_TRAINING_DIAGNOSTIC_EVIDENCE'
assert d['strict_model_keys']==100 and d['actual_head']=='GMMHead' and not d['actual_low_eval_noise']
assert all(d[k]==0 for k in ('optimizer_updates','backwards','environment_steps','server_writes')) and not d['cuda_initialized']
r=d['rows'];scores=d['candidate_scores'];n=len(r)
assert n==781==d['teacher_meta']['count'] and len({x['sample'] for x in r})==n
assert d['teacher_replay']['mode_mismatches']==0 and d['teacher_replay']['means_max_abs']<1e-4
def close(a,b):assert math.isclose(a,b,rel_tol=2e-9,abs_tol=2e-8),(a,b)
def mean(v):return sum(v)/len(v)
def avg(k,rr=r):return mean([x[k] for x in rr])
for x in r:
 assert all(math.isfinite(z) for z in x['kl_by_dim'])
 for a,b,c in zip(x['kl_by_dim'],x['mean_term_by_dim'],x['variance_term_by_dim']):close(a,b+c)
 close(x['arm_contribution_R'],sum(x['kl_by_dim'][:6])/7)
 close(x['gripper_contribution_R'],x['kl_by_dim'][6]/7)
 assert x['switched']==(x['teacher_mode']!=x['current_mode'])
 close(x['explicit_switch_contribution_R'],.1*x['switched'])
 close(x['R'],x['arm_contribution_R']+x['gripper_contribution_R']+x['explicit_switch_contribution_R'])
 close(x['actual_minus_same_mode_R'],(sum(x['kl_by_dim'])-sum(x['same_mode_kl_by_dim']))/7)
 assert x['R']>=-1e-8
assert abs(avg('R')-d['constraint']['R'])/max(1,abs(d['constraint']['R']))<1e-6
counts={'selected':[0,0,0],'budget_unmet':0,'selected_R_better':0,'selected_R_equal':0,'selected_R_worse':0,'BC_rejected_candidates':0}
delta=[];relative=[];margins=[];old_scale=[];epochs=[]
for j,x in enumerate(scores):
 assert x['step']==15700+j and x['optimizer_trials']==3 and x['committed_updates']==1
 ss=x['scores'];c0=ss[0]['C'];valid=[i for i,s in enumerate(ss) if c0 is None or s['C']<=c0+.02]
 budget=[i for i in valid if ss[i]['R']<=.1];eligible=budget or valid
 k=min(eligible,key=lambda i:(ss[i]['R'],ss[i]['C'] or 0.,i));assert k==x['selected'] and x['budget_unmet']==(not bool(budget))
 counts['selected'][k]+=1;counts['budget_unmet']+=x['budget_unmet'];counts['BC_rejected_candidates']+=3-len(valid)
 diff=ss[0]['R']-ss[k]['R'];delta.append(diff)
 counts['selected_R_better' if diff>0 else 'selected_R_worse' if diff<0 else 'selected_R_equal']+=1
 if ss[0]['R']>0:relative.append(diff/ss[0]['R'])
 if c0 is not None:margins.append(ss[k]['C']-c0)
 old_scale.append(x['retention_loss'])
 if (j+1)%227==0:
  chunk=scores[j-226:j+1];epochs.append({'epoch':j//227,'budget_unmet':sum(z['budget_unmet'] for z in chunk),'mean_native_R':mean([z['scores'][0]['R'] for z in chunk]),'mean_selected_R':mean([z['scores'][z['selected']]['R'] for z in chunk]),'mean_retention_loss_same_mode_before_update':mean([z['retention_loss'] for z in chunk])})
assert len(scores)==227*(d['snapshot']['epoch']+1)
assert counts['selected_R_worse']==0 and max(margins)<=.02
groups={}
for name,rr in [('switched',[x for x in r if x['switched']]),('not_switched',[x for x in r if not x['switched']]),('gripper_sign_changed',[x for x in r if x['gripper_mean_sign_changed']]),('gripper_sign_unchanged',[x for x in r if not x['gripper_mean_sign_changed']])]:
 groups[name]={'count':len(rr),'mean_R':avg('R',rr),'mean_arm_R':avg('arm_contribution_R',rr),'mean_gripper_R':avg('gripper_contribution_R',rr),'total_R_share':sum(x['R'] for x in rr)/sum(x['R'] for x in r)}
native=mean([x['scores'][0]['R'] for x in scores]);sel=mean([x['scores'][x['selected']]['R'] for x in scores])
result={'status':'PASS_ZERO_TRAINING_DIAGNOSTIC','snapshot':d['snapshot'],'evidence_sha256':hashlib.sha256(raw).hexdigest(),'rows':n,'teacher_replay':d['teacher_replay'],'actual_head':'GMMHead','config_head_is_not_constructor':True,
 'R_decomposition':{k:avg(k) for k in ['R','arm_contribution_R','gripper_contribution_R','explicit_switch_contribution_R']},
 'gripper_share_of_KL':sum(x['kl_by_dim'][6] for x in r)/sum(sum(x['kl_by_dim']) for x in r),
 'dim_KL_means':[mean([x['kl_by_dim'][j] for x in r]) for j in range(7)],
 'arm_mean_shift_R':mean([sum(x['mean_term_by_dim'][:6])/7 for x in r]),'arm_variance_R':mean([sum(x['variance_term_by_dim'][:6])/7 for x in r]),
 'same_teacher_mode_KL_per7':mean([sum(x['same_mode_kl_by_dim'])/7 for x in r]),'switch_induced_signed_KL_per7_difference':avg('actual_minus_same_mode_R'),
 'groups':groups,'sign_changes_with_mode_switch':sum(x['switched'] and x['gripper_mean_sign_changed'] for x in r),
 'arithmetic_only_ablations':{'without_gripper_R':avg('arm_only_R'),'without_gripper_rows_le_budget':sum(x['arm_only_R']<=.1 for x in r),'all_rows_le_budget':sum(x['R']<=.1 for x in r),'gripper_std_floors_R':{k:mean([x['arithmetic_floor_R'][k] for x in r]) for k in ['0.001','0.01','0.05']}},
 'candidate_audit':dict(counts,commits=len(scores),optimizer_calls_logged=3*len(scores),unmet_fraction=counts['budget_unmet']/len(scores),mean_native_R=native,mean_selected_R=sel,relative_mean_R_reduction=(native-sel)/native,max_current_NLL_delta=max(margins),median_R_reduction=statistics.median(delta),median_relative_R_reduction=statistics.median(relative),epoch_summaries=epochs),
 'resources':{k:d[k] for k in ['model_forward_calls','optimizer_updates','backwards','environment_steps','server_writes','cuda_initialized','cpu_threads','seconds']},
 'limits':['One committed Task7 epoch snapshot; per-dimension values for historical three candidates were not saved and cannot be reconstructed without optimizer replay.','Candidate total-R comparisons use original logs; no counterfactual training or causal closed-loop benefit claim.','Gripper mean sign and Gaussian sign probability diagnostics do not replace native controller rollout.','Arithmetic variance floors and dimension removal are diagnostic only; no change to frozen experiment.','Same-mode comparison is a counterfactual metric, not a valid replacement for actual execution distribution.']}
(p/'LOCAL_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
# Preserve a >8MiB scalar source locally; publish losslessly reconstructible JSON parts.
keys=list(d);header={k:v for k,v in d.items() if k!='candidate_scores'};parts={}
def part(name,obj):
 raw2=(json.dumps(obj,ensure_ascii=True)+'\n').encode();assert len(raw2)<8*1024**2
 (p/name).write_bytes(raw2);parts[name]={'sha256':hashlib.sha256(raw2).hexdigest(),'bytes':len(raw2)}
part('HEADER_AND_DIVERGENCES.json',header)
for start in range(0,len(scores),1000):part('CANDIDATE_SCORES_%05d.json'%start,scores[start:start+1000])
recovered=[]
for name in sorted(parts):
 if name.startswith('CANDIDATE_SCORES_'):recovered.extend(json.loads((p/name).read_text()))
hr=json.loads((p/'HEADER_AND_DIVERGENCES.json').read_text());reconstructed={k:recovered if k=='candidate_scores' else hr[k] for k in keys}
rebuilt=(json.dumps(reconstructed)+'\n').encode();assert rebuilt==raw
(p/'LIGHT_PACKAGE_AUDIT.json').write_text(json.dumps({'status':'PASS','original_sha256':hashlib.sha256(raw).hexdigest(),'original_bytes':len(raw),'key_order':keys,'parts':parts,'reconstructed_raw_bytes_exact':True},indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='candidate_audit'},indent=2,ensure_ascii=False))
print(json.dumps({k:v for k,v in result['candidate_audit'].items() if k!='epoch_summaries'},indent=2))
