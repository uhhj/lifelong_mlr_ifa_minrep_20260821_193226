"""Independent local identity, arithmetic, scalar aggregation and CPU/GPU parity audit."""
import sys,json,math,hashlib,datetime
from pathlib import Path
R=Path(__file__).resolve().parent
O=Path(sys.argv[1]);O=O if O.is_absolute() else R/O
S=R/'snapshot_20261024_132735';B=S/'metadata/B0'
def js(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(v):return math.fsum(v)/len(v)
def quantile(v,q):
 v=sorted(v);i=(len(v)-1)*q;lo=math.floor(i);hi=math.ceil(i);return v[lo]+(v[hi]-v[lo])*(i-lo)
def stats(v):return dict(mean=mean(v),p10=quantile(v,.1),median=quantile(v,.5),p90=quantile(v,.9),p99=quantile(v,.99),minimum=min(v),maximum=max(v))
ev=js(O/'EVIDENCE.json');assert ev['status']=='PASS_CPU_READONLY_DIAGNOSTIC_EXTRA_FORWARDS_NOT_NEW_CLOSED_LOOP'
assert ev['optimizer_steps']==ev['environment_steps']==ev['server_artifact_writes']==0 and not ev['cuda_initialized']
assert ev['script_sha256']==sha(R/'B0_action_diagnostics_readonly.py')
assert ev['contract']=='2f6af444986a5710cbc84572f25ec15f9b3d44ea72aacbd01fa3ad4e624681b0'
allrows={};identities={};stages={};parity=[];sources={'EVIDENCE.json':sha(O/'EVIDENCE.json')}
for st in range(6,10):
 x=js(O/f'STAGE{st}.json');stages[st]=x;sources[f'STAGE{st}.json']=sha(O/f'STAGE{st}.json')
 manifest=js(B/'dataset'/f'task{st}.json')['rows'];identities[st]=manifest
 assert x['dataset_sha256']==sha(B/'dataset'/f'task{st}.json')
 assert x['old_diagnostic_sha256']==sha(B/'diagnostics'/f'task{st}.json')
 assert x['checkpoint_sha256']==ev['checkpoints'][str(st)]==js(B/'TRAINING_COMPLETE.json')['checkpoints'][str(st)]
 assert x['strict_keys']==104 and x['model_tensors_unchanged'] and x['input_sha_before_after']
 assert len(manifest)==x['windows'] and x['forward_calls']==math.ceil(x['windows']/32)
 assert len(x['parity'])==x['windows']
for name,info in ev['case_files'].items():
 p=O/(name+'.json');assert sha(p)==info['sha256'] and p.stat().st_size==info['bytes'];sources[p.name]=sha(p)
 rows=js(p);assert len(rows)==info['rows']
 st=int(name.split('_')[0][5:]);task=int(name.split('_')[1][4:]);h=int(name.split('_')[2][1:])
 expected={i for i,r in enumerate(identities[st]) if r['task']==task and r['action_mask'][h-1]}
 assert len(rows)==len(expected) and {r['index'] for r in rows}==expected
 for r in rows:
  m=identities[st][r['index']]
  assert r['sample_sha256']==m['sha256'] and r['task']==task and r['horizon']==h and r['current']==(task==st)
  assert all(math.isfinite(v) for v in r.values() if isinstance(v,(int,float)))
  assert r['oracle_huber']<=r['huber']+1e-7 and r['sigma_min']>0
  for k in ('pmax','weight_margin','entropy','selected_posterior','box_tail'):
   assert -1e-6<=r[k]<=1+1e-6,(name,k,r[k])
  for g in ('translation','rotation','gripper'):
   assert r[g+'_mae']**2<=r[g+'_mse']+1e-6
   assert 0<=r[g+'_coverage1']<=r[g+'_coverage2']<=1
   assert r['sigma_min']-1e-6<=r[g+'_sigma_rms']<=r['sigma_max']+1e-6
 allrows[st,task,h]=rows
assert len(allrows)==40
for st,x in stages.items():
 rows=[r for (s,t,h),rs in allrows.items() if s==st for r in rs]
 assert len(rows)==x['valid_positions']==sum(sum(r['action_mask']) for r in identities[st])
 by_index={i:[] for i in range(x['windows'])}
 for r in rows:by_index[r['index']].append(r)
 for p in x['parity']:
  assert p['count']==len(by_index[p['index']])
  assert abs(math.fsum(r['huber'] for r in by_index[p['index']])-p['new_sum'])<=1e-5*max(1,abs(p['new_sum']))
 for task in range(6,st+1):
  pr=[r for r in x['parity'] if r['task']==task];count=sum(r['count'] for r in pr)
  a=math.fsum(r['old_sum'] for r in pr)/count;b=math.fsum(r['new_sum'] for r in pr)/count
  parity.append(dict(stage=st,task=task,old_gpu_mean=a,new_cpu_mean=b,absolute_difference=abs(a-b),tolerance=1e-4+.01*abs(a),within_tolerance=abs(a-b)<=1e-4+.01*abs(a),max_sample_sum_difference=max(abs(r['old_sum']-r['new_sum']) for r in pr)))
assert all(r['within_tolerance'] for r in parity),parity
def summarize(rows):
 numeric=[k for k,v in rows[0].items() if isinstance(v,float)]
 out=dict(count=len(rows),statistics={k:stats([r[k] for r in rows]) for k in numeric})
 out['selected_not_oracle_mean_fraction']=mean([int(r['selected_mode']!=r['oracle_mode']) for r in rows])
 out['selected_not_target_posterior_fraction']=mean([int(r['selected_mode']!=r['posterior_mode']) for r in rows])
 out['oracle_huber_gap']=stats([max(0,r['huber']-r['oracle_huber']) for r in rows])
 out['selected_mode_counts']={str(i):sum(r['selected_mode']==i for r in rows) for i in range(5)}
 total=math.fsum(r['huber'] for r in rows)
 grip_huber=math.fsum((.5*r['gripper_mae']**2 if r['gripper_mae']<1 else r['gripper_mae']-.5)/3 for r in rows)
 bad=[r for r in rows if r['gripper_mae']>1]
 assert grip_huber<=total+1e-5
 out['error_concentration']=dict(gripper_huber_share=grip_huber/total if total else 0,largest_1percent_huber_share=math.fsum(r['huber'] for r in sorted(rows,key=lambda x:x['huber'],reverse=True)[:math.ceil(.01*len(rows))])/total if total else 0,gripper_abs_error_gt1_count=len(bad),gripper_abs_error_gt1_fraction=len(bad)/len(rows),large_gripper_selected_pmax=stats([r['pmax'] for r in bad]) if bad else None,large_gripper_weight_margin=stats([r['weight_margin'] for r in bad]) if bad else None,large_gripper_oracle_huber=stats([r['oracle_huber'] for r in bad]) if bad else None)
 for g in ('translation','rotation','gripper'):
  mse=mean([r[g+'_mse'] for r in rows]);var=mean([r[g+'_sigma_rms']**2 for r in rows])
  out[g+'_variance_analysis']=dict(mean_error_mse=mse,mean_predicted_variance=var,expected_sample_mse=mse+var,variance_share_of_expected_sample_mse=var/(mse+var),rmse=math.sqrt(mse),sigma_rms=math.sqrt(var))
 return out
summary={}
paired={}
for st in range(6,10):
 for task in range(6,st+1):
  for horizons,label in [([1],'first'),([2,3,4],'future3'),([1,2,3,4],'all4')]+[([h],f'h{h}') for h in range(1,5)]:
   rows=[r for h in horizons for r in allrows[st,task,h]]
   summary[f'stage{st}_task{task}_{label}']=summarize(rows)
  by_h={h:{r['index']:r for r in allrows[st,task,h]} for h in range(1,5)}
  common=set.intersection(*(set(v) for v in by_h.values()))
  differences=[by_h[1][i]['huber']-mean([by_h[h][i]['huber'] for h in (2,3,4)]) for i in sorted(common)]
  paired[f'stage{st}_task{task}']=dict(common_four_valid_windows=len(common),first_mean_huber=mean([by_h[1][i]['huber'] for i in common]),future3_mean_huber=mean([mean([by_h[h][i]['huber'] for h in (2,3,4)]) for i in common]),first_minus_future3=stats(differences),first_worse_fraction=mean([int(d>0) for d in differences]),interpretation='Paired training/RA windows, descriptive only; overlapping windows are not independent trials.')
cost=dict(windows=sum(x['windows'] for x in stages.values()),forward_calls=sum(x['forward_calls'] for x in stages.values()),valid_positions=sum(x['valid_positions'] for x in stages.values()),cpu_seconds=ev['cpu_seconds'],wall_seconds=ev['wall_seconds'],optimizer_steps=0,environment_steps=0,cuda_initialized=False)
assert cost['windows']==ev['policy_windows']==10524 and cost['valid_positions']==39958
audit=dict(status='PASS_LOCAL_B0_ACTION_SCALAR_AUDIT_NOT_CLOSED_LOOP_ACCEPTANCE',time=datetime.datetime.now().astimezone().isoformat(),source_sha256=sources,full_case_count=40,cost=cost,CPU_GPU_parity=parity,checkpoint_binding=True,identity_mask_complete_unique=True,arithmetic_verified=True,limitations=['Training/current demos and old RA only; not independent held-out generalization.', 'CPU mean predictions reproduce old aggregate within registered tolerance; new CPU scalars are not new rollout evidence.', 'Oracle component is chosen with target hindsight, not a deployable decision rule.', 'Selected-component coverage combines mode choice and variance; cannot uniquely diagnose variance collapse.', 'Expected Normal sample MSE is analytic, no action draws or new environments.'])
(O/'LOCAL_AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
(O/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
(O/'PAIRED_HORIZONS.json').write_text(json.dumps(paired,indent=2)+'\n')
print(json.dumps(dict(status=audit['status'],cost=cost,parity_max=max(x['absolute_difference'] for x in parity)),indent=2))
for st in range(6,10):
 for task in range(6,st+1):
  row={}
  for part in ('first','future3'):
   z=summary[f'stage{st}_task{task}_{part}'];v=z['statistics']
   row[part]=dict(huber=v['huber']['mean'],oracle=v['oracle_huber']['mean'],pmax=v['pmax']['mean'],posterior=v['selected_posterior']['mean'],wrong_mean=z['selected_not_oracle_mean_fraction'],wrong_posterior=z['selected_not_target_posterior_fraction'],sigma_grip=v['gripper_sigma_rms']['median'],coverage2_grip=v['gripper_coverage2']['mean'],grip_mse=z['gripper_variance_analysis'],translation=z['translation_variance_analysis'])
  print(json.dumps(dict(stage=st,task=task,**row)))
