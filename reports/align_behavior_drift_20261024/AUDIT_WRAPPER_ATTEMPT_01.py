"""Independent local recomputation from allowed per-window scalar records."""
import hashlib,json,sys,math
from pathlib import Path
from fractions import Fraction
import numpy as np
R=Path(__file__).resolve().parent;OUT=Path(sys.argv[1]).resolve();sources={}
def read(p):
 raw=p.read_bytes();sources[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
def describe(v):
 v=np.asarray(v,dtype=float);assert np.isfinite(v).all()
 if not len(v):return {'count':0}
 return {'count':len(v),'mean':float(v.mean()),'median':float(np.median(v)),'p90':float(np.quantile(v,.9)),'max':float(v.max()),'min':float(v.min())}
def group(rows):
 valid=[r for r in rows if r['bc_valid']]
 return {'windows':len(rows),'valid_expert':len(valid),'mode_switch_count':sum(r['mode_switched'] for r in rows),'sign_flip_count':sum(r['gripper_strict_sign_flip'] for r in rows),
 'selected_w2_squared':describe([r['selected_w2_squared'] for r in rows]),
 'expert_huber_delta':describe([r['expert']['grouped_huber']['delta'] for r in valid]),
 'expert_huber_worsened':sum(r['expert']['grouped_huber']['delta']>0 for r in valid)}
spec=read(R/'INPUT_SPEC.json');frozen=read(R/'FROZEN_DIAGNOSTIC_SPEC.json')
for name,h in frozen['files'].items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
compat=read(R/'WRAPPER_COMPATIBILITY_V2.json');attempt=read(OUT/'ATTEMPT.json')
assert attempt['exit']==0 and attempt['script_sha256']==compat['new_script_sha256']==hashlib.sha256((R/'drift_readonly_v2.py').read_bytes()).hexdigest()
complete=read(OUT/'complete.json');assert complete['status']=='PASS_CPU_READONLY_NOT_CAUSAL_CLOSED_LOOP_TEST'
assert complete['windows']==1356 and complete['policy_windows']==2712 and complete['forward_calls']==174
assert not any(complete[k] for k in ('optimizer_updates','backwards','environment_steps','server_artifact_writes','cuda_initialized'))
summaries={}
for e in spec['seeds']:
 seed=e['seed'];d=read(OUT/f'seed_{seed}.json');p=Path(e['local_package']);old=read(p/'LOCAL_FINAL_AUDIT.json')
 assert old['status']=='INDEPENDENT_FULL_MATRIX_LEDGER_PASS'
 assert old['contract_sha256']==d['contract_sha256']==e['contract_sha256'] and d['unchanged_after']
 assert d['forward_calls']==58 and d['windows']==452
 assert all(d['checkpoints'][t]['sha256']==h and d['checkpoints'][t]['strict_keys']==100 for t,h in e['checkpoints'].items())
 storage=read(p/'current/ra_coreset/storage/task8.json')['selected'];man=read(p/'current/ra_coreset/dataset_manifests/task9.json')['rows']
 man={Path(x['path']).name:x for x in man if x['task']==8};rows=d['rows']
 assert len(rows)==452 and [r['sample'] for r in rows]==[r['destination'] for r in storage]
 for r,s in zip(rows,storage):
  m=man[r['sample']];assert r['sample_sha256']==s['sha256']==m['sha256'] and r['bc_valid']==m['bc_valid']
  assert (r['demo'],r['u'])==(m['demo'],m['u'])
  assert r['mode_switched']==(r['teacher_mode']!=r['student_mode'])
  for key in ('teacher_mode','student_mode','nearest_student_component'):assert 0<=r[key]<5
  mu=r['mu_delta_squared_by_dim'];sd=r['sigma_delta_squared_by_dim'];same=r['same_teacher_index_w2_squared_by_dim']
  assert len(mu)==len(sd)==len(same)==7 and min(mu+sd+same)>=0
  assert math.isclose(sum(mu)+sum(sd),r['selected_w2_squared'],abs_tol=1e-10)
  assert math.isclose(sum(same),r['same_teacher_index_w2_squared'],abs_tol=1e-10)
  assert r['nearest_student_component_w2_squared']<=min(r['selected_w2_squared'],r['same_teacher_index_w2_squared'])+1e-10
  if not r['mode_switched']:assert math.isclose(r['selected_w2_squared'],r['same_teacher_index_w2_squared'],abs_tol=1e-10)
  for tag,sl in [('translation',slice(0,3)),('rotation',slice(3,6)),('gripper',slice(6,7))]:
   assert math.isclose(sum(mu[sl])+sum(sd[sl]),r[tag+'_w2_squared'],abs_tol=1e-10)
   assert math.isclose(sum(mu[sl]),r[tag+'_mean_l2_delta']**2,abs_tol=1e-10)
  if r['bc_valid']:
   assert len(r['next_action_sha256'])==64
   for v in r['expert'].values():assert all(math.isfinite(z) for z in v.values()) and math.isclose(v['student']-v['teacher'],v['delta'],abs_tol=1e-8,rel_tol=1e-12)
  else:assert r['expert']=={}
 valid=[r for r in rows if r['bc_valid']];assert len(valid)==d['valid_expert_windows']
 numeric={k:describe([r[k] for r in rows]) for k in ('selected_w2_squared','same_teacher_index_w2_squared','nearest_student_component_w2_squared','translation_mean_l2_delta','rotation_mean_l2_delta','gripper_mean_l2_delta','translation_w2_squared','rotation_w2_squared','gripper_w2_squared','teacher_gripper_sigma','student_gripper_sigma')}
 expert={k:{field:describe([r['expert'][k][field] for r in valid]) for field in ('teacher','student','delta')} for k in ('mixture_nll','selected_nll','grouped_huber')}
 margins=np.asarray([r['teacher_probability_margin'] for r in rows]);cuts=np.quantile(margins,[.25,.5,.75]);bins=np.searchsorted(cuts,margins,side='right')
 quartiles={str(i+1):group([r for j,r in enumerate(rows) if bins[j]==i]) for i in range(4)}
 bydemo={demo:group([r for r in rows if r['demo']==demo]) for demo in sorted({r['demo'] for r in rows})}
 matrix={(x['stage'],x['task']):float(Fraction(x['current'])*100) for x in old['matrix']}
 summaries[str(seed)]={'all':group(rows),'valid_only':group(valid),'numeric':numeric,'expert':expert,
 'by_switch':{str(s):group([r for r in rows if r['mode_switched']==s]) for s in (False,True)},
 'by_gripper_flip':{str(s):group([r for r in rows if r['gripper_strict_sign_flip']==s]) for s in (False,True)},
 'teacher_margin_quartile_cuts':cuts.tolist(),'teacher_margin_quartiles':quartiles,'demo_groups':bydemo,
 'closed_loop_existing_percent':{'stage8_task8':matrix[8,8],'stage9_task8':matrix[9,8],'stage9_task9':matrix[9,9]},
 'variance_fraction_of_total_w2':sum(sum(r['sigma_delta_squared_by_dim']) for r in rows)/sum(r['selected_w2_squared'] for r in rows),
 'mean_index_vs_actual_gap':float(np.mean([r['selected_w2_squared']-r['same_teacher_index_w2_squared'] for r in rows])),
 'mean_selected_vs_nearest_gap':float(np.mean([r['selected_w2_squared']-r['nearest_student_component_w2_squared'] for r in rows]))}
(OUT/'SUMMARY.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
sources[str(OUT/'SUMMARY.json')]=hashlib.sha256((OUT/'SUMMARY.json').read_bytes()).hexdigest()
(OUT/'LOCAL_AUDIT.json').write_text(json.dumps({'status':'PASS_LOCAL_SCALAR_DRIFT_AUDIT_NOT_CAUSAL_CLOSED_LOOP_PROOF','sources':sources,'strict_checkpoints':6,'windows':1356,'valid_expert_windows':sum(s['all']['valid_expert'] for s in summaries.values()),'model_windows':2712,'forward_calls':174,'new_updates':0,'new_environments':0,'failed_compatibility_attempt_forwards':0},indent=2)+'\n',encoding='utf8')
print(json.dumps({k:{'all':v['all'],'closed_loop':v['closed_loop_existing_percent'],'expert':{m:{f:stat['mean'] for f,stat in vv.items()} for m,vv in v['expert'].items()},'translation':v['numeric']['translation_mean_l2_delta'],'rotation':v['numeric']['rotation_mean_l2_delta'],'gripper':v['numeric']['gripper_mean_l2_delta'],'W2':v['numeric']['selected_w2_squared'],'variance_fraction':v['variance_fraction_of_total_w2'],'selected_vs_nearest_gap':v['mean_selected_vs_nearest_gap']} for k,v in summaries.items()},ensure_ascii=False,indent=2))
