from pathlib import Path
import json,hashlib,datetime,struct
R=Path(__file__).resolve().parent.parent;B=R.parent;A=Path(__file__).resolve().parent
j=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
remote=j(A/'capacity_readonly_20261110_165538/EVIDENCE.json')
accept=j(B/'20261109_q1_user_resume/ACCEPTANCE.json');out={};differences=[]
for suite in ['OBJECT','GOAL']:
 entry=accept['suites'][suite];n=j(Path(entry['current_final'])/'EVIDENCE.json')['light'];s=j(Path(entry['baseline']['final_path'])/'EVIDENCE.json')['light']
 for seed in [41,42,43]:
  key=f'{suite}_{seed}';read=lambda L,k:json.loads(L[k]['text'])
  rows=read(n,f'Q1_{seed}/EPOCHS_S.json');ref=read(s,f'seed{seed}/EPOCHS_S.json');assert len(rows)==len(ref)==400
  cols=['stage','epoch','steps','batches','lr_first','lr_after','sampler_sha256','physical_optimizer_calls','gate_blocks']
  bad=[(i,k) for i,(x,y) in enumerate(zip(rows,ref)) for k in cols if x[k]!=y[k]];assert not bad,(key,bad)
  init=read(n,f'Q1_{seed}/INITIAL_S.json');old=read(s,f'seed{seed}/INITIAL_S.json');assert init['tensors']==old['tensors'] and init['rng']==old['rng']
  stages={}
  for t in range(6,10):
   inp=read(n,f'Q1_{seed}/TRAIN_INPUT_S_{t}.json');oldinp=read(s,f'seed{seed}/TRAIN_INPUT_S_{t}.json')
   assert inp['manifest_sha256']==oldinp['manifest_sha256'] and inp['lambda_R']==.1
   if inp['scale']!=oldinp['scale']:
    bits=lambda z:struct.unpack('I',struct.pack('f',z))[0]
    differences.append(dict(case=key,stage=t,actual=inp['scale'],S=oldinp['scale'],ulp=[abs(bits(x)-bits(y)) for x,y in zip(inp['scale'],oldinp['scale'])],max_absolute=max(abs(x-y) for x,y in zip(inp['scale'],oldinp['scale'])),same_scale_function=remote['q1'][key]['files']['train.py']['same_top_level_functions'].count('scale_for')==1,cause='NOT_ESTABLISHED_BY_THIS_READONLY_AUDIT'))
   assert all(m['effective_support']==(1. if m['count'] else 0.) for m in inp['memory'].values())
   ad=read(n,f'Q1_{seed}/ADMISSION_S_{t}.json')
   if t>6:assert ad['q1_actual_memory_uniform_exact'] and ad['q1_short_history_exact'] and all(x['raw_q_point_one_ignored'] for x in ad['short_history'])
   stages[t]=dict(same_manifest=True,same_action_scale=inp['scale']==oldinp['scale'],lambda_R=inp['lambda_R'],support=[dict(task=k,raw=m['support'],effective=m['effective_support'],rows=m['count'],same_cache_as_S=m['cache_sha256']==oldinp['memory'][k]['cache_sha256']) for k,m in inp['memory'].items()],admission=ad['status'])
  for f,v in remote['q1'][key]['files'].items():
   local=B/'20261109_support_calibration_rerun'/suite/f'Q1_{seed}'/f
   assert hashlib.sha256(local.read_text(encoding='utf8').encode()).hexdigest()==v['actual_sha256'],(key,f)
  hard=read(n,f'Q1_{seed}/HARDWARE_MIGRATION.json');par=read(n,f'Q1_{seed}/TASK6_HARDWARE_COMPARISON.json')
  caches={}
  for t in range(6,10):
   cur=read(n,f'Q1_{seed}/MEMORY_S_{t}.json');bas=read(s,f'seed{seed}/MEMORY_S_{t}.json')
   identity=lambda z:(z['trajectory'],z['initial_index'],z['step'],z['length'])
   ci={identity(z):z for z in cur['rows']};bi={identity(z):z for z in bas['rows']};common=ci.keys()&bi.keys()
   assert len(ci)==cur['count'] and len(bi)==bas['count']
   caches[t]=dict(actual_successes=cur['collection_successes'],S_successes=bas['collection_successes'],actual_windows=len(ci),S_windows=len(bi),same_window_identity_set=ci.keys()==bi.keys(),common_window_identities=len(common),same_feature_hashes=sum(ci[z]['feature_sha256']==bi[z]['feature_sha256'] for z in common),same_teacher_mu_hashes=sum(ci[z]['mu_sha256']==bi[z]['mu_sha256'] for z in common),same_teacher_sd_hashes=sum(ci[z]['sd_sha256']==bi[z]['sd_sha256'] for z in common),identical_full_row_metadata=cur['rows']==bas['rows'],terminal_unused=t==9)
  out[key]=dict(epochs=400,exact_matching_epoch_protocol_fields=cols,initial_tensors_rng_equal=True,recorded_hardware_equal=hard['hardware_equal'],Task6_all_tensors_exact=par['all_tensors_exact'],stages=stages,online_cache_content_comparison=caches,final_evidence_sha256=sha(Path(entry['current_final'])/'EVIDENCE.json'),S_evidence_sha256=sha(Path(entry['baseline']['final_path'])/'EVIDENCE.json'))
review=j(R/'SOURCE_REVIEW.json');align={}
for case,x in review['cases'].items():
 checked={}
 for name in ['coreset_exp/evaluate.py','coreset_exp/eval_metric.py','coreset_exp/model.py','libero/lifelong/models/complete_policy.py']:
  bind=x['exact_parent_sources'][name];parent=B.parent/bind['original'];actual=R/'cases'/case/'source'/name
  a=actual.read_text(encoding='utf8');b=parent.read_text(encoding='utf8');assert a==b,(case,name)
  checked[name]=dict(equal_normalized_text=True,sha256=hashlib.sha256(a.encode()).hexdigest(),parent_path=str(parent))
 align[case]=checked
report=dict(status='AUDITED_PROTOCOL_MATCH_WITH_RECORDED_NUMERICAL_AND_ENGINEERING_DIFFERENCES',observed=datetime.datetime.now().astimezone().isoformat(),q1=out,action_scale_differences=differences,align_parent_files=align,q1_actual_remote_source_sha256=sha(A/'capacity_readonly_20261110_165538/EVIDENCE.json'),align_input_audit_sha256=sha(R/'INPUT_AUDIT.json'),q1_acceptance_sha256=sha(B/'20261109_q1_user_resume/ACCEPTANCE.json'),new_training_or_environment_calls=0)
(A/'LOCAL_PROTOCOL_AUDIT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=report['status'],q1_exact_epoch_protocol_rows=2400,scale_differences=differences,q1_Task6_tensor_equal={k:v['Task6_all_tensors_exact'] for k,v in out.items()})))
