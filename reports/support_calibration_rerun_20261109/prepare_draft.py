from pathlib import Path
import json,hashlib,ast,datetime
R=Path('D:/论文课题研究/MLR+IFA/execution/20261109_support_calibration_rerun');E=R.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:p.read_text(encoding='utf-8-sig')
records=[]
for suite in ['OBJECT','GOAL']:
 old=E/f'20261108_{suite.lower()}_s_q1_ablation';new=R/suite
 assert not new.exists(),'Do not overwrite rerun preparation'
 new.mkdir()
 oldname=f'S_Q1_{suite}41_42_43_20261108_v1';newname=f'S_Q1_{suite}41_42_43_20261109_rerun_v1'
 files={}
 # Copy only reviewed source and baseline bindings, never prior run output/state/contracts.
 names=['controller_parallel.py','train_wrapper.py','environment_locked.py','check_readonly.py','audit_progress.py','audit_seed_evaluation.py','audit_final_local.py','compare_primary_s.py','final_readonly.py','final_Q1_41_readonly.py','final_Q1_42_readonly.py','final_Q1_43_readonly.py','ENGINEERING_REUSE.json','S_ACCEPTANCE.json']
 paths=[old/n for n in names]
 for seed in [41,42,43]:paths += [p for p in (old/f'Q1_{seed}').iterdir() if p.is_file() and (p.suffix=='.py' or p.name in ['PARENT_BINDING.json','INITIAL_SPLIT.json','S_REFERENCE.json'])]
 for p in paths:
  rel=p.relative_to(old);text=read(p);s=text.replace(oldname,newname)
  if p.name=='initialization.py':
   assert "hardware['device']=='NVIDIA GeForce RTX 3090'" in s
   s=s.replace("hardware['device']=='NVIDIA GeForce RTX 3090'","hardware['device']=='NVIDIA GeForce RTX 4080 SUPER'").replace('USER_AUTHORIZED_RTX3090','USER_AUTHORIZED_Q1_RERUN_OBSERVED_RTX4080_SUPER')
  if p.suffix=='.py':ast.parse(s,feature_version=(3,8))
  else:json.loads(s)
  dest=new/rel;dest.parent.mkdir(exist_ok=True);dest.write_text(s,encoding='utf8',newline='\n')
  files[rel.as_posix()]=dict(source=str(p),source_sha256=sha(p),prepared_sha256=sha(dest),change='new output identity; initialization hardware guard reflects observed device' if s!=text else 'identical text')
 (new/'SOURCE_PREPARATION.json').write_text(json.dumps(dict(status='DRAFT_NOT_FROZEN_NOT_DEPLOYED',suite=suite,root='/data/mlr_ifa/experiment_workspaces/'+newname,files=files,final_wrapper_status='STATIC_ONLY_PENDING_FULL_REVIEW_NO_ACCEPTANCE'),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 records.append(dict(suite=suite,prepared_files=len(files),name=newname,old_acceptance_sha256=sha(old/'ACCEPTANCE.json')))
evpath=R/'capacity_readonly_20261109_151837/EVIDENCE.json';ev=json.loads(read(evpath));free=ev['capacity']['/data']['available_bytes']
plan=dict(status='PREFLIGHT_BLOCKED_CAPACITY_NOT_LAUNCHED',authorized='Only q=1 OBJECT and GOAL seed41/42/43; six runs, old calibrated S reused, all earlier results preserved',planned_order='OBJECT three train (max two parallel) then 600 formal, then GOAL three train (max two parallel) then 600 formal; no overlap between suites',hardware=ev['hardware'],capacity=dict(available_bytes=free,conservative_new_outputs_and_save_peak_bytes=8*1024**3,reserve_bytes=2*1024**3,required_bytes=10*1024**3,shortfall_bytes=10*1024**3-free,recommended_additional_free_bytes=5*1024**3),source_evidence=str(evpath),source_evidence_sha256=sha(evpath),prepared=records,training_calls=0,environment_calls=0,deployments=0,baseline_retrained=False,baseline_reevaluated=False,old_results_changed=False,local_transport_dependency_failure=dict(interpreter='bundled python',error='paramiko unavailable before SSH import; zero remote command',resolution='existing Python313 with installed paramiko; read-only preflight succeeded'),observation=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
(R/'PREPARATION_STATUS.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=plan['status'],prepared=records,available_gib=free/1024**3,shortfall_gib=(10*1024**3-free)/1024**3),ensure_ascii=False,indent=2))
