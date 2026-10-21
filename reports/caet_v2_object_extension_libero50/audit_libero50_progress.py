"""Independent metadata admission/progress audit, not pretraining completion."""
import datetime,hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);raw=(p/'SNAPSHOT.json').read_bytes();d=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()==json.loads((p/'RECEIPT.json').read_text())['sha256']
assert d['hashes'] and all(d['hashes'].values())
O='/data/mlr_ifa/experiments/LIBERO50_reproduced_pretraining_seed42_20261021_taskmap01'
D='/data/mlr_ifa/experiment_workspaces/CAET_v2_OBJECT_extension_LIBERO50_20261021/libero50_taskmap01'
def get(root,name):return d['files'][root+'/'+name]
gate=get(O,'PRETRAIN_GPU_PREFLIGHT.json');launch=get(O,'LAUNCH_ACCEPTANCE.json')
assert gate['status']=='PASS' and gate['optimizer_calls']==4 and gate['environment_steps']==gate['formal_updates']==0
assert gate['batch_size']==10 and gate['fp32'] and gate['frozen_clip_parameters_unchanged'] and gate['resume_tensor_exact']
assert gate['expected_steps']==1941000
assert launch['status']=='PASS' and launch['gpu_preflight_sha256']==d['file_sha256'][O+'/PRETRAIN_GPU_PREFLIGHT.json']
assert launch['contract_sha256']==d['file_sha256'][O+'/CONTRACT.json']
assert get(D,'MAPPING_AUDIT.json')['status']=='PASS_50_NATIVE_IDENTITIES'
for name in ('software','ra_software','prepare','gpu_preflight'):
 assert get(D+'/shared_pretraining/admission',name+'.exit').strip()=='0'
initial=d['files'].get(O+'/PRETRAIN_INITIALIZATION.json')
if initial:
 assert initial['seed']==42 and (initial['windows']+9)//10==19410 and initial['demos']==1250
 assert initial['batches']==19410 and initial['epochs']==100 and initial['expected_steps']==1941000
 assert initial['official_task_checkpoint_loaded'] is False
epochs=d['files'].get(O+'/pretraining/epochs.json',[])
for i,row in enumerate(epochs):
 assert row['epoch']==i and row['steps']==(i+1)*19410 and row['batches']==19410 and row['seconds']>0
assert len(epochs)<=100
for name in ('pretrain.exit','pipeline.exit','supervisor.exit'):
 if O+'/'+name in d['files']:assert get(O,name).strip()=='0'
assert D+'/FAILURE.json' not in d['files']
active=any('l50_exp.pretrain' in x['command'] and 'preflight' not in x['command'] for x in d['processes'])
assert active or (O+'/PRETRAIN_COMPLETE.json' in d['files'])
warm_step=gate['measured_two_updates_seconds'][1]/2
out=dict(status='PARTIAL_LIBERO50_SHARED_ADMISSION_PROGRESS_PASS',observed_at=datetime.datetime.fromtimestamp(d['observed_unix']).astimezone().isoformat(),hashes_verified=len(d['hashes']),gpu_admission_updates=4,initialization_present=initial is not None,completed_epochs=len(epochs),completed_epoch_updates=len(epochs)*19410,within_epoch_updates='not instrumented; not inferred',expected_updates=1941000,real_pretrain_process=active,formal_environment_records=0,preflight_warm_seconds_per_update=warm_step,preflight_linear_days=1941000*warm_step/86400,estimate_limit='Only two warm isolated updates; revise after complete monotonic-timed epoch; excludes downstream',peak_gpu_bytes=gate['peak_gpu_bytes'],gpu=d['gpu'],disk=d['disk'],not_final_acceptance=True)
if epochs:out['epoch_based_remaining_days']=(100-len(epochs))*sum(x['seconds'] for x in epochs[-3:])/len(epochs[-3:])/86400
(p/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
