"""Deploy one versioned recovery supervisor; never modify frozen scientific source."""
import ast,datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261024_recovery06'
proof=json.loads((R/'recovery06_preflight_readonly_20261024_132349/EVIDENCE.json').read_text())
assert proof['status']=='PASS_STOPPED_EXACT_EPOCH_RECOVERABLE'
sources=[R/'recovery06_controller.py',R/'recovery06_admission.py']
for p in sources:ast.parse(p.read_text(),feature_version=(3,8))
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
out=R/('recovery06_launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None;sftp=c.open_sftp()
 try:sftp.stat(Q);raise RuntimeError('Recovery root exists; inspect, never blindly redeploy')
 except FileNotFoundError:pass
 sftp.mkdir(Q);files={}
 for p in sources:
  raw=p.read_bytes();target=Q+'/'+p.name
  with sftp.open(target,'wb') as f:f.write(raw)
  files[target]=hashlib.sha256(raw).hexdigest()
 code='''
import os,sys,json,subprocess,shutil,fcntl,time
from pathlib import Path
Q=Path(__Q__);P=__PROOF__;files=__FILES__
ROOT=Path('/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261023_v1_engineering05')
os.environ.update(EM_ROOT=str(ROOT),EM_RECOVERY_ROOT=str(Q),LIBERO_CONFIG_PATH='/data/mlr_ifa/config',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/mlr_ifa/cache/huggingface',MPLBACKEND='Agg',MUJOCO_GL='egl',TOKENIZERS_PARALLELISM='false')
sys.path.insert(0,str(ROOT))
from caet_em_exp.common import *
assert verify()==P['contract'];assert not (ROOT/'controller.exit').exists()
for pp in Path('/proc').iterdir():
 if pp.name.isdigit() and int(pp.name)!=os.getpid():
  try:env=(pp/'environ').read_bytes().split(b'\\0')
  except OSError:continue
  assert ('EM_ROOT='+str(ROOT)).encode() not in env
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with open('/data/mlr_ifa/physical_effect_probe.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
root=arm_root('B2');rec=Path(P['recovery']['path']);assert sha(rec)==P['recovery']['sha256']
assert sha(root/'teachers/task8.pth')==P['recovery']['teacher_sha256'] and sha(root/'dataset/task8.json')==P['recovery']['dataset_sha256']
assert len(js(root/'epochs/task8.json'))==90 and not (root/'models/task8.pth').exists() and not list(root.glob('evaluation/trials*.jsonl'))
disk=capacity();stable=sum(p.stat().st_blocks*512 for p in root.rglob('*') if p.is_file() and not p.is_symlink() and p.stat().st_nlink==1)
remaining=max(0,int(1.8*1024**3)-stable);assert disk['/data']['root_free']-rec.stat().st_size-remaining>=2*1024**3
prov=Q/'provenance';prov.mkdir();preserved=prov/'B2_task8_epoch90_recovery.pth';shutil.copyfile(rec,preserved);assert sha(preserved)==P['recovery']['sha256']
for src,name in [(ROOT/'RUN_STATUS.json','interrupted_RUN_STATUS.json'),(ROOT/'controller.pid','interrupted_controller.pid.txt'),(ROOT/'controller.log','interrupted_controller.log.txt'),(ROOT/'EVENTS.jsonl','interrupted_EVENTS.jsonl'),(ROOT/'phases/B2_train8.log','interrupted_B2_train8.log.txt'),(ROOT/'ADMISSION_B2.json','prior_ADMISSION_B2.json'),(root/'epochs/task8.json','task8_epochs.json'),(root/'dataset/task8.json','task8_dataset.json'),(root/'teachers/task8.json','task8_teacher.json')]:
 shutil.copyfile(src,prov/name);files[str(prov/name)]=sha(prov/name)
files[str(preserved)]=sha(preserved)
for v in P['checkpoints'].values():assert sha(v['path'])==v['sha256'];files[v['path']]=v['sha256']
for p in [ROOT/'CONTRACT.json',root/'teachers/task8.pth',frozen_codec()]:files[str(p)]=sha(p)
manifest=dict(version='recovery06',science_root=str(ROOT),science_contract=P['contract'],cause=P['cause'],stopped_observation=P['observed_unix'],recovery=P['recovery'],preserved_recovery=str(preserved),resume_at='B2 task8 epoch90 native --resume',completed_main_updates_reused=277210,remaining_main_updates=39290,completed_training_repeated=False,scientific_changes=False,own_admission='native B2 3 isolated updates plus exact saved-state next-batch replay 2 isolated updates',interrupted_partial_epoch_work='Unknown; not accepted as complete. Restore last complete epoch only.',remaining_formal_episodes=200,automatic_seed_expansion=False)
write(Q/'RECOVERY_MANIFEST.json',manifest);files[str(Q/'RECOVERY_MANIFEST.json')]=sha(Q/'RECOVERY_MANIFEST.json')
assert all(sha(p)==h for p,h in files.items())
write(Q/'EXECUTION_CONTRACT.json',dict(version='CAET-EM-recovery06-orchestration-only',files=files,science_contract=P['contract']))
binding=dict(recovery_root=str(Q),execution_contract_sha256=sha(Q/'EXECUTION_CONTRACT.json'),manifest_sha256=sha(Q/'RECOVERY_MANIFEST.json'),science_contract=P['contract'],completed_training_repeated=False)
write(ROOT/'RECOVERY06_BINDING.json',binding)
env=os.environ.copy();env['PYTHONPATH']=str(ROOT)+':'+str(SOURCE)
with open(Q/'controller.log','wb') as log:p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'recovery06_controller.py')],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(Q/'controller.pid').write_text(str(p.pid)+'\\n');(ROOT/'controller.pid').write_text(str(p.pid)+'\\n')
print(json.dumps(dict(status='RECOVERY_LAUNCHED_NOT_ADMITTED',pid=p.pid,root=str(ROOT),recovery_root=str(Q),binding=binding,science_files_unchanged=True,next_epoch=90,steps=66210,remaining_main_updates=39290,data_margin_after_preservation_and_remaining=disk['/data']['root_free']-rec.stat().st_size-remaining-2*1024**3)))
'''.replace('__Q__',repr(Q)).replace('__PROOF__',repr(proof)).replace('__FILES__',repr(files))
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -c '+shlex.quote(code),timeout=600);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
 (out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));(out/'STDOUT.txt').write_bytes(raw);assert rc==0,err
 result=json.loads(raw.decode().splitlines()[-1]);(out/'EVIDENCE.json').write_text(json.dumps(result,indent=2))
 for n in ('EXECUTION_CONTRACT.json','RECOVERY_MANIFEST.json'):sftp.get(Q+'/'+n,str(out/n))
 active=json.loads((R/'ACTIVE_ROOT.json').read_text());active.update(recovery_root=Q,recovery_launch=out.name);(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,indent=2))
 print(json.dumps(result,indent=2))
finally:pw=None;c.close()
