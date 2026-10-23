"""One versioned continuation; reuse completed immutable assets, no completed training rerun."""
import getpass,json,hashlib,datetime,shlex,ast
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261023_v1_engineering05'
def main():
 sources=[R/'PROTOCOL.md',R/'USER_METHOD.txt',R/'controller.py',*sorted((R/'caet_em_exp').glob('*.py'))]
 for p in sources:
  if p.suffix=='.py':ast.parse(p.read_text(),feature_version=(3,8))
 proof=json.loads((R/'repair05_preflight_readonly_20261023_221421/EVIDENCE.json').read_text());assert proof['status']=='PASS_STOPPED_COMPLETE_STAGE_REUSE'
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ');out=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None;sftp=c.open_sftp()
  try:sftp.stat(Q);raise RuntimeError('Continuation root already exists; inspect before any retry')
  except FileNotFoundError:pass
  sftp.mkdir(Q);sftp.mkdir(Q+'/caet_em_exp');files={}
  for p in sources:
   raw=p.read_bytes();target=Q+'/'+p.relative_to(R).as_posix()
   with sftp.open(target,'wb') as f:f.write(raw)
   files[target]=hashlib.sha256(raw).hexdigest()
  code='''
import os,sys,json,hashlib,subprocess,fcntl,shutil
from pathlib import Path
Q=Path(__ROOT__);files=__FILES__;expected=__PROOF__
OLD=Path('/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261023_v1_engineering04')
S=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source')
os.environ.update(EM_ROOT=str(Q),LIBERO_CONFIG_PATH='/data/mlr_ifa/config',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/mlr_ifa/cache/huggingface',MPLBACKEND='Agg',MUJOCO_GL='egl')
sys.path.insert(0,str(Q))
from caet_em_exp.common import sha,policy_fingerprint
import torch
parent=json.loads((OLD/'CONTRACT.json').read_text());parent_sha=sha(OLD/'CONTRACT.json');assert parent_sha==expected['contract']
assert all(sha(p)==h for p,h in parent['files'].items()) and all(sha(p)==h for p,h in files.items())
assert (OLD/'controller.exit').read_text().strip()=='1' and (OLD/'phases/B0_train7.exit').read_text().strip()=='0' and (OLD/'phases/B0_diagnose7.exit').read_text().strip()=='1'
assert "KeyError: 't7__1360.pt'" in (OLD/'phases/B0_diagnose7.log').read_text()
for pp in Path('/proc').iterdir():
 if pp.name.isdigit() and int(pp.name)!=os.getpid():
  try:env=(pp/'environ').read_bytes().split(b'\\0')
  except OSError:continue
  assert ('EM_ROOT='+str(OLD)).encode() not in env and ('EM_ROOT='+str(Q)).encode() not in env
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with open('/data/mlr_ifa/physical_effect_probe.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
entry=json.loads((OLD/'ENTRY.json').read_text());arms=entry['arms'];b0=Path(arms['B0']);recovery=b0/'recovery/latest.pth'
assert sha(recovery)==expected['recovery']['sha256'] and not (b0/'models/task8.pth').exists() and not list(b0.glob('evaluation/trials*.jsonl'))
assert all(not list(Path(arms[a]).rglob('*.pth')) for a in ('B1','B2'))
d=os.statvfs('/data');s=os.statvfs('/');assert d.f_bfree*d.f_frsize>=int(4.1*1024**3)+recovery.stat().st_size and s.f_bavail*s.f_frsize>=int(4.6*1024**3)
(Q/'provenance').mkdir();preserved=Q/'provenance/B0_task7_complete_recovery.pth';shutil.copyfile(recovery,preserved);assert sha(preserved)==expected['recovery']['sha256']
(Q/'shared').symlink_to(OLD/'shared',target_is_directory=True);(Q/'evidence').mkdir();(Q/'phases').mkdir()
for t in (6,7):(Q/'evidence'/('task%d'%t)).symlink_to(OLD/'evidence'/('task%d'%t),target_is_directory=True)
for name in ('INPUT_MANIFEST.json','CAPACITY_PLAN.json','ADMISSION_CODEC.json','CODEC_TRAINING_AND_VALIDATION.json','CODEC_FREEZE_MANIFEST.json'):
 (Q/name).symlink_to(OLD/name)
for name in ('prepare','evidence7','codec_admission','codec','B0_train6','B0_train7','B0_diagnose6'):
 assert (OLD/'phases'/(name+'.exit')).read_text().strip()=='0'
 for ext in ('.exit','.log'):(Q/'phases'/(name+ext)).symlink_to(OLD/'phases'/(name+ext))
files.update(parent['files'])
for p in [OLD/'CONTRACT.json',OLD/'ENTRY.json',OLD/'controller.exit',OLD/'FAILURE.json',OLD/'EVENTS.jsonl',preserved,*list((OLD/'phases').glob('*'))]:
 if p.is_file():files[str(p)]=sha(p)
for name in ('INPUT_MANIFEST.json','CAPACITY_PLAN.json','ADMISSION_CODEC.json','ADMISSION_B0.json','CODEC_TRAINING_AND_VALIDATION.json','CODEC_FREEZE_MANIFEST.json'):
 files[str(OLD/name)]=sha(OLD/name)
for p in (OLD/'shared').rglob('*'):
 if p.is_file():files[str(p)]=sha(p)
for t in (6,7):
 for p in (OLD/'evidence'/('task%d'%t)).rglob('*'):
  if p.is_file():files[str(p)]=sha(p)
accepted={}
for t in (6,7):
 cp=b0/'models'/('task%d.pth'%t);assert sha(cp)==expected['checkpoints'][str(t)]['checkpoint_sha256'];raw=torch.load(cp,map_location='cpu')
 accepted['B0_task%d'%t]=dict(path=str(cp),sha256=sha(cp),payload_fingerprint=policy_fingerprint(raw))
 for sub,ext in [('models','.pth'),('epochs','.json'),('dataset','.json'),('optimizer','.json'),('storage','.json'),('sidecars','.pth'),('sidecars','.json'),('TASK_COMPLETE','.json')]:
  p=b0/sub/('task%d'%t+ext);files[str(p)]=sha(p)
files[str(b0/'diagnostics/task6.json')]=sha(b0/'diagnostics/task6.json')
for p in (b0/'buffer').glob('*.pt'):files[str(p)]=sha(p)
resume=dict(parent_root=str(OLD),parent_contract=parent_sha,parent_controller_exit=1,failure='B0_diagnose7 post-retain dataset KeyError',completed_main_epochs=200,completed_main_updates=38400,formal_episodes=0,accepted_checkpoints=accepted,preserved_recovery=dict(path=str(preserved),sha256=sha(preserved),bytes=preserved.stat().st_size),reuse_codec_sha256=expected['codec_sha256'],resume_at='B0_diagnose7 then B0_train8',completed_training_repeated=False,scientific_configuration_changed=False)
(Q/'RESUME_MANIFEST.json').write_text(json.dumps(resume,indent=2)+'\\n');files[str(Q/'RESUME_MANIFEST.json')]=sha(Q/'RESUME_MANIFEST.json')
(Q/'ENTRY.json').write_text(json.dumps(dict(root=str(Q),arms=arms),indent=2)+'\\n');files[str(Q/'ENTRY.json')]=sha(Q/'ENTRY.json')
assert all(sha(p)==h for p,h in files.items())
(Q/'CONTRACT.json').write_text(json.dumps(dict(version='CAET-EM-v1-engineering05',seed=41,arms=list(arms),files=files,parent_contract=parent_sha,resume_manifest_sha256=sha(Q/'RESUME_MANIFEST.json'),main_epochs_per_arm=400,main_steps_per_arm=105500,formal_episodes_per_arm=200,automatic_seed_expansion=False),indent=2)+'\\n')
env=os.environ.copy();env['PYTHONPATH']=str(Q)+':'+str(S)
log=open(Q/'controller.log','wb');p=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-B',str(Q/'controller.py')],cwd=str(Q),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);(Q/'controller.pid').write_text(str(p.pid)+'\\n')
print(json.dumps(dict(status='CONTINUATION_LAUNCHED_NOT_ADMITTED',pid=p.pid,root=str(Q),contract_sha256=sha(Q/'CONTRACT.json'),frozen_files=len(files),arms=arms,parent_contract=parent_sha,resume_at=resume['resume_at'],completed_main_updates_reused=38400,preserved_recovery_bytes=preserved.stat().st_size)))
'''.replace('__ROOT__',repr(Q)).replace('__FILES__',repr(files)).replace('__PROOF__',repr(proof))
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -c '+shlex.quote(code),timeout=600);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));(out/'STDOUT.txt').write_bytes(raw);assert rc==0,err
  result=json.loads(raw.decode().splitlines()[-1]);(out/'EVIDENCE.json').write_text(json.dumps(result,indent=2))
  for name in ('CONTRACT.json','ENTRY.json','RESUME_MANIFEST.json'):sftp.get(Q+'/'+name,str(out/name))
  (R/'ACTIVE_ROOT.json').write_text(json.dumps(dict(root=Q,launch=out.name),indent=2));print(json.dumps(result,indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
