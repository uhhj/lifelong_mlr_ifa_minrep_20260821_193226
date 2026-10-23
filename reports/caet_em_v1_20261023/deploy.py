"""Create one immutable experiment root; credentials only in getpass memory."""
import getpass,json,hashlib,datetime,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent;Q='/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261023_v1_engineering04'
def main():
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ');out=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None;sftp=c.open_sftp();files={}
  try:sftp.stat(Q);existing=True
  except FileNotFoundError:sftp.mkdir(Q);sftp.mkdir(Q+'/caet_em_exp');existing=False
  if existing:
   assert 'CONTRACT.json' not in sftp.listdir(Q) and 'controller.pid' not in sftp.listdir(Q),'Already frozen/launched'
  for p in [R/'PROTOCOL.md',R/'USER_METHOD.txt',R/'controller.py',*sorted((R/'caet_em_exp').glob('*.py'))]:
   target=Q+'/'+p.relative_to(R).as_posix();raw=p.read_bytes()
   if existing:
    with sftp.open(target,'rb') as f:assert f.read()==raw,'Uploaded scientific source changed'
   else:
    with sftp.open(target,'wb') as f:f.write(raw)
   files[target]=hashlib.sha256(raw).hexdigest()
  code='''
import os,json,hashlib,subprocess,fcntl
from pathlib import Path
Q=Path(__ROOT__);files=__FILES__
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert all(sha(p)==h for p,h in files.items())
FAILED=Path('/data/mlr_ifa/experiment_workspaces/CAET_EM_OBJECT41_20261023_v1_engineering03');fc=json.loads((FAILED/'CONTRACT.json').read_text())
assert sha(FAILED/'CONTRACT.json')=='05c9bdf09971539b38cc715dc6b3eb53509c745134b90995c919b0e63e4c0e68' and all(sha(p)==h for p,h in fc['files'].items())
assert (FAILED/'controller.exit').read_text().strip()=='1' and (FAILED/'phases/evidence6.exit').read_text().strip()=='0' and (FAILED/'phases/codec_admission.exit').read_text().strip()=='1'
assert not (FAILED/'shared/codec_frozen.pth').exists() and not (FAILED/'shared/codec_recovery.pth').exists()
assert json.loads((FAILED/'ADMISSION_CODEC.json').read_text())['isolated_optimizer_calls']==3
assert "event() got multiple values for argument 'kind'" in (FAILED/'phases/codec_admission.log').read_text()
for pp in Path('/proc').iterdir():
 if pp.name.isdigit():
  try:ee=(pp/'environ').read_bytes().split(b'\\0')
  except OSError:continue
  assert ('EM_ROOT='+str(FAILED)).encode() not in ee
for p in [FAILED/'CONTRACT.json',FAILED/'controller.exit',FAILED/'FAILURE.json',FAILED/'phases/codec_admission.log',FAILED/'ADMISSION_CODEC.json']:files[str(p)]=sha(p)
(Q/'shared').mkdir();(Q/'evidence').mkdir();(Q/'evidence/task6').symlink_to(FAILED/'evidence/task6',target_is_directory=True)
meta=json.loads((FAILED/'evidence/task6/TEMPORAL_BINDING_AUDIT.json').read_text());assert sha(FAILED/'evidence/task6/labels.pth')==meta['labels_sha256']
for name in ('projection.pth','normalization.pth'):
 target=FAILED/'shared'/name;assert sha(target)==meta[name.split('.')[0]+'_sha256'];(Q/'shared'/name).symlink_to(target);files[str(target)]=sha(target)
for p in [FAILED/'evidence/task6/labels.pth',FAILED/'evidence/task6/TEMPORAL_BINDING_AUDIT.json']:files[str(p)]=sha(p)
S=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source');P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
for p in S.rglob('*'):
 if p.is_file() and p.suffix in ('.py','.yaml','.json') and '__pycache__' not in p.parts:files[str(p)]=sha(p)
for p in [P/'CONTRACT.json',P/'MODAL_CONTRACT.json',P/'FEATURES.json',P/'resolved_eval.json',Path('/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth')]:files[str(p)]=sha(p)
for t in range(6,10):
 for p in [P/'bindings'/('task%d.json'%t),P/'ra_coreset/storage'/('task%d.json'%t)]:files[str(p)]=sha(p)
 binding=json.loads((P/'bindings'/('task%d.json'%t)).read_text());assert sha(binding['hdf5'])==binding['hdf5_sha256'];files[binding['hdf5']]=binding['hdf5_sha256']
OLD=Path('/data/mlr_ifa/experiment_workspaces/READOUT_CAPACITY_OBJECT41_20261023_v1_engineering02');assert (OLD/'controller.exit').read_text().strip()=='0'
assert sha(OLD/'CONTRACT.json')=='8ad12b59271e3d6478c4b8e6a2480adba5ccc55b8b38ec5fe1fe6d4e305eb9b0'
files[str(OLD/'CONTRACT.json')]=sha(OLD/'CONTRACT.json');files[str(OLD/'controller.exit')]=sha(OLD/'controller.exit')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
with open('/data/mlr_ifa/physical_effect_probe.lock','a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lock,fcntl.LOCK_UN)
d=os.statvfs('/data');s=os.statvfs('/');assert d.f_bfree*d.f_frsize>=int(4.1*1024**3) and s.f_bavail*s.f_frsize>=int(4.6*1024**3)
arms={a:('/root' if a in ('B0','B1') else '/data')+'/mlr_ifa/experiments/CAET_EM_'+a+'_OBJECT_seed41_20261023_engineering04' for a in ('B0','B1','B2')}
for path in arms.values():Path(path).mkdir(exist_ok=False)
(Q/'ENTRY.json').write_text(json.dumps(dict(root=str(Q),arms=arms),indent=2)+'\\n');files[str(Q/'ENTRY.json')]=sha(Q/'ENTRY.json')
(Q/'CONTRACT.json').write_text(json.dumps(dict(version='CAET-EM-v1',seed=41,arms=list(arms),files=files,main_epochs_per_arm=400,main_steps_per_arm=105500,formal_episodes_per_arm=200,automatic_seed_expansion=False),indent=2)+'\\n')
env=os.environ.copy();env.update(EM_ROOT=str(Q),PYTHONPATH=str(Q)+':'+str(S),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/mlr_ifa/cache/huggingface',MPLBACKEND='Agg',MUJOCO_GL='egl')
log=open(Q/'controller.log','wb');p=subprocess.Popen(['/data/mlr_ifa/env/bin/python',str(Q/'controller.py')],cwd=str(Q),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);(Q/'controller.pid').write_text(str(p.pid)+'\\n')
print(json.dumps(dict(status='LAUNCHED_NOT_ADMITTED',pid=p.pid,root=str(Q),contract_sha256=sha(Q/'CONTRACT.json'),frozen_files=len(files),arms=arms)))
'''.replace('__ROOT__',repr(Q)).replace('__FILES__',repr(files))
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(code),timeout=600);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));(out/'STDOUT.txt').write_bytes(raw);assert rc==0,err
  result=json.loads(raw);(out/'EVIDENCE.json').write_text(json.dumps(result,indent=2));sftp.get(Q+'/CONTRACT.json',str(out/'CONTRACT.json'));sftp.get(Q+'/ENTRY.json',str(out/'ENTRY.json'));(R/'ACTIVE_ROOT.json').write_text(json.dumps(dict(root=Q,launch=out.name),indent=2));print(json.dumps(result,indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
