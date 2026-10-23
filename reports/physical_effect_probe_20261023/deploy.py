"""Deploy a new immutable independent package; credentials stay in getpass memory."""
import getpass,json,hashlib,datetime,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023_engineering03'
def main():
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 dest=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));dest.mkdir()
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None;sftp=c.open_sftp()
  sftp.mkdir(Q);sftp.mkdir(Q+'/physical_effect_probe')
  paths=[R/'PROTOCOL.md',R/'USER_METHOD.txt',R/'controller.py']+sorted((R/'physical_effect_probe').glob('*.py'))
  files={}
  for p in paths:
   rel=p.relative_to(R).as_posix();data=p.read_bytes();target=Q+'/'+rel
   with sftp.open(target,'wb') as f:f.write(data)
   files[target]=hashlib.sha256(data).hexdigest()
  code='''
import hashlib,json,os,subprocess
from pathlib import Path
Q=Path(%r);files=%r
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
assert all(sha(p)==h for p,h in files.items())
S=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01/ALIGN/source')
for p in S.rglob('*.py'):files[str(p)]=sha(p)
P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
for n in ['resolved_eval.json','resolved_ra_coreset.json','bindings/task6.json','bindings/task7.json','ra_coreset/storage/task6.json','ra_coreset/storage/task7.json']:files[str(P/n)]=sha(P/n)
cp='/data/mlr_ifa/checkpoints/object_multitask_model_ep99.pth';files[cp]=sha(cp)
assert files[cp]=='159351b1df166dd5391c6b1e1317c0ab299fc8b1447fe466acef14ed7896bb7b'
contract=dict(version='PEP_A_v1',files=files,scope='OBJECT41 Task6/7; A gate first; B requires separate implemented admission')
(Q/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\\n')
env=os.environ.copy();env.update(PYTHONPATH=str(Q)+':'+str(S),PYTHONDONTWRITEBYTECODE='1',PEP_ROOT=str(Q),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/mlr_ifa/cache/huggingface',MPLBACKEND='Agg',MUJOCO_GL='egl')
log=open(Q/'controller.log','wb');p=subprocess.Popen(['/data/mlr_ifa/env/bin/python',str(Q/'controller.py')],cwd=str(Q),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(Q/'controller.pid').write_text(str(p.pid)+'\\n');print(json.dumps(dict(status='LAUNCHED_NOT_ACCEPTED',pid=p.pid,root=str(Q),contract_sha256=sha(Q/'CONTRACT.json'),frozen_files=len(files))))
'''%(Q,files)
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(code),timeout=180);i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
  (dest/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err
  (dest/'EVIDENCE.json').write_bytes(raw);sftp.get(Q+'/CONTRACT.json',str(dest/'CONTRACT.json'));print(raw.decode())
  (R/'ACTIVE_ROOT.json').write_text(json.dumps(dict(root=Q,launch=str(dest.name)),indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
