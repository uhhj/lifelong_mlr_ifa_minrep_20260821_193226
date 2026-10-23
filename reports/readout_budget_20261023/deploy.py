"""Create one immutable new root; no credential files or overwriting scientific sources."""
import getpass,json,hashlib,datetime,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q='/data/mlr_ifa/experiment_workspaces/READOUT_BUDGET_OBJECT41_20261023_v1'
def main():
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 dest=R/('launch_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));dest.mkdir()
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None;sftp=c.open_sftp();sftp.mkdir(Q);sftp.mkdir(Q+'/pbd');files={}
  for p in [R/'PROTOCOL.md',R/'PARENT_LOCAL_ACCEPTANCE.json',R/'controller.py',*sorted((R/'pbd').glob('*.py'))]:
   target=Q+'/'+p.relative_to(R).as_posix();raw=p.read_bytes()
   with sftp.open(target,'wb') as f:f.write(raw)
   files[target]=hashlib.sha256(raw).hexdigest()
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
V2=Path('/data/mlr_ifa/experiment_workspaces/READOUT_PATCH_OBJECT41_20261023_v2')
assert sha(V2/'CONTRACT.json')=='a5667bde73769a1385c05fd1989d839a7e1cb30ab76853e225c579436d0c65d1'
vc=json.loads((V2/'CONTRACT.json').read_text());assert all(sha(p)==h for p,h in vc['files'].items())
assert (V2/'controller.exit').read_text().strip()=='0' and all(p.read_text().strip()=='0' for p in V2.glob('*.exit'))
assert sha(V2/'pbd/models.py')==sha(Q/'pbd/models.py')
assert sha(V2/'output/PATCH_MANIFEST.json')=='8344cac6065e3ab190d7343c5f9d35e51eeb81226449b35f181ddcb29389fcee'
for name,h in [('patch6.pth','ea97a9689454f3feccd042c3268ff12ed554634a3ade2853a11b4490ff6d3d3b'),('patch7.pth','2a9b61d9fb0e46039876b3cb83fbbf20d20e9c8bdc3ab54e8715a36334bfa5fa')]:assert sha(V2/'output/private'/name)==h
for p in [V2/'CONTRACT.json',V2/'controller.exit',V2/'output/PATCH_MANIFEST.json',V2/'output/A_GATE.json']:files[str(p)]=sha(p)
for p in (V2/'output/cases').glob('*.json'):files[str(p)]=sha(p)
for pp in Path('/proc').iterdir():
 if pp.name.isdigit():
  try:envbytes=(pp/'environ').read_bytes().split(b'\\0')
  except OSError:continue
  assert ('PBD_ROOT='+str(V2)).encode() not in envbytes,'Parent process remains'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
ds=os.statvfs('/data');ss=os.statvfs('/');assert ds.f_bfree*ds.f_frsize>=int(2.5*1024**3) and ss.f_bavail*ss.f_frsize>=1024**3
OLD=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023_engineering03');old=json.loads((OLD/'CONTRACT.json').read_text());assert (OLD/'controller.exit').read_text().strip()=='0';assert all(sha(p)==h for p,h in old['files'].items())
for p in (OLD/'physical_effect_probe').glob('*.py'):files[str(p)]=sha(p)
for n in ['DEMO_SPLITS.json','NATIVE_VIEW_MANIFEST.json','INPUT_BINDING_AUDIT.json','STATE_PROBE_RESULTS.json']:files[str(OLD/'output'/n)]=sha(OLD/'output'/n)
contract=dict(version='READOUT_BUDGET_A_v1',v2_contract_sha256=sha(V2/'CONTRACT.json'),files=files,old_contract_sha256=sha(OLD/'CONTRACT.json'),scope='Only training budget 200; fixed P0/G/S;50/100 descriptive,200 gate; no B')
(Q/'CONTRACT.json').write_text(json.dumps(contract,indent=2)+'\\n')
env=os.environ.copy();env.update(PYTHONPATH=str(Q)+':'+str(OLD),PYTHONDONTWRITEBYTECODE='1',PBD_ROOT=str(Q),PEP_ROOT=str(OLD),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/mlr_ifa/cache/huggingface',MPLBACKEND='Agg',MUJOCO_GL='egl')
log=open(Q/'controller.log','wb');p=subprocess.Popen(['/data/mlr_ifa/env/bin/python',str(Q/'controller.py')],cwd=str(Q),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(Q/'controller.pid').write_text(str(p.pid)+'\\n');print(json.dumps(dict(status='LAUNCHED_NOT_ACCEPTED',pid=p.pid,root=str(Q),contract_sha256=sha(Q/'CONTRACT.json'),frozen_files=len(files))))
'''%(Q,files)
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(code),timeout=180);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(dest/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err
  (dest/'EVIDENCE.json').write_bytes(raw);sftp.get(Q+'/CONTRACT.json',str(dest/'CONTRACT.json'));(R/'ACTIVE_ROOT.json').write_text(json.dumps(dict(root=Q,launch=dest.name),indent=2));print(raw.decode())
 finally:pw=None;c.close()
if __name__=='__main__':main()
