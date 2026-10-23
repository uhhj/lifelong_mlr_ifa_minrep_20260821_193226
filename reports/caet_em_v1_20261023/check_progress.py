"""Read-only remote snapshot, allowlisted metadata only. No weight or payload download."""
import getpass,json,datetime,shlex,hashlib
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
def main():
 Q=json.loads((R/'ACTIVE_ROOT.json').read_text())['root'];out=R/('snapshot_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir();c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  code='''
import os,json,hashlib,subprocess,time
from pathlib import Path
Q=Path(%r)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
contract=json.loads((Q/'CONTRACT.json').read_text());bad=[]
for p,h in contract['files'].items():
 d=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8388608),b''):d.update(b)
 if d.hexdigest()!=h:bad.append(p)
roots={'shared':Q};roots.update({a:Path(p) for a,p in json.loads((Q/'ENTRY.json').read_text())['arms'].items()});files={}
for prefix,root in roots.items():
 for p in root.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.jsonl','.exit') and 'private' not in p.parts:
   assert p.stat().st_size<8*1024**2;files[prefix+'/'+p.relative_to(root).as_posix()]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
own=[]
for p in Path('/proc').iterdir():
 if p.name.isdigit():
  try:
   if ('EM_ROOT='+str(Q)).encode() in (p/'environ').read_bytes().split(b'\\0'):own.append(dict(pid=int(p.name),cmd=(p/'comm').read_text().strip()))
  except OSError:pass
report=dict(unix=time.time(),hash_bad=bad,contract_sha256=sha(Q/'CONTRACT.json'),frozen_files=len(contract['files']),status=json.loads((Q/'RUN_STATUS.json').read_text()) if (Q/'RUN_STATUS.json').exists() else None,controller_exit=(Q/'controller.exit').read_text().strip() if (Q/'controller.exit').exists() else None,own_processes=own,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),disk={m:dict(root_free=os.statvfs(m).f_bfree*os.statvfs(m).f_frsize,ordinary_free=os.statvfs(m).f_bavail*os.statvfs(m).f_frsize) for m in ('/data','/')},phase_exits={p.name:p.read_text().strip() for p in (Q/'phases').glob('*.exit')},logs={p.name:p.read_text(errors='replace')[-5000:] for p in [Q/'controller.log',*list((Q/'phases').glob('*.log'))] if p.exists()},files=files)
print(json.dumps(report))
'''%Q
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(code),timeout=600);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err;d=json.loads(raw);(out/'EVIDENCE.json').write_bytes(raw);sftp=c.open_sftp();download={}
  for name,md in d['files'].items():
   assert '..' not in name and '\\' not in name;p=out/'metadata'/name;p.parent.mkdir(parents=True,exist_ok=True);sftp.get(md['path'],str(p));download[name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,at_snapshot=md)
  (out/'DOWNLOAD.json').write_text(json.dumps(download,indent=2));print(out);print(json.dumps({k:d[k] for k in ('status','controller_exit','hash_bad','gpu','phase_exits','own_processes')},indent=2))
  if d['controller_exit']=='1':print(json.dumps(d['logs'],indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
