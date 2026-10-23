"""One read-only snapshot + whitelist metadata download; never restarts a worker."""
import getpass,json,datetime,shlex,hashlib
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
Q=json.loads((R/'ACTIVE_ROOT.json').read_text())['root'] if (R/'ACTIVE_ROOT.json').exists() else '/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023'
def main():
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 out=R/('snapshot_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  code='''
import json,hashlib,os,subprocess,time
from pathlib import Path
Q=Path(%r);O=Q/'output'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
contract=json.loads((Q/'CONTRACT.json').read_text());bad=[p for p,h in contract['files'].items() if sha(p)!=h]
OLD=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023_engineering03');old=json.loads((OLD/'CONTRACT.json').read_text());bad += [p for p,h in old['files'].items() if sha(p)!=h]
if sha(OLD/'CONTRACT.json')!=contract['old_contract_sha256']:bad.append('old_contract')
V2=Path('/data/mlr_ifa/experiment_workspaces/READOUT_PATCH_OBJECT41_20261023_v2');vc=json.loads((V2/'CONTRACT.json').read_text());bad += [p for p,h in vc['files'].items() if sha(p)!=h]
if sha(V2/'CONTRACT.json')!=contract['v2_contract_sha256']:bad.append('v2_contract')
BUDGET=Path('/data/mlr_ifa/experiment_workspaces/READOUT_BUDGET_OBJECT41_20261023_v1');bc=json.loads((BUDGET/'CONTRACT.json').read_text());bad += [p for p,h in bc['files'].items() if sha(p)!=h]
if sha(BUDGET/'CONTRACT.json')!=contract['budget_contract_sha256']:bad.append('budget_contract')
files={}
for p in O.iterdir():
 if p.is_file() and p.suffix in ('.json','.jsonl'):
  assert p.stat().st_size<8*1024**2;files[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
for p in (O/'cases').glob('*.json'):
 assert p.stat().st_size<8*1024**2;files['cases/'+p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
ev=O/'EVENTS.jsonl';lines=ev.read_text().splitlines() if ev.exists() else []
pid=int((Q/'controller.pid').read_text());pp=Path('/proc')/str(pid)
r=dict(status=json.loads((O/'RUN_STATUS.json').read_text()) if (O/'RUN_STATUS.json').exists() else None,unix=time.time(),hash_bad=bad,files=files,events_last=[json.loads(x) for x in lines[-3:]],controller_pid=pid,controller_alive=pp.exists(),controller_exit=(Q/'controller.exit').read_text().strip() if (Q/'controller.exit').exists() else None,gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),disk={m:dict(ordinary=(s:=os.statvfs(m)).f_bavail*s.f_frsize,root=s.f_bfree*s.f_frsize) for m in ('/data','/')},processes=subprocess.check_output(['ps','-eo','pid,ppid,comm'],text=True))
r['phase_logs']={p.name:p.read_text(errors='replace')[-6000:] for p in Q.glob('*.log')}
r['phase_exits']={p.name:p.read_text().strip() for p in Q.glob('*.exit')}
print(json.dumps(r))
'''%Q
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(code),timeout=240);i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err
  (out/'EVIDENCE.json').write_bytes(raw);d=json.loads(raw);sftp=c.open_sftp();dst=out/'metadata';dst.mkdir()
  # Mutable progress files may advance during download; preserve snapshot, record mismatch rather than claim exact.
  download={}
  for name,md in d['files'].items():
   assert '\\' not in name and '..' not in name and (len(Path(name).parts)==1 or name.startswith('cases/') and len(Path(name).parts)==2)
   p=dst/name;p.parent.mkdir(parents=True,exist_ok=True);sftp.get(Q+'/output/'+name,str(p));download[name]=dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,at_snapshot=md)
  (out/'DOWNLOAD.json').write_text(json.dumps(download,indent=2));print(out);print(json.dumps({k:d[k] for k in ('status','events_last','hash_bad','controller_exit','gpu','phase_exits')},indent=2))
  if 'FAILURE.json' in d['files']:print(d['phase_logs'])
 finally:pw=None;c.close()
if __name__=='__main__':main()
