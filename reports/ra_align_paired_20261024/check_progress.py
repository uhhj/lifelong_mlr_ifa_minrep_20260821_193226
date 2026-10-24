"""Only lightweight read-only remote state, never private histories."""
import datetime,getpass,hashlib,json,sys
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent;active=json.loads((R/'ACTIVE_ROOT.json').read_text());Q=active['root']
out=R/('snapshot_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 code="""import os,json,hashlib,subprocess,time
from pathlib import Path
q=Path(%r)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
ct=json.loads((q/'CONTRACT.json').read_text())
assert all(sha(q/n)==h for n,h in ct['files'].items())
files={}
for n in ['ADMISSION.json','RUN_STARTED.json','OUTCOMES.json','COMPLETE.json','FAILURE.json','EVENTS.jsonl','worker.exit','supervisor.exit','worker.log','supervisor.log']:
 p=q/n
 if p.exists():files[n]=p.read_text()[-50000:] if p.suffix=='.log' else p.read_text()
procs=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
print(json.dumps({'time':time.time(),'root':str(q),'contract_sha256':sha(q/'CONTRACT.json'),'source_unchanged':True,'files':files,'processes':[l for l in procs.splitlines() if str(q) in l],'gpu':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),'disk':{p:{'bavail':os.statvfs(p).f_bavail*os.statvfs(p).f_frsize,'bfree':os.statvfs(p).f_bfree*os.statvfs(p).f_frsize} for p in ['/data','/root']}}))
"""%Q
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -');i.write(code);i.flush();i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(out/'ATTEMPT.json').write_text(json.dumps({'exit':rc,'stderr':err},indent=2));assert rc==0,err
 d=json.loads(raw);assert d['contract_sha256']==active['contract_sha256'];(out/'SNAPSHOT.json').write_text(json.dumps(d,indent=2),encoding='utf8')
 for n,t in d['files'].items():
  if not n.endswith('.log'):(out/n).write_text(t,encoding='utf8')
 print(json.dumps({'snapshot':str(out),'time':d['time'],'files':list(d['files']),'gpu':d['gpu'],'processes':d['processes'],'last_events':d['files'].get('EVENTS.jsonl','').splitlines()[-2:],'failure':d['files'].get('FAILURE.json'),'exit':d['files'].get('worker.exit')}))
finally:pw=None;c.close()
