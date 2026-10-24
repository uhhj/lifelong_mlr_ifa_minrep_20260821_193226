import datetime,getpass,hashlib,json,posixpath
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent;A=json.loads((R/'ACTIVE_ROOT.json').read_text());out=R/('final_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 script="Q="+repr(A['root'])+'\n'+(R/'final_readonly.py').read_text()
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -',timeout=1800);i.write(script);i.flush();i.channel.shutdown_write();raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status();(out/'ATTEMPT.json').write_text(json.dumps({'exit':rc,'stderr':err,'script_sha256':hashlib.sha256(script.encode()).hexdigest()},indent=2));assert rc==0,err
 result=json.loads(raw);(out/'REMOTE_FINAL_AUDIT.json').write_text(json.dumps(result,indent=2))
 sf=c.open_sftp()
 for n,v in result['light_files'].items():
  assert '..' not in Path(n).parts and not n.startswith('/') and Path(n).suffix in ['.json','.jsonl','.exit'] and v['bytes']<8*1024**2
  p=out/n;p.parent.mkdir(parents=True,exist_ok=True);sf.get(A['root']+'/'+n,str(p));assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
 sf.close();print(out)
finally:pw=None;c.close()
