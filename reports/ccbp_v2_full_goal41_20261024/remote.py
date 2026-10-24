"""Explicit script runner; credentials remain only in getpass memory."""
import datetime,getpass,json,sys,hashlib
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
def run(name):
 out=R/(Path(name).stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
 pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  script=(R/name).read_text(encoding='utf8')
  if (R/'DEPLOY_TEXT_BUNDLE.json').exists() and name=='deploy_remote.py':script='BUNDLE='+ascii(json.loads((R/'DEPLOY_TEXT_BUNDLE.json').read_text(encoding='utf8')))+'\n'+script
  script='# -*- coding: utf-8 -*-\n'+script
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -',timeout=1800);i.write(script.encode('utf8'));i.flush();i.channel.shutdown_write()
  raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
  (out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err,script_sha256=hashlib.sha256(script.encode()).hexdigest()),indent=2));assert rc==0,err
  data=json.loads(raw);(out/'EVIDENCE.json').write_text(json.dumps(data,indent=2))
  if 'sources' in data:
   for n,s in data.pop('sources').items():
    p=R/'reference_source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s,encoding='utf8')
   (out/'EVIDENCE.json').write_text(json.dumps(data,indent=2))
  print(out);print(json.dumps({k:v for k,v in data.items() if k not in ('light','log_tails','diagnostics','sources')})[:4000])
 finally:pw=None;c.close()
if __name__=='__main__':run(sys.argv[1])
