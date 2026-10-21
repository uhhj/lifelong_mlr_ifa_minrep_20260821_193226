"""Fixed read-only SSH snapshot, metadata only; never a resume command."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
SCRIPT=r'''
from pathlib import Path
import json
p=Path('/data/mlr_ifa/activate.sh')
print(json.dumps(dict(path=str(p),text=p.read_text())))
'''

def main():
 p=R/('environment_inspection_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir()
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
  i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(SCRIPT),timeout=180);raw=o.read();err=e.read();rc=o.channel.recv_exit_status()
  (p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode()),indent=2)+'\n',encoding='utf8');assert rc==0,err.decode()
  d=json.loads(raw);(p/'SNAPSHOT.json').write_bytes(raw);(p/'RECEIPT.json').write_text(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest()),indent=2)+'\n')
  print(raw.decode())
 finally:pw=None;c.close()

if __name__=='__main__':main()

