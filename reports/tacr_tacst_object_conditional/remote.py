"""Run reviewed task-specific remote script; credential stays in getpass memory."""
import sys,getpass,json,datetime,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
script=R/sys.argv[1];out=R/(script.stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -c '+shlex.quote(script.read_text(encoding='utf8')),timeout=600);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
 (out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err
 (out/'EVIDENCE.json').write_bytes(raw);print(out);print(raw.decode()[:24000])
finally:pw=None;c.close()
