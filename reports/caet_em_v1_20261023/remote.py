"""Run reviewed task-specific remote script; credential stays in getpass memory."""
import sys,getpass,json,datetime,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
script=R/sys.argv[1];out=R/(script.stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 prefix=''
 if (R/'ACTIVE_ROOT.json').exists():prefix='import os; os.environ["EM_ROOT"]='+repr(json.loads((R/'ACTIVE_ROOT.json').read_text())['root'])+'; os.environ["PYTHONDONTWRITEBYTECODE"]="1"\n'
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -c '+shlex.quote(prefix+script.read_text(encoding='utf8')),timeout=600);raw=o.read();err=e.read().decode();rc=o.channel.recv_exit_status()
 (out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err),indent=2));assert rc==0,err
 try:parsed=json.loads(raw)
 except json.JSONDecodeError:
  (out/'STDOUT.txt').write_bytes(raw);parsed=json.loads(raw.decode().splitlines()[-1])
 if 'cases' in parsed:
  case_manifest={}
  for name,value in parsed.pop('cases').items():
   assert '/' not in name and '\\' not in name
   data=json.dumps(value,separators=(',',':')).encode();(out/(name+'.json')).write_bytes(data)
   case_manifest[name]=dict(sha256=__import__('hashlib').sha256(data).hexdigest(),bytes=len(data))
  parsed['case_files']=case_manifest
 (out/'EVIDENCE.json').write_text(json.dumps(parsed,indent=2));print(out);print(json.dumps({k:v for k,v in parsed.items() if k not in ('files','sources','case_files')})[:8000])
finally:pw=None;c.close()
