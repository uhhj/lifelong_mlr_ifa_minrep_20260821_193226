"""Send reviewed read-only CPU program via stdin; retain only local scalar files."""
import datetime,getpass,hashlib,json,sys
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
script=R/'B0_action_diagnostics_readonly.py'
out=R/('B0_action_diagnostics_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
pw=getpass.getpass('SSH password (memory only): ');case_manifest={};stages=[];summary=None
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 prefix='import os; os.environ["EM_ROOT"]='+repr(json.loads((R/'ACTIVE_ROOT.json').read_text())['root'])+'\n'
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -',timeout=3600)
 i.write(prefix+script.read_text(encoding='utf8'));i.flush();i.channel.shutdown_write()
 for line in o:
  item=json.loads(line)
  if item['kind']=='case':
   name=item['name'];assert '/' not in name and '\\' not in name
   raw=json.dumps(item['rows'],separators=(',',':'),allow_nan=False).encode();assert len(raw)<8*1024**2
   (out/(name+'.json')).write_bytes(raw);case_manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),rows=len(item['rows']))
  elif item['kind']=='stage':
   value=item['evidence'];stages.append(value['stage']);(out/('STAGE%d.json'%value['stage'])).write_text(json.dumps(value,indent=2));print(json.dumps({k:v for k,v in value.items() if k!='parity'}),flush=True)
  elif item['kind']=='complete':summary=item
  else:(out/'START.json').write_text(json.dumps(item,indent=2));print(json.dumps(item),flush=True)
 err=e.read().decode();rc=o.channel.recv_exit_status()
 (out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err,script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),case_manifest=case_manifest),indent=2))
 assert rc==0 and summary and stages==[6,7,8,9],err
 summary['case_files']=case_manifest;summary['script_sha256']=hashlib.sha256(script.read_bytes()).hexdigest()
 (out/'EVIDENCE.json').write_text(json.dumps(summary,indent=2));print(str(out),flush=True);print(json.dumps({k:v for k,v in summary.items() if k not in ('costs','case_files')}),flush=True)
except Exception as ex:
 (out/'FAILURE.json').write_text(json.dumps(dict(exception=type(ex).__name__,message=str(ex),completed_stages=stages,case_manifest=case_manifest),indent=2));raise
finally:pw=None;c.close()
