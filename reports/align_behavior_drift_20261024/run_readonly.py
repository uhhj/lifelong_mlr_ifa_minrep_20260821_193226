"""Reviewed program on remote stdin; credentials in getpass memory only."""
import datetime,getpass,hashlib,json,sys
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
script=R/sys.argv[1];out=R/(script.stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
pw=getpass.getpass('SSH password (memory only): ');count=0
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 prefix='SPEC='+repr(json.loads((R/'INPUT_SPEC.json').read_text()))+'\n'
 i,o,e=c.exec_command('/data/mlr_ifa/env/bin/python -B -',timeout=1200)
 i.write(prefix+script.read_text(encoding='utf8'));i.flush();i.channel.shutdown_write()
 for line in o:
  with (out/'STREAM.jsonl').open('a',encoding='utf8') as f:f.write(line)
  d=json.loads(line);name=d['kind']+('_'+str(d['seed']) if 'seed' in d else '')
  assert name.replace('_','').isalnum();(out/(name+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8');count+=1
  print(json.dumps({'file':name,'status':d.get('status'),'windows':d.get('windows'),'forward_calls':d.get('forward_calls')},ensure_ascii=False),flush=True)
 err=e.read().decode();rc=o.channel.recv_exit_status()
 (out/'ATTEMPT.json').write_text(json.dumps({'exit':rc,'stderr':err,'script_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'messages':count},indent=2),encoding='utf8')
 assert rc==0,err
 print(out,flush=True)
except Exception as ex:
 (out/'FAILURE.json').write_text(json.dumps({'exception':type(ex).__name__,'message':str(ex),'messages':count},indent=2),encoding='utf8');raise
finally:pw=None;c.close()
