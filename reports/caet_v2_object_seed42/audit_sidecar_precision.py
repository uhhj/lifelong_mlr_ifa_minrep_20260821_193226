"""Pinned SSH fixed read-only scientific feasibility gate; local evidence only."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
script=(R/'audit_sidecar_precision_remote.py').read_text();protocol=(R/'PROTOCOL.md').read_bytes()
p=R/('sidecar_precision_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir()
request={'protocol_sha256':hashlib.sha256(protocol).hexdigest()}
(p/'PRECOMMIT.json').write_text(json.dumps(dict(request,script_sha256=hashlib.sha256(script.encode()).hexdigest(),time=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 stdin,out,err=c.exec_command("env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /data/mlr_ifa/env/bin/python -c "+shlex.quote(script),timeout=600)
 stdin.write(json.dumps(request));stdin.flush();stdin.channel.shutdown_write()
 raw=out.read();error=err.read();rc=out.channel.recv_exit_status()
 (p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=error.decode(),optimizer_calls=0,environment_calls=0,server_writes=0),indent=2)+'\n')
 assert rc==0,error.decode();d=json.loads(raw)
 (p/'EVIDENCE.json').write_bytes(raw);(p/'RECEIPT.json').write_text(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),protocol_sha256=request['protocol_sha256']),indent=2)+'\n')
 print(json.dumps(dict(directory=str(p),result=d)))
finally:pw=None;c.close()
