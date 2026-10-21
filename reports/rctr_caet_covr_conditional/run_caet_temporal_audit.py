"""Pinned SSH, fixed read-only temporal audit, credentials in getpass memory only."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
script=(R/'audit_caet_temporal_remote.py').read_text()
p=R/('caet_temporal_audit_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy())
pw=getpass.getpass('SSH password (memory only): ')
try:
 c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,look_for_keys=False,allow_agent=False,timeout=20);pw=None
 _,out,err=c.exec_command("env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/mlr_ifa/env/bin/python -c "+shlex.quote(script),timeout=600)
 raw=out.read();error=err.read();rc=out.channel.recv_exit_status()
 (p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=error.decode(),script_sha256=hashlib.sha256(script.encode()).hexdigest(),optimizer_calls=0,environment_calls=0),indent=2)+'\n')
 assert rc==0,error.decode()
 data=raw.split(b'TEMPORAL_AUDIT_JSON=')[-1];d=json.loads(data)
 (p/'EVIDENCE.json').write_bytes(data)
 (p/'RECEIPT.json').write_text(json.dumps(dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),script_sha256=hashlib.sha256(script.encode()).hexdigest()),indent=2)+'\n')
 print(json.dumps(dict(directory=str(p),status=d['status'],conclusion=d['conclusion'],seconds=d['finished_unix']-d['started_unix']),ensure_ascii=True))
finally:pw=None;c.close()
