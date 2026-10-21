"""One fixed CPU read-only diagnostic over existing GOAL RA assets; pinned SSH."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
sources={'labels':R/'goal_preparation/source/caet_exp/labels.py','transfer':R/'caet_additions/caet_exp/transfer.py',
 'native':R/'source/libero/lifelong/algos/mlr.py','fit':R/'caet_additions/caet_exp/fit.py'}
request={k:p.read_text(encoding='utf8') for k,p in sources.items()}
request['previous']=json.loads((R/'goal_preparation/GOAL_EFFECT_FEASIBILITY.json').read_text())
script=(R/'audit_goal_pairs_remote.py').read_text(encoding='utf8')
out=R/('goal_pair_audit_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));out.mkdir()
client=paramiko.SSHClient();client.load_system_host_keys();client.set_missing_host_key_policy(paramiko.RejectPolicy())
pw=getpass.getpass('SSH password (memory only): ')
try:
 client.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
 command="env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/mlr_ifa/env/bin/python -c "+shlex.quote(script)
 stdin,stdout,stderr=client.exec_command(command,timeout=600)
 stdin.write(json.dumps(request));stdin.flush();stdin.channel.shutdown_write()
 raw=stdout.read();err=stderr.read();rc=stdout.channel.recv_exit_status()
 (out/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode(),script_sha256=hashlib.sha256(script.encode()).hexdigest(),
  optimizer_calls=0,environment_steps=0,server_writes=0),indent=2)+'\n')
 assert rc==0,err.decode()
 d=json.loads(raw);assert d['status']=='PASS' and not d['cuda_initialized']
 (out/'PAIR_EVIDENCE.json').write_bytes(raw)
 (out/'RECEIPT.json').write_text(json.dumps(dict(status='READ_ONLY_CPU_PASS',sha256=hashlib.sha256(raw).hexdigest(),
  bytes=len(raw),source_hashes=d['sources'],unix=d['finished_unix'],optimizer_calls=0,environment_steps=0,server_writes=0),indent=2)+'\n')
 print(json.dumps(dict(directory=str(out),status=d['status'],seconds=d['finished_unix']-d['started_unix'])))
finally:pw=None;client.close()
