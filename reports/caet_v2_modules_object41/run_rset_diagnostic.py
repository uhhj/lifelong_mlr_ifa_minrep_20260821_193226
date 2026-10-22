"""Run diagnostic on existing assets only; preserve every local attempt."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
def main():
 prefix=(R/'gate1_remote.py').read_text(encoding='utf8').split('for stage in (7,8,9):')[0]
 script=prefix+'\n'+(R/'diagnose_rset_remote.py').read_text(encoding='utf8')
 evidence=(R/'gate1_RSET/EVIDENCE.json').read_bytes();g=json.loads(evidence)
 request=dict(seed=41,variant='RSET',protocol_sha256=g['protocol_sha256'],diagnostic_protocol_sha256=hashlib.sha256((R/'RSET_DIAGNOSTIC_PROTOCOL.md').read_bytes()).hexdigest(),input_evidence_sha256=hashlib.sha256(evidence).hexdigest(),stages={k:v for k,v in g['stages'].items() if k in ('8','9')})
 p=R/('rset_diagnostic_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir()
 (p/'PRECOMMIT.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(script.encode()).hexdigest(),protocol_sha256=request['diagnostic_protocol_sha256'],input_evidence_sha256=request['input_evidence_sha256'],time=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  env="env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LIBERO_CONFIG_PATH=/data/mlr_ifa/config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data/mlr_ifa/cache/huggingface TRANSFORMERS_CACHE=/data/mlr_ifa/cache/huggingface/hub "
  i,o,e=c.exec_command(env+'/data/mlr_ifa/env/bin/python -c '+shlex.quote(script),timeout=1800);i.write(json.dumps(request));i.flush();i.channel.shutdown_write()
  raw=o.read();err=e.read();rc=o.channel.recv_exit_status();(p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode(),optimizer_calls=0,environment_calls=0),indent=2)+'\n');assert rc==0,err.decode()
  (p/'EVIDENCE.json').write_bytes(raw);d=json.loads(raw);(p/'RECEIPT.json').write_text(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),server_writes=0),indent=2)+'\n')
  print(json.dumps(dict(directory=str(p),policy_status=d['policy_status'],error=d.get('policy_error'),summaries={k:v['summary']['effect_supported_train'] for k,v in d['stages'].items()}),indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
