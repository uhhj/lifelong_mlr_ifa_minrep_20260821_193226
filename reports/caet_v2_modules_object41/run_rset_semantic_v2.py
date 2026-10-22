"""Independent semantic diagnostic. No mutations to deployed training source."""
import datetime,getpass,hashlib,json,shlex
from pathlib import Path
import paramiko
R=Path(__file__).resolve().parent
def composed():
 prefix=(R/'gate1_remote.py').read_text(encoding='utf8').split('for stage in (7,8,9):')[0]
 old=(R/'diagnose_rset_remote.py').read_text(encoding='utf8')
 old=old.replace("grip=(a[:,None,6]-bank['a'][None,:,6]).abs()/scale[6]","grip=torch.zeros_like(arm)")
 old=old.replace("sq=state.square()+arm.square()+grip.square()","sq=state.square()+arm.square()")
 old=old.replace(".clamp_min(torch.tensor([.001]*6+[.05]))","[:6].clamp_min(.001)")
 old=old.replace(",gripper=quant(cal,'gripper',.99)","")
 old=old.replace("('d','state','arm','gripper')","('d','state','arm')")
 old=old.replace("gripper=float(val[:,3].square().mean().sqrt()),","gripper_class_matched=True,")
 split="result['calibrations']=";head,tail=old.split(split)
 return prefix+'\n'+head+'\n'+(R/'rset_semantic_checks.py').read_text(encoding='utf8')+'\n'+split+tail
def main():
 script=composed();source=(R/'gripper_semantics_20261022_125239/SOURCE_EVIDENCE.json').read_bytes();sd=json.loads(source);evidence=(R/'gate1_RSET/EVIDENCE.json').read_bytes();g=json.loads(evidence)
 request=dict(seed=41,variant='RSET',protocol_sha256=g['protocol_sha256'],diagnostic_protocol_sha256=hashlib.sha256((R/'RSET_SEMANTIC_V2_PROTOCOL.md').read_bytes()).hexdigest(),input_evidence_sha256=hashlib.sha256(evidence).hexdigest(),semantic_source_sha256=hashlib.sha256(source).hexdigest(),stages={k:v for k,v in g['stages'].items() if k in ('8','9')})
 # Bind semantic source paths again in the actual remote diagnostic, before imports.
 check="\nassert all(sha(p)==h for p,h in "+repr({k:v['sha256'] for k,v in sd['files'].items()})+".items()), 'Semantic source changed'\n"
 script=script.replace("print('Verified data loaded; diagnosing support',file=sys.stderr,flush=True)",check+"print('Verified data loaded; diagnosing support',file=sys.stderr,flush=True)")
 p=R/('rset_semantic_v2_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'));p.mkdir();(p/'PRECOMMIT.json').write_text(json.dumps(dict(script_sha256=hashlib.sha256(script.encode()).hexdigest(),protocol_sha256=request['diagnostic_protocol_sha256'],semantic_source_sha256=request['semantic_source_sha256'],input_evidence_sha256=request['input_evidence_sha256'],time=datetime.datetime.now().astimezone().isoformat()),indent=2)+'\n')
 c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.RejectPolicy());pw=getpass.getpass('SSH password (memory only): ')
 try:
  c.connect('jq1.9gpu.com',port=11920,username='root',password=pw,allow_agent=False,look_for_keys=False,timeout=20);pw=None
  env="env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LIBERO_CONFIG_PATH=/data/mlr_ifa/config MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=surfaceless HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data/mlr_ifa/cache/huggingface TRANSFORMERS_CACHE=/data/mlr_ifa/cache/huggingface/hub "
  i,o,e=c.exec_command(env+'/data/mlr_ifa/env/bin/python -c '+shlex.quote(script),timeout=1800);i.write(json.dumps(request));i.flush();i.channel.shutdown_write();raw=o.read();err=e.read();rc=o.channel.recv_exit_status();(p/'ATTEMPT.json').write_text(json.dumps(dict(exit=rc,stderr=err.decode(),optimizer_calls=0,environment_calls=0),indent=2)+'\n');assert rc==0,err.decode();(p/'EVIDENCE.json').write_bytes(raw);d=json.loads(raw);(p/'RECEIPT.json').write_text(json.dumps(dict(sha256=hashlib.sha256(raw).hexdigest(),server_writes=0),indent=2)+'\n')
  print(json.dumps(dict(directory=str(p),policy_status=d['policy_status'],summaries={k:v['summary']['effect_supported_train'] for k,v in d['stages'].items()},predictors={str(x['task']):{k:v['accepted'] for k,v in x['phases'].items()} for x in d['predictors']}),indent=2))
 finally:pw=None;c.close()
if __name__=='__main__':main()
