"""Finite CAET-v2 gate2→gate3 sequence. No automatic scientific retries."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
W=Path(sys.argv[1]).resolve();entry=json.loads((W/'ENTRY.json').read_text());SEED=int(entry['seed']);assert SEED==41;O=Path(entry['output']);S=W/'source';PY='/data/mlr_ifa/env/bin/python'
env=dict(os.environ,CORESET_SOURCE=str(S),CORESET_OUT=str(O),CORESET_SEED=str(SEED),RA_BASE=entry['parent'],PYTHONPATH=str(S),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4',MUJOCO_GL='egl',EGL_PLATFORM='surfaceless')
env.update(LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',HUGGINGFACE_HUB_CACHE='/data/mlr_ifa/cache/huggingface/hub',TRANSFORMERS_CACHE='/data/mlr_ifa/cache/huggingface/hub',TORCH_HOME='/data/mlr_ifa/cache/torch',XDG_CACHE_HOME='/data/mlr_ifa/cache',TMPDIR='/data/mlr_ifa/tmp',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',HYDRA_FULL_ERROR='1',MPLBACKEND='Agg')

def phase(name,args):
 d=W/'admission';d.mkdir(exist_ok=True);assert not (d/(name+'.exit')).exists()
 (d/(name+'.command.json')).write_text(json.dumps(dict(argv=args,time=time.time()))+'\n')
 with (d/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(args,env=env,cwd=S,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
  (d/(name+'.pid.json')).write_text(json.dumps(dict(pid=proc.pid,time=time.time()))+'\n');rc=proc.wait()
 (d/(name+'.exit')).write_text(str(rc)+'\n');assert rc==0,(name,rc)

def main():
 assert not O.exists() and not (W/'supervisor.pid.json').exists()
 (W/'supervisor.pid.json').write_text(json.dumps(dict(pid=os.getpid(),time=time.time()))+'\n');rc=1
 try:
  phase('prepare',[PY,'-m','caet_v2_exp.prepare'])
  phase('labels6',[PY,'-m','caet_v2_exp.labels','6'])
  phase('gpu_preflight',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m','caet_v2_exp.preflight'])
  assert json.loads((O/'GPU_PREFLIGHT.json').read_text())['status']=='PASS'
  phase('variant_gpu_preflight',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m','caet_v2_exp.variant_preflight'])
  assert json.loads((O/'VARIANT_GPU_PREFLIGHT.json').read_text())['status']=='PASS'
  (O/'LAUNCH_ACCEPTANCE.json').write_text(json.dumps(dict(status='PASS',gate1='PASS',gate2='PASS',time=time.time()))+'\n')
  phase('pipeline',[PY,'-u','-m','caet_v2_exp.pipeline'])
  phase('final_process',[PY,'-m','caet_v2_exp.final_process']);rc=0
 except Exception:
  (W/'FAILURE.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),time=time.time(),status='BLOCKED_ENGINEERING',retry=False))+'\n')
  raise
 finally:
  (W/'supervisor.exit').write_text(str(rc)+'\n')
  if O.exists():(O/'supervisor.exit').write_text(str(rc)+'\n')

if __name__=='__main__':main()
