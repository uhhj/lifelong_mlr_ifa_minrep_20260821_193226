"""One TACR/TACRST entry; no retries and no canceled experiments."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
W=Path(sys.argv[1]).resolve();e=json.loads((W/'ENTRY.json').read_text());s=e['seed'];assert s in (41,42,43) and e['variant'] in ('TACR','TACRST')
O=Path(e['output']);S=W/'source';PY='/data/mlr_ifa/env/bin/python'
env=dict(os.environ,CORESET_SOURCE=str(S),CORESET_OUT=str(O),CORESET_SEED=str(s),RA_BASE=e['parent'],PYTHONPATH=str(S),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4',MUJOCO_GL='egl',EGL_PLATFORM='surfaceless')
env.update(LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',HUGGINGFACE_HUB_CACHE='/data/mlr_ifa/cache/huggingface/hub',TRANSFORMERS_CACHE='/data/mlr_ifa/cache/huggingface/hub',TORCH_HOME='/data/mlr_ifa/cache/torch',XDG_CACHE_HOME='/data/mlr_ifa/cache',TMPDIR='/data/mlr_ifa/tmp',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',HYDRA_FULL_ERROR='1',MPLBACKEND='Agg')
def phase(name,args):
 d=W/'admission';d.mkdir(exist_ok=True);assert not (d/(name+'.exit')).exists()
 (d/(name+'.command.json')).write_text(json.dumps(dict(argv=args,unix=time.time()))+'\n')
 with (d/(name+'.log')).open('x') as f:
  p=subprocess.Popen(args,env=env,cwd=S,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
  (d/(name+'.pid.json')).write_text(json.dumps(dict(pid=p.pid,unix=time.time()))+'\n');code=p.wait()
 (d/(name+'.exit')).write_text(str(code)+'\n');assert code==0,(name,code)
def main():
 assert not O.exists() and not (W/'supervisor.pid.json').exists()
 (W/'supervisor.pid.json').write_text(json.dumps(dict(pid=os.getpid(),unix=time.time()))+'\n');rc=1
 try:
  phase('prepare',[PY,'-m','caet_v3_exp.prepare']);phase('labels6',[PY,'-m','caet_v3_exp.labels','6'])
  phase('gpu_preflight',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m','caet_v3_exp.preflight'])
  assert json.loads((O/'GPU_PREFLIGHT.json').read_text())['status']=='PASS'
  phase('pipeline',[PY,'-u','-m','caet_v3_exp.pipeline']);phase('final_process',[PY,'-m','caet_v3_exp.final_process']);rc=0
 except Exception:
  (W/'FAILURE.json').write_text(json.dumps(dict(status='BLOCKED_ENGINEERING_NO_RETRY',traceback=traceback.format_exc(),unix=time.time()))+'\n');raise
 finally:
  (W/'supervisor.exit').write_text(str(rc)+'\n')
  if O.exists():(O/'supervisor.exit').write_text(str(rc)+'\n')
if __name__=='__main__':main()
