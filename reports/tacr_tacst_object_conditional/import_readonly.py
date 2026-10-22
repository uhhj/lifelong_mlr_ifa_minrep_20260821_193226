"""Import every active module in all six own entry environments, CPU only."""
import os,json,subprocess,time,hashlib
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023')
code="""
import importlib,json,torch,os
names=['variant','model','observe','teacher_keys','gradient_audit','dataset','labels','train','preflight','evaluate','audit_mechanism','audit','prepare','pipeline','final_process']
for n in names:importlib.import_module('caet_v3_exp.'+n)
from caet_v3_exp.variant import VARIANT
from coreset_exp.common import SEED,O
assert not torch.cuda.is_initialized() and not O.exists()
print(json.dumps(dict(variant=VARIANT,seed=SEED,modules=len(names),cuda_initialized=False,output_exists=False)))
"""
result=[]
for e in json.loads((Q/'QUEUE.json').read_text()):
 S=Q/e['key']/'source';env=dict(os.environ,PYTHONPATH=str(S),CORESET_SOURCE=str(S),CORESET_OUT=e['output'],CORESET_SEED=str(e['seed']),RA_BASE=e['parent'],CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/mlr_ifa/cache/huggingface',MPLBACKEND='Agg',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='1')
 p=subprocess.run(['/data/mlr_ifa/env/bin/python','-c',code],cwd=S,env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=100)
 assert p.returncode==0,(e['key'],p.stderr)
 lines=p.stdout.splitlines();v=json.loads(next(s for s in reversed(lines) if s.startswith('{')));assert v['variant']==e['variant'] and v['seed']==e['seed'];result.append(dict(key=e['key'],proof=v))
c=json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in c['files'].items())
print(json.dumps(dict(status='ALL_SIX_CPU_IMPORT_ENVIRONMENTS_PASS_NOT_GPU_ADMISSION',entries=result,frozen_files=len(c['files']),optimizer_calls=0,model_forwards=0,environment_steps=0,server_writes=0,unix=time.time())))
