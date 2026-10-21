"""Versioned launcher-only repair; frozen source/contract and failed exit untouched."""
import json,os,time,traceback,hashlib
from pathlib import Path
from supervisor import W,O,S,PY,env,phase

def main():
 assert (W/'supervisor.exit').read_text().strip()=='1'
 assert (W/'admission/prepare.exit').read_text().strip()=='0'
 assert (W/'admission/labels6.exit').read_text().strip()=='0'
 assert 'EOFError' in (W/'admission/gpu_preflight.log').read_text()
 assert not (O/'preflight/optimizer_calls.json').exists() and not (O/'ra_coreset/epochs.json').exists()
 for p in Path('/proc').glob('[0-9]*'):
  try:
   if int(p.name) in (os.getpid(),os.getppid()):continue
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   assert not ((p/'cwd').resolve()==S and any(k in cmd for k in ('caet_v2_exp','torch.distributed.run','coreset_exp.evaluate'))),cmd
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 env.update(LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',HUGGINGFACE_HUB_CACHE='/data/mlr_ifa/cache/huggingface/hub',TRANSFORMERS_CACHE='/data/mlr_ifa/cache/huggingface/hub',TORCH_HOME='/data/mlr_ifa/cache/torch',XDG_CACHE_HOME='/data/mlr_ifa/cache',TMPDIR='/data/mlr_ifa/tmp',PYOPENGL_PLATFORM='egl',MUJOCO_EGL_DEVICE_ID='0',TOKENIZERS_PARALLELISM='false',WANDB_MODE='disabled',HYDRA_FULL_ERROR='1',MPLBACKEND='Agg')
 assert Path(env['LIBERO_CONFIG_PATH'],'config_adaptation.yaml').is_file()
 contract=hashlib.sha256((O/'CONTRACT.json').read_bytes()).hexdigest()
 receipt=dict(pid=os.getpid(),time=time.time(),reason='Missing inherited LIBERO/HF runtime paths; failed before any model or optimizer update',source_changed=False,contract_sha256=contract,prior_optimizer_calls=0,prior_formal_steps=0,prior_environment_calls=0,environment_additions={k:env[k] for k in ('LIBERO_CONFIG_PATH','HF_HOME','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','TORCH_HOME','TMPDIR','PYOPENGL_PLATFORM','MUJOCO_EGL_DEVICE_ID')})
 (W/'RECOVERY_ENVIRONMENT01.json').write_text(json.dumps(receipt,indent=2)+'\n');rc=1
 try:
  phase('gpu_preflight_environment01',[PY,'-m','torch.distributed.run','--standalone','--nproc_per_node=1','-m','caet_v2_exp.preflight'])
  assert json.loads((O/'GPU_PREFLIGHT.json').read_text())['status']=='PASS'
  (O/'LAUNCH_ACCEPTANCE.json').write_text(json.dumps(dict(status='PASS',gate1='PASS',gate2='PASS',time=time.time(),recovery='RECOVERY_ENVIRONMENT01',contract_sha256=contract,failed_import_attempt_preserved=True))+'\n')
  phase('pipeline',[PY,'-u','-m','caet_v2_exp.pipeline'])
  phase('final_process',[PY,'-m','caet_v2_exp.final_process']);rc=0
 except Exception:
  (W/'RECOVERY_ENVIRONMENT01_FAILURE.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),time=time.time()))+'\n');raise
 finally:
  (W/'recovery_environment01.exit').write_text(str(rc)+'\n')
  assert hashlib.sha256((O/'CONTRACT.json').read_bytes()).hexdigest()==contract

if __name__=='__main__':main()
