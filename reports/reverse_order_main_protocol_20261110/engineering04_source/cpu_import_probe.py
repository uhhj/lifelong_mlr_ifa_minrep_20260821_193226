"""Read-only backend import probe; no model, environment or parameter update."""
import os,sys,json,time
os.environ.update(CUDA_VISIBLE_DEVICES='0',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HOME='/data/mlr_ifa/cache/huggingface',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
Q='/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering02'
sys.path[:0]=[Q+'/cases/OBJECT_MLR_41/source',Q]
os.environ.update(CORESET_OUT='/data/mlr_ifa/experiments/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering02/OBJECT_MLR_41/mlr_run',CORESET_SEED='41')
import torch
from libero.lifelong.algos.mlr import build_ifa_similarity_matrix,_sample
from order import install
install()
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_CPU_IMPORT_AND_ORDER_INSTALL_NO_CUDA_TENSOR_INITIALIZATION',observed=time.time(),backend='inherited_egl_device0_import_only',new_model_forward=0,new_optimizer=0,new_environment=0,server_writes=0)))
