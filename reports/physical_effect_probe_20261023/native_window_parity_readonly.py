"""CPU formal full-window path vs exported frame-batched features; no training or env."""
import os,sys,json,hashlib,ast
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/PHYSICAL_EFFECT_PROBE_OBJECT41_20261023_engineering03')
os.environ.update(PEP_ROOT=str(Q),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',LIBERO_CONFIG_PATH='/data/mlr_ifa/config',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HOME='/data/mlr_ifa/cache/huggingface',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1')
sys.path.insert(0,str(Q))
from physical_effect_probe.common import *
import h5py
import robomimic.utils.obs_utils as ObsUtils
from libero.lifelong.utils import get_task_embs,safe_device
import robomimic.utils.tensor_utils as TensorUtils
# Execute the unmodified preprocessing function only; importing metric also initializes EGL.
metric_source=(SOURCE/'libero/lifelong/metric.py').read_text();node=next(x for x in ast.parse(metric_source).body if isinstance(x,ast.FunctionDef) and x.name=='raw_obs_to_tensor_obs')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE/'libero/lifelong/metric.py'),'exec'),globals())
macro_tree=ast.parse(Path('/data/mlr_ifa/env/lib/python3.8/site-packages/robosuite/macros.py').read_text())
macro_convention=next(ast.literal_eval(n.value) for n in macro_tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='IMAGE_CONVENTION' for t in n.targets))
before=verify();p=full_policy('cpu');p.cfg.device='cpu';c=p.cfg
ObsUtils.initialize_obs_utils_with_obs_specs({'obs':c.data.obs.modality});proof=[]
for task in (6,7):
 b=js(PARENT/'bindings'/('task%d.json'%task));data=torch.load(OUT/'private'/('labels%d.pt'%task),map_location='cpu');i=next(i for i,r in enumerate(data['metadata']) if r['state_valid']);r=data['metadata'][i]
 with h5py.File(b['hdf5'],'r') as h:
  image_convention=str(h['data'].attrs['macros_image_convention']);assert image_convention==macro_convention
  lang=json.loads(h['data'].attrs['problem_info'])['language_instruction'];tok=get_task_embs(c,[lang]);g=h['data'][r['demo_id']];steps=[]
  for u in range(r['window_start'],r['window_start']+8):
   obs={c.data.obs_key_mapping[k]:g['obs'][k][u] for k in c.data.obs_key_mapping};steps.append(raw_obs_to_tensor_obs([obs],tok['input_ids'][0],tok['attention_mask'][0],c))
 x=dict(obs={k:torch.stack([d['obs'][k] for d in steps],dim=1) for k in steps[0]['obs']},task_emb=steps[0]['task_emb'],masks=steps[0]['masks'])
 with torch.no_grad():full=p.spatial_encode(p.preprocess_input(x,train_mode=False))
 exported=torch.load(OUT/'private'/('native%d.pt'%task),map_location='cpu')[i:i+1]
 delta=float((full-exported).abs().max());assert torch.allclose(full,exported,atol=1e-3,rtol=1e-4),(task,delta)
 proof.append(dict(task=task,sample=r['sample'],full_window_max_abs_difference=delta,tolerance_atol=1e-3,tolerance_rtol=1e-4,image_convention=image_convention,passes=True))
assert verify()==before and not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_NATIVE_FORMAL_WINDOW_PARITY',proof=proof,contract_sha256=before,strict_full_official_keys=len(p.state_dict()),cpu_full_window_forwards=2,cuda_initialized=False,optimizer_calls=0,environment_steps=0,server_writes=0)))
