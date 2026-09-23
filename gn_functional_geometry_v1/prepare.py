import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import json,subprocess,collections,time
import numpy as np
import torch,h5py
from .common import *
def main():
 assert not (OUT/'DATA_SPLIT.json').exists(),'Preserve prior selection'
 # Existing historical package resolution, without modifying any formal source.
 from cb_ifa_experiment import gpu_runtime as env
 from libero.libero.benchmark import get_benchmark
 from libero.lifelong.models.brief import BriefPolicy
 from easydict import EasyDict
 cfgpath=ROOT/'runs/fixed_kappa_confirmation_v2/formal/seed45/mlr/run_001/config.json'
 cfg=read(cfgpath);checkpoint=cfgpath.parent/'mlrifa_8_model_ep99.pth'
 cfg.update(aug=False,device='cuda:0',pretrain=str(checkpoint))
 cfg=EasyDict(cfg);policy=BriefPolicy(cfg,cfg.shape_meta)
 saved=torch.load(checkpoint,map_location='cpu')['state_dict']
 saved={k.removeprefix('module.') if hasattr(k,'removeprefix') else (k[7:] if k.startswith('module.') else k):v for k,v in saved.items()}
 policy.load_state_dict(saved,strict=True)
 assert policy.policy_head.low_eval_noise is False
 planpath=ROOT/'runs/fixed_kappa_confirmation_v2/formal/seed45/mlr/plans/task9.json'
 plan=read(planpath);bench=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
 contexts={};rng=np.random.RandomState(2026092301);selected={}
 for task in (8,9):
  hdf=ROOT/'data'/bench.get_task_demonstration(task)
  raw={};index={}
  with h5py.File(hdf,'r') as f:
   for demo in sorted(f['data'],key=lambda x:int(x.split('_')[-1]))[:10]:
    a=np.asarray(f['data'][demo]['actions'],dtype=np.float32);raw[demo]=a
    for t in range(len(a)):
     arr=a[np.minimum(np.arange(t,t+8),len(a)-1)]
     index.setdefault(arr.tobytes(),[]).append((demo,t))
  pools=collections.defaultdict(list);ambiguous=0
  for record,path in enumerate(plan['immutable_feature_origins']):
   if int(Path(path).parent.name)!=task:continue
   z=torch.load(ROOT/path,map_location='cpu');a=z['actions'].numpy().reshape(8,7).astype(np.float32)
   matches=index.get(a.tobytes(),[])
   if len(matches)!=1:ambiguous+=1;continue
   demo,t=matches[0]
   pools[demo].append(dict(task_id=task,trajectory_id=demo,window_index=t,window_length=8,
    trajectory_length=len(raw[demo]),source_record_index=record,origin=path,sha256=sha(ROOT/path)))
  demos=sorted(pools,key=lambda x:int(x.split('_')[-1]));rng.shuffle(demos)
  assert len(demos)==10
  for d in demos:pools[d].sort(key=lambda r:r['window_index'])
  groups={'old_build':demos[:4],'old_gradient':demos[4:6],'old_probe':demos[6:]} if task==8 else {'new_gradient':demos[:5],'new_probe':demos[5:]}
  for name,ds in groups.items():
   rows=[];total=128 if name in ('old_build','old_probe') else 64
   for j,d in enumerate(ds):
    n=total//len(ds)+(j<total%len(ds));pool=pools[d];assert len(pool)>=n,(task,d,len(pool),n)
    if name=='old_build':
     starts=np.linspace(0,len(pool)-8,4).round().astype(int);ix=np.concatenate([np.arange(v,v+8) for v in starts]);assert len(set(ix))==32
    else:ix=np.linspace(0,len(pool)-1,n).round().astype(int)
    rows.extend(pool[int(i)] for i in ix)
   selected[name]=rows
  contexts[str(task)]=dict(hdf_path=str(hdf),hdf_sha256=sha(hdf),eligible_windows=sum(map(len,pools.values())),ambiguous_windows_excluded=ambiguous,trajectory_assignment=groups)
  print('SPLIT',task,{k:len(selected[k]) for k in groups},flush=True)
 # Native historical head step scale, recovered from stored AdamW moments only.
 # No optimizer.step or model update is executed.
 names=[n for n,p in policy.named_parameters()];norms=[]
 total=read(ROOT/'runs/fixed_kappa_confirmation_v2/formal/seed45/mlr/plans/task8.json')['batches_per_epoch']*100
 for path in sorted(Path('/root/functional_geometry_diagnostic_v1_scratch/seed45').glob('s45_t8_*.pt')):
  z=torch.load(path,map_location='cpu');o=z['optimizer'];pg=o['param_groups'][0];ids=pg['params'];assert len(ids)==len(names)
  t=int(z['scheduler_step']);lr=1e-4*(total-(t-1))/(total-500)
  assert abs(float(pg['lr'])-1e-4*(total-t)/(total-500))<1e-14
  sq=0.;quant=0.
  for n,id in zip(names,ids):
   if not n.startswith('policy_head.'):continue
   w=z['model']['module.'+n].double();st=o['state'][id];step=float(st['step']);beta1,beta2=pg['betas']
   term=(st['exp_avg'].double()/(1-beta1**step))/(torch.sqrt(st['exp_avg_sq'].double()/(1-beta2**step))+pg['eps'])
   # w_after=(1-lr*wd)*w_before-lr*term; no finite precision inverse equality claimed.
   d=-lr*(term+pg['weight_decay']*w)/(1-lr*pg['weight_decay'])
   sq+=float(d.square().sum());quant+=float((torch.finfo(torch.float32).eps*w.abs()).square().sum())
  norms.append(dict(path=str(path),sha256=sha(path),scheduler_step=t,previous_lr=lr,head_step_norm=sq**.5,float32_roundoff_scale=quant**.5))
 assert len(norms)==6
 write(OUT/'UPDATE_SCALE.json',dict(method='Algebraic reconstruction of the immediately preceding ordinary native AdamW head displacement from post-step moments, weights and exact previous LR. Pre-quantization estimate, not a stored bitwise displacement.',sources=norms,median_update_norm=float(np.median([x['head_step_norm'] for x in norms])),optimizer_steps_executed=0))
 write(OUT/'DATA_SPLIT.json',dict(plan=str(planpath),plan_sha256=sha(planpath),roles=selected,contexts=contexts,
  source='Only stage9 frozen plan immutable feature origins; old retained buffer membership preserved, no resampling of formal buffer',
  old_gradient_and_old_probe_disjoint=True,GN_build_and_probes_disjoint=True,new_gradient_and_probe_disjoint=True))
 provenance=[]
 for mod in ['libero.lifelong.models.brief','libero.lifelong.models.policy_head','libero.lifelong.models.modules.transformer_modules','libero.lifelong.models.base_policy']:
  m=__import__('importlib').import_module(mod);provenance.append(dict(module=mod,path=m.__file__,sha256=sha(m.__file__)))
 write(OUT/'EXPERIMENT_CONTEXT.json',dict(git_HEAD=subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),python=__import__('sys').version,
  torch=torch.__version__,cuda=torch.version.cuda,gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv,noheader']).decode().strip(),
  checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),checkpoint_role='Original MLR seed45 task8 ep99 completed, before any task9 update',
  completion_evidence=str(ROOT/'reports/fixed_kappa_confirmation_v2/COMPLETION.json'),old_task=8,new_task=9,seed=45,
  seed_role='development, one checkpoint',policy='BriefPolicy',policy_head='GMMHead 5 modes, learned scale; exact expectation, no action sampling',
  native_trainable_blocks=['temporal_transformer','policy_head'],diagnostic_parameter_block='policy_head',P=sum(p.numel() for p in policy.policy_head.parameters()),
  front_end=dict(use_language=True,use_state=False),model_strict_load=True,source_files=provenance,training_active=False,formal_solver_modified=False))
 print('PREPARED',flush=True)
if __name__=='__main__':main()
