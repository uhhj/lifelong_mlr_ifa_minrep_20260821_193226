"""CPU reconstruction only: no optimizer.step, policy forward or environments."""
import os,sys,json,hashlib,random,math,contextlib,io
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='1'
from runtime import *
case=sys.argv[1];assert case in CASES;d=P/case
sys.path.insert(0,str(Q/'cases'/case/'source'))
import torch,numpy as np
torch.set_num_threads(1)
from easydict import EasyDict
from libero.lifelong.models.brief import BriefPolicy
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,np.ndarray):return np.array_equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
def finite(x):
 if torch.is_tensor(x):assert torch.isfinite(x).all()
 elif isinstance(x,dict):
  for v in x.values():finite(v)
 elif isinstance(x,(list,tuple)):
  for v in x:finite(v)
class Wrapped(torch.nn.Module):
 def __init__(self,c):super().__init__();self.module=BriefPolicy(c,None)
cfg=EasyDict(js(d/'resolved_random.json'));cfg.device='cpu'
with contextlib.redirect_stdout(io.StringIO()):model=Wrapped(cfg)
cp={}
for t in range(6,10):
 p=d/'random/models'/('mlrifa_%d_model_ep99.pth'%t);raw=torch.load(p,map_location='cpu');sd=raw['state_dict'];finite(sd)
 assert raw['contract_sha256']==sha(d/'CONTRACT.json') and raw['modal_sha256']==sha(d/'MODAL_CONTRACT.json')
 assert raw['task']==t and raw['epoch_index']==99 and len(sd)==len(model.state_dict())
 model.load_state_dict(sd,strict=True);assert equal(model.state_dict(),sd)
 cp[str(t)]=dict(sha256=sha(p),keys=len(sd),strict_load=True,finite=True)
r=torch.load(d/'random/recovery/latest.pth',map_location='cpu');s=r['state'];finite(s['optimizer']);finite(s['grads'])
assert r['task']==9 and r['next_epoch']==100 and r['stage_complete']
assert r['contract_sha256']==sha(d/'CONTRACT.json') and r['modal_sha256']==sha(d/'MODAL_CONTRACT.json')
assert equal(s['model'],model.state_dict()) and s['step']==js(d/'INPUT.json')['production_budget']
for n,p in model.named_parameters():p.requires_grad_(s['requires_grad'][n]);p.grad=None if s['grads'][n] is None else s['grads'][n].clone()
for n,m in model.named_modules():m.training=s['modes'][n]
optim=torch.optim.AdamW(model.parameters(),**dict(cfg.train.optimizer.kwargs));optim.load_state_dict(s['optimizer']);assert equal(optim.state_dict(),s['optimizer'])
total=js(d/'manifests/task9.json')['batches']*100
fn=lambda step:float(step)/500 if step<500 else max(0.,float(total-step)/max(1,total-500))
sched=torch.optim.lr_scheduler.LambdaLR(optim,lr_lambda=fn);sched.load_state_dict(s['scheduler']);optim.load_state_dict(s['optimizer'])
assert equal(sched.state_dict(),s['scheduler']) and equal(optim.state_dict(),s['optimizer'])
random.setstate(s['rng']['python']);np.random.set_state(s['rng']['numpy']);torch.set_rng_state(s['rng']['cpu'])
assert equal(random.getstate(),s['rng']['python']) and equal(np.random.get_state(),s['rng']['numpy']) and torch.equal(torch.get_rng_state(),s['rng']['cpu'])
assert len(s['rng']['cuda'])==1 and s['rng']['cuda'][0].dtype==torch.uint8
for n,p in model.named_parameters():assert equal(p.grad,s['grads'][n]) and p.requires_grad==s['requires_grad'][n]
assert {n:m.training for n,m in model.named_modules()}==s['modes']
actual={p.name:sha(p) for p in sorted((d/'random/buffer').glob('*.pt'))};assert actual==r['buffer_hashes']
from libero.libero.benchmark import get_benchmark
benchmark=get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
noise={};evaluation=d/'random/evaluation'
for stage in range(6,10):
 for task in range(6,stage+1):
  cell=js(evaluation/('cell_stage%d_task%d.json'%(stage,task)));rp=evaluation/('rng_before_stage%d_task%d.pth'%(stage,task));rn=torch.load(rp,map_location='cpu')
  assert set(rn)=={'python','numpy','cpu','cuda'} and len(rn['cuda'])==1 and rn['cuda'][0].dtype==torch.uint8
  assert cell['checkpoint_sha256']==cp[str(stage)]['sha256'] and cell['loaded_downstream_keys']==len(model.state_dict())
  # Source filenames are authoritative; validate 20 recorded hashes against the actual fixed initial-state tensor.
  task_info=benchmark.get_task(task);initial_path=Path(cfg.init_states_folder)/task_info.problem_folder/task_info.init_states_file
  assert sha(initial_path)==cell['initial_file_sha256']
  states=torch.load(initial_path,map_location='cpu');actual_init=[hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for v in states[:20]]
  assert actual_init==cell['initial_hashes']
  noise[str(stage)+'_'+str(task)]=dict(rng_before_sha256=sha(rp),initial_file_sha256=cell['initial_file_sha256'],initial_hashes=actual_init,scheme='accepted stage-reset global action RNG; no per-trajectory noise equality claim')
assert not torch.cuda.is_initialized()
print(json.dumps(dict(status='PASS_CPU_CASE_RECONSTRUCTION',case=case,checkpoints=cp,recovery=dict(strict_model=True,optimizer=True,scheduler=True,gradients=True,modes=True,requires_grad=True,cpu_rng_exact=True,saved_cuda_rng_preserved_not_executed=True,sha256=sha(d/'random/recovery/latest.pth')),initial_states_and_rng=noise,optimizer_calls=0,environments=0)))
