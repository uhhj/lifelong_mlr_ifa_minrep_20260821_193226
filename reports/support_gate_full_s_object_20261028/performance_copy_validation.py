"""Bounded standalone state-copy equivalence and timing; zero optimizer/environment calls."""
import os,time,json,copy,hashlib,gc,statistics,random
from pathlib import Path
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
import torch,numpy as np
torch.set_num_threads(1)
P=Path('/data/mlr_ifa/experiments/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1/seed41/S/stage9.pth')
Q=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1')
contract=json.loads((Q/'CONTRACT.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(p)==h for p,h in contract['files'].items())
assert (Q/'seed41/train_S.exit').read_text().strip()=='0'
cp_sha=sha(P);cp=torch.load(P,map_location='cpu');state=cp['state'];assert cp['next_epoch']==100 and cp['stage']==9
def original(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:original(v) for k,v in x.items()}
 if isinstance(x,list):return [original(v) for v in x]
 if isinstance(x,tuple):return tuple(original(v) for v in x)
 return copy.deepcopy(x)
def candidate(x):
 if torch.is_tensor(x):return x.detach().cpu() if x.device.type=='cuda' else x.detach().cpu().clone()
 if isinstance(x,dict):return {k:candidate(v) for k,v in x.items()}
 if isinstance(x,list):return [candidate(v) for v in x]
 if isinstance(x,tuple):return tuple(candidate(v) for v in x)
 return copy.deepcopy(x)
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b) and a.dtype==b.dtype and a.shape==b.shape and a.stride()==b.stride() and a.device==b.device
 if isinstance(a,np.ndarray):return np.array_equal(a,b) and a.dtype==b.dtype
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
def cuda_tree(x):
 if torch.is_tensor(x):return x.cuda()
 if isinstance(x,dict):return {k:cuda_tree(v) for k,v in x.items()}
 if isinstance(x,list):return [cuda_tree(v) for v in x]
 if isinstance(x,tuple):return tuple(cuda_tree(v) for v in x)
 return copy.deepcopy(x)
def tensors(x):
 if torch.is_tensor(x):yield x
 elif isinstance(x,dict):
  for y in x.values():yield from tensors(y)
 elif isinstance(x,(list,tuple)):
  for y in x:yield from tensors(y)
mixed=copy.deepcopy(state)
for k in ('model','optimizer','grads'):mixed[k]=cuda_tree(state[k])
torch.cuda.synchronize()
before_rng=dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()])
base=original(mixed);fast=candidate(mixed);assert equal(base,fast) and equal(base,state)
# The CPU result owns distinct storage for every tensor, both from source and other result.
assert all(x.device!=y.device or x.data_ptr()!=y.data_ptr() or x.numel()==0 for x,y in zip(tensors(mixed),tensors(fast)))
assert all(x.data_ptr()!=y.data_ptr() or x.numel()==0 for x,y in zip(tensors(base),tensors(fast)))
source_tensor=next(x for x in tensors(mixed['model']) if x.numel());backup=source_tensor.clone()
source_tensor.reshape(-1)[0]+=1;assert equal(fast,state);source_tensor.copy_(backup)
cpu_input=copy.deepcopy(state);cpu_result=candidate(cpu_input);t=next(x for x in tensors(cpu_input['model']) if x.numel());t.reshape(-1)[0]+=1;assert equal(cpu_result,state)
del base,fast,cpu_input,cpu_result,backup;gc.collect()
timings={'original':[],'candidate':[]}
for i in range(6):
 for name,fn in ([('original',original),('candidate',candidate)] if i%2==0 else [('candidate',candidate),('original',original)]):
  torch.cuda.synchronize();start=time.perf_counter();out=fn(mixed);torch.cuda.synchronize();timings[name].append(time.perf_counter()-start)
  assert equal(out,state);del out;gc.collect()
after_rng=dict(py=random.getstate(),np=np.random.get_state(),cpu=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()]);assert equal(before_rng,after_rng)
assert sha(P)==cp_sha and all(sha(p)==h for p,h in contract['files'].items())
print(json.dumps(dict(status='PASS_COPY_ONLY_EXACT_NO_TRAINING_REPLAY',time=time.time(),checkpoint_sha256=cp_sha,checkpoint_bytes=P.stat().st_size,state_tensor_bytes=sum(t.numel()*t.element_size() for t in tensors(state)),tensor_count=sum(1 for t in tensors(state)),dtype_shape_stride_value_exact=True,source_mutation_isolation=True,cpu_input_independence=True,rng_exact=True,frozen_files_exact=True,timings_seconds=timings,median_speedup=statistics.median(timings['original'])/statistics.median(timings['candidate']),extra_optimizer_calls=0,extra_autograd_calls=0,extra_environments=0,target_process_modified=False,limits=['Copy primitive only; not full optimizer replay or whole-run performance acceptance.','Separate process shares the GPU during bounded copy microbenchmark; do not use these times as isolated production speedup.','Current training interpreter already imported original helper; editing files would not safely replace it.'])))
