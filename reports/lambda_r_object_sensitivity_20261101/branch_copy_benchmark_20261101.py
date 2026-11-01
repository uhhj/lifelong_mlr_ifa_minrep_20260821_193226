"""Single-use isolated copy/restore benchmark; zero optimizer/rollout calls."""
import os,sys,time,json,hashlib,io,traceback,subprocess
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
D=Path('/data/mlr_ifa/diagnostics/lambda_r_branch_copy_benchmark_20261101_v1')
assert not D.exists(),'single use diagnostic';D.mkdir()
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0,str(Q/'L005_41'))
ledger=dict(status='RUNNING',optimizer_calls=0,environments=0,production_code_or_state_written=False,target_signals=0,started=time.time())
def persist():
 (D/'RESULT.json').write_text(json.dumps(ledger,indent=2))
persist()
try:
 import runtime as rt
 import copy
 torch=rt.torch
 def fast(x):
  if torch.is_tensor(x):return x.detach().clone()
  if isinstance(x,dict):return {k:fast(v) for k,v in x.items()}
  if isinstance(x,list):return [fast(v) for v in x]
  if isinstance(x,tuple):return tuple(fast(v) for v in x)
  return copy.deepcopy(x)
 def normal(x):
  if torch.is_tensor(x):return x.detach().cpu()
  if isinstance(x,dict):return {k:normal(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return type(x)(normal(v) for v in x)
  return x
 contract=rt.js(Q/'CONTRACT.json');assert all(rt.sha(p)==h for p,h in contract['files'].items())
 before={str(p):rt.sha(p) for p in (Q/'L005_41').glob('*.py')}
 free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip());assert free>12000
 cp=rt.OUTPUT['S']/'recovery.pth';raw=cp.read_bytes();ledger.update(input_sha256=hashlib.sha256(raw).hexdigest(),input_bytes=len(raw));loaded=torch.load(io.BytesIO(raw),map_location='cpu');del raw
 ledger.update(input_stage=loaded['stage'],input_next_epoch=loaded['next_epoch'])
 # All process-local output destinations are redirected before constructing model.
 rt.R=D;rt.OUTPUT['S']=D/'private';rt.ROOT_PRIVATE=D/'private_root';rt.DATA_PRIVATE=D/'private'
 torch.distributed.init_process_group('nccl',init_method='file://'+str(D/'store'),rank=0,world_size=1);torch.cuda.set_device(0)
 a=rt.make('S');ds=rt.Dataset(loaded['stage']);loader,sampler=rt.setup(a,ds);rt.restore(a,loaded['state']);del loaded
 original=rt.cpu
 def capture(fn):
  rt.cpu=fn
  try:return rt._reference_capture(a)
  finally:rt.cpu=original
 baseline=capture(original);gpu=capture(fast)
 assert rt.equal(baseline,normal(gpu))
 for snap in [gpu,baseline]:
  rt.restore(a,snap);assert rt.equal(baseline,capture(original))
 del gpu
 rounds=[]
 for i in range(10):
  order=['cpu','device'] if i%2==0 else ['device','cpu']
  for mode in order:
   rt.restore(a,baseline);fn=original if mode=='cpu' else fast
   torch.cuda.synchronize();tic=time.perf_counter()
   base=capture(fn);cs=capture(fn);rt.restore(a,base);qs=capture(fn);rt.restore(a,qs);assert rt.equal(qs,capture(fn))
   torch.cuda.synchronize();elapsed=time.perf_counter()-tic
   assert rt.equal(baseline,capture(original))
   rounds.append(dict(round=i,mode=mode,seconds=elapsed));del base,cs,qs
 ledger.update(status='PASS_COPY_RESTORE_EQUALITY_AND_MICROBENCH_ONLY',rounds=rounds,full_training_equivalence='NOT_TESTED',production_activation=False,ended=time.time(),max_memory_allocated=torch.cuda.max_memory_allocated())
 assert rt.CALLS=={'production':0,'admission':0}
 assert all(rt.sha(p)==h for p,h in contract['files'].items()) and all(rt.sha(p)==h for p,h in before.items())
 torch.distributed.destroy_process_group();persist();print(json.dumps(ledger))
except BaseException:
 ledger.update(status='FAIL_DIAGNOSTIC_ONLY',error=traceback.format_exc(),ended=time.time());persist();print(json.dumps(ledger));raise
