"""Isolated serial/concurrent path and throughput probe; no production files written."""
import os,sys,json,time,hashlib,copy,inspect,io,traceback
from pathlib import Path
D=Path(sys.argv[1]);seed=int(sys.argv[2]);mode=sys.argv[3];stages=[int(x) for x in sys.argv[4].split(',')];lambdas=[float(x) for x in sys.argv[5].split(',')]
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1')
B=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1')/('seed'+str(seed))
P=Path('/data/mlr_ifa/experiments/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1')/('seed'+str(seed))/'S'
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0,str(Q/('L005_'+str(seed))))
import runtime as rt
import train as tr
import fork_tools as ft
import memory as mm
import support_memory as sm
torch=rt.torch
ledger=dict(status='RUNNING',seed=seed,mode=mode,started=time.time(),optimizer_started=0,optimizer_completed=0,environments=0,extra_autograd=0,production_writes=False,production_epochs=0,target_signals=0,stages=[])
def put(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def persist():put(D/'PROGRESS.json',ledger)
def clone(x):
 if torch.is_tensor(x):return x.detach().clone()
 if isinstance(x,dict):return {k:clone(v) for k,v in x.items()}
 if isinstance(x,list):return [clone(v) for v in x]
 if isinstance(x,tuple):return tuple(clone(v) for v in x)
 return copy.deepcopy(x)
def capture(a):
 old=rt.cpu
 try:rt.cpu=clone;return rt._reference_capture(a)
 finally:rt.cpu=old
def digest(x):
 h=hashlib.sha256()
 def visit(v):
  if torch.is_tensor(v):
   h.update(repr(('tensor',str(v.dtype),tuple(v.shape),tuple(v.stride()))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
  elif isinstance(v,rt.np.ndarray):h.update(repr(('ndarray',str(v.dtype),v.shape)).encode());h.update(v.tobytes())
  elif isinstance(v,dict):
   h.update(b'dict')
   for k,w in v.items():visit(k);visit(w)
  elif isinstance(v,(tuple,list)):
   h.update(type(v).__name__.encode())
   for w in v:visit(w)
  else:h.update(repr((type(v).__name__,v)).encode())
 visit(x);return h.hexdigest()
def finite(a,loss):return bool(torch.stack([torch.isfinite(loss).all()]+[torch.isfinite(p.grad).all() for p in a.policy.parameters() if p.grad is not None]).all())
src=inspect.getsource(tr.update);needle='assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in a.policy.parameters())';assert src.count(needle)==1
ns=dict(tr.update.__globals__,finite=finite);exec(compile(src.replace(needle,'assert finite(a,loss)'),str(D/'exact_update.py'),'exec'),ns)
def update(*args,**kwargs):
 ledger['optimizer_started']+=1;r=ns['update'](*args,**kwargs);ledger['optimizer_completed']+=1;return r
def wait_file(p):
 start=time.time()
 while not p.exists():
  assert time.time()-start<900,'Barrier timeout'
  time.sleep(.05)
def main():
 assert not (D/'PROGRESS.json').exists(),'single use'
 assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
 rt.R=D;tr.R=D;ft.R=D;mm.R=B;sm.R=B
 contract=rt.js(Q/'CONTRACT.json');assert all(rt.sha(p)==h for p,h in contract['files'].items())
 baseline={str(p):rt.sha(p) for p in [*[P/('stage%d.pth'%i) for i in range(6,10)],*[P/'memory'/('task%d.pth'%i) for i in range(6,9)]]};ledger['baseline_hashes']=baseline;persist()
 torch.distributed.init_process_group('nccl',init_method='file://'+str(D/'store'),rank=0,world_size=1);torch.cuda.set_device(0)
 try:
  for lam in lambdas:
   rt.control_seed(seed);rt.OUTPUT['S']=D/('private_'+str(lam));a=rt.make('S');rt.OUTPUT['S']=P;ft.RETAIN_WEIGHT=lam
   for stage in stages:
    rt.control_seed(seed*1000+stage)
    ds=rt.Dataset(stage);loader,sampler=rt.setup(a,ds)
    if stage>6:
     cp=torch.load(P/('stage%d.pth'%(stage-1)),map_location='cpu');a.policy.load_state_dict(cp['state_dict'],strict=True);a.step=cp['steps'];del cp
    a.policy.train();memory=sm.load_memory('S',stage);scale=tr.scale_for(ds);info=rt.js(rt.O/('ra_coreset/ifa_task%d.json'%stage))['info'];pr=ft.setup_probes(ds,memory) if stage>6 else None
    sampler.set_epoch(0);iterator=iter(loader);initial=digest(capture(a));label=str(lam)+'_'+str(stage)
    put(D/(label+'_READY.json'),dict(time=time.time(),initial=initial));wait_file(D.parent/(mode+'_'+label+'_GO.json'))
    torch.cuda.synchronize();start=time.time();compute=0.;blocks=[];nb=0
    while True:
     batch=[]
     for _ in range(10 if stage>6 else 1):
      try:batch.append(next(iterator))
      except StopIteration:break
     if not batch:break
     torch.cuda.synchronize();t=time.perf_counter()
     if stage==6:
      rows=[update(a,batch[0],'S',memory,scale,info,stage,retain_weight=lam)];selected=capture(a);record=dict(rows=rows)
     else:
      base=capture(a);cr=[update(a,b,'C',memory,scale,info,stage,retain_weight=0.) for b in batch];cs=capture(a);cv=ft.probe(a,pr,memory)
      rt.restore(a,base);qr=[update(a,b,'Q',memory,scale,info,stage,retain_weight=lam) for b in batch];qs=capture(a);qv=ft.probe(a,pr,memory);choice,checks=ft.choose(cv,qv)
      selected=cs if choice=='C' else qs;rt.restore(a,selected);assert rt.equal(selected,capture(a))
      record=dict(C_rows=cr,Q_rows=qr,C_probe=cv,Q_probe=qv,choice=choice,checks=checks)
     torch.cuda.synchronize();compute+=time.perf_counter()-t
     # Canonical state digests are diagnostic-only and excluded from compute spans.
     if stage>6:del base,cs,qs
     blocks.append(record)
     del selected;nb+=len(batch);ledger['current']=dict(lambda_R=lam,stage=stage,batches=nb,blocks=len(blocks));persist()
    torch.cuda.synchronize();ended=time.time();assert nb==len(loader)
    record=dict(lambda_R=lam,stage=stage,initial=initial,final=digest(capture(a)),blocks=blocks,batches=nb)
    put(D/(label+'_TRACE.json'),record)
    summary=dict(lambda_R=lam,stage=stage,batches=nb,trace_sha256=rt.sha(D/(label+'_TRACE.json')),started=start,ended=ended,elapsed_seconds=ended-start,compute_span_seconds=compute,extra_state_digest_overhead_in_elapsed=False)
    ledger['stages'].append(summary);persist();put(D/(label+'_DONE.json'),summary)
    del ds,loader,iterator,memory,pr,blocks
   del a;torch.cuda.empty_cache()
  assert ledger['optimizer_started']==ledger['optimizer_completed']==sum((157 if st==6 else 2*{7:227,8:309,9:362}[st]) for st in stages)*len(lambdas)
  assert all(rt.sha(p)==h for p,h in baseline.items()) and all(rt.sha(p)==h for p,h in contract['files'].items())
  ledger.update(status='PASS_WORKER_EXECUTION_PENDING_PAIR_COMPARISON',ended=time.time());persist();put(D/'RESULT.json',ledger)
 finally:torch.distributed.destroy_process_group()
if __name__=='__main__':
 try:main()
 except BaseException:
  ledger.update(status='FAIL_DIAGNOSTIC_ONLY',error=traceback.format_exc(),ended=time.time());persist();raise
