"""Isolated paired one-epoch path per stage and lambda; no production writes."""
import os,sys,json,time,hashlib,traceback,io,copy,inspect
from pathlib import Path
D=Path(__file__).resolve().parent
Q=D/'source'
B=Path('/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1/seed41')
P=Path('/data/mlr_ifa/experiments/SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1/seed41/S')
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0,str(Q/'L005_41'))
import runtime as rt
import train as tr
import fork_tools as ft
import memory as mm
import support_memory as sm
from updates import update
def finite_check(loss,parameters):
 return bool(rt.torch.stack([rt.torch.isfinite(loss).all()]+[rt.torch.isfinite(p.grad).all() for p in parameters if p.grad is not None]).all())
update_source=inspect.getsource(update)
needle="assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in a.policy.parameters())"
assert update_source.count(needle)==1
namespace=dict(update.__globals__,finite_check=finite_check)
exec(compile(update_source.replace(needle,"assert finite_check(loss,a.policy.parameters())"),str(D/'candidate_updates.py'),'exec'),namespace)
candidate_update=namespace['update']
fast_mode=False

torch=rt.torch
ledger=dict(status='RUNNING',started=time.time(),optimizer_started=0,optimizer_completed=0,extra_autograd=0,environments=0,production_epochs=0,production_writes=False,target_signals=0,stages=[],scope='GOAL original CPU snapshot and checks versus device snapshots and exact device/batched finite checks; paired every batch of full epoch Task6–9 at .05/.2. Baseline seed41 CP/cache diagnostic-only; no production epochs.')
def persist():
 p=D/'VALIDATION_PROGRESS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));os.replace(tmp,p)
def clone_device(x):
 if torch.is_tensor(x):return x.detach().clone()
 if isinstance(x,dict):return {k:clone_device(v) for k,v in x.items()}
 if isinstance(x,list):return [clone_device(v) for v in x]
 if isinstance(x,tuple):return tuple(clone_device(v) for v in x)
 return copy.deepcopy(x)
original_cpu=rt.cpu
def capture(a,device=False):
 rt.cpu=clone_device if device else original_cpu
 try:return rt._reference_capture(a)
 finally:rt.cpu=original_cpu
def exact(x,y,label):
 if torch.is_tensor(x):
  assert torch.is_tensor(y) and x.dtype==y.dtype and x.shape==y.shape and x.stride()==y.stride(),label
  assert torch.equal(x.detach().cpu(),y.detach().cpu()),label
 elif isinstance(x,dict):
  assert x.keys()==y.keys(),label
  for k in x:exact(x[k],y[k],label+'/'+str(k))
 elif isinstance(x,(tuple,list)):
  assert type(x)==type(y) and len(x)==len(y),label
  for i,(u,v) in enumerate(zip(x,y)):exact(u,v,label+'/'+str(i))
 else:assert rt.equal(x,y),label
def upd(*a,**kw):
 ledger['optimizer_started']+=1
 r=(candidate_update if fast_mode else update)(*a,**kw);ledger['optimizer_completed']+=1
 return r
def main():
 assert not (D/'VALIDATION_PROGRESS.json').exists(),'single use'
 assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
 contract=rt.js(Q/'CONTRACT.json');assert all(rt.sha(p)==h for p,h in contract['files'].items())
 baseline_hash={str(p):rt.sha(p) for p in [*[P/('stage%d.pth'%t) for t in range(6,10)],*[P/'memory'/('task%d.pth'%t) for t in range(6,9)]]}
 ledger['baseline_hashes']=baseline_hash;ledger['diagnostic_source_contract_sha256']=rt.sha(Q/'CONTRACT.json');persist()
 torch.distributed.init_process_group('nccl',init_method='file://'+str(D/'store'),rank=0,world_size=1);torch.cuda.set_device(0)
 rt.R=D;tr.R=D;ft.R=D;mm.R=B;sm.R=B
 try:
  for lam in [.05,.2]:
   rt.OUTPUT['S']=D/('private_'+str(lam));a=rt.make('S');rt.OUTPUT['S']=P
   ft.RETAIN_WEIGHT=lam
   for stage in range(6,10):
    ds=rt.Dataset(stage);loader,sampler=rt.setup(a,ds)
    if stage>6:
     cp=torch.load(P/('stage%d.pth'%(stage-1)),map_location='cpu');a.policy.load_state_dict(cp['state_dict'],strict=True);a.step=cp['steps'];del cp
    a.policy.train();memory=sm.load_memory('S',stage);scale=tr.scale_for(ds);info=rt.js(rt.O/('ra_coreset/ifa_task%d.json'%stage))['info'];pr=ft.setup_probes(ds,memory) if stage>6 else None
    sampler.set_epoch(0);iterator=iter(loader);nb=0;blocks=0;choices={'C':0,'Q':0};started=time.time()
    while True:
     batch=[]
     for _ in range(10 if stage>6 else 1):
      try:batch.append(next(iterator))
      except StopIteration:break
     if not batch:break
     base=capture(a)
     def path(device):
      global fast_mode
      fast_mode=device
      rt.restore(a,base)
      path_started=time.perf_counter()
      if stage==6:
       rows=[upd(a,batch[0],'S',memory,scale,info,stage,retain_weight=lam)]
       return dict(rows=rows,end=capture(a))
      branch_base=capture(a,device)
      cr=[upd(a,b,'C',memory,scale,info,stage,retain_weight=0.) for b in batch];cs=capture(a,device);cv=ft.probe(a,pr,memory)
      rt.restore(a,branch_base)
      qr=[upd(a,b,'Q',memory,scale,info,stage,retain_weight=lam) for b in batch];qs=capture(a,device);qv=ft.probe(a,pr,memory)
      decision,checks=ft.choose(cv,qv)
      for state in [cs,qs]:
       rt.restore(a,state);assert rt.equal(state,capture(a,True)) if device else rt.equal(original_cpu(state),original_cpu(capture(a)))
      selected=cs if decision=='C' else qs;rt.restore(a,selected);assert rt.equal(selected,capture(a,True)) if device else rt.equal(original_cpu(selected),original_cpu(capture(a)))
      return dict(C_rows=cr,Q_rows=qr,C_state=cs,Q_state=qs,C_probe=cv,Q_probe=qv,decision=decision,checks=checks,end=capture(a))
     paths={}
     for mode in ([False,True] if blocks%2==0 else [True,False]):paths[mode]=path(mode)
     ref=paths[False];fast=paths[True];exact(ref,fast,'lambda%s/stage%d/block%d'%(lam,stage,blocks))
     if stage>6:choices[fast['decision']]+=1
     if blocks==0:
      selected=fast['end'];stream=io.BytesIO();torch.save(dict(state=selected,stage=stage,cursor=a.step),stream);stream.seek(0);loaded=torch.load(stream,map_location='cpu');exact(selected,loaded['state'],'serialized')
      rt.restore(a,selected);r1=upd(a,batch[0],'S' if stage==6 else 'Q',memory,scale,info,stage,retain_weight=lam);s1=capture(a)
      rt.restore(a,loaded['state']);r2=upd(a,batch[0],'S' if stage==6 else 'Q',memory,scale,info,stage,retain_weight=lam);s2=capture(a)
      exact(r1,r2,'next step scalars');exact(s1,s2,'next step state');rt.restore(a,selected)
      del stream,loaded,s1,s2,selected
      if stage>6:
       dg=[]
       for device in [False,True]:
        state=capture(a,device);rt.restore(a,state);x=ft.directional_diagnostic(a,batch[0],memory,pr,'Q');dg.append(x)
        if x is not None:ledger['extra_autograd']+=x['extra_autograd_calls']
       exact(dg[0],dg[1],'directional diagnostic')
     nb+=len(batch);blocks+=1
     del ref,fast,base,paths
     ledger['current']=dict(lambda_R=lam,stage=stage,batches=nb,blocks=blocks,seconds=time.time()-started);persist()
    assert nb==len(loader)
    ledger['stages'].append(dict(lambda_R=lam,stage=stage,batches=nb,blocks=blocks,choices=choices,all_rows_states_probes_decisions_exact=True,serialization_next_step_exact=True,seconds=time.time()-started));persist()
   del a;torch.cuda.empty_cache()
  assert ledger['optimizer_started']==ledger['optimizer_completed']==5704
  assert all(rt.sha(p)==h for p,h in contract['files'].items()) and all(rt.sha(p)==h for p,h in baseline_hash.items())
  ledger.update(status='PASS_GOAL_FULL_PATH_BOTH_LAMBDAS',ended=time.time(),production_activation=False);persist();(D/'VALIDATION.json').write_text(json.dumps(ledger,indent=2))
 finally:torch.distributed.destroy_process_group()
if __name__=='__main__':
 try:main()
 except BaseException:
  ledger.update(status='FAIL_DIAGNOSTIC_ONLY',error=traceback.format_exc(),ended=time.time());persist();raise
