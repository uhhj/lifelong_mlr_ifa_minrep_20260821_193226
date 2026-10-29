"""Paired full-epoch training-path replay, isolated from all scientific outputs."""
import os,sys,json,time,hashlib,traceback,copy,io
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent
D=Path('/data/mlr_ifa/diagnostics/support_gate_copy_training_20261029_v1')
os.environ.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.path.insert(0,str(Q/'seed41'))
import runtime as rt
from candidate import cpu_snapshot
import train as tr
import fork_tools as ft
from updates import update as original_update
from support_memory import load_memory
import torch.distributed as dist
torch=rt.torch
ledger=dict(status='RUNNING',started=time.time(),optimizer_calls_started=0,optimizer_calls_completed=0,extra_autograd_calls=0,environments=0,accepted_scientific_epochs=0,stages=[])
def persist():
 p=E/'VALIDATION_PROGRESS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(ledger,indent=2));os.replace(tmp,p)
def upd(*args,**kwargs):
 ledger['optimizer_calls_started']+=1
 out=original_update(*args,**kwargs);ledger['optimizer_calls_completed']+=1
 return out
original_cpu=rt.cpu
def select(fn):
 rt.cpu=fn;tr.cpu=fn;ft.cpu=fn
def check_equal(x,y,label):
 assert rt.equal(x,y),label
 if torch.is_tensor(x):assert x.dtype==y.dtype and x.shape==y.shape and x.stride()==y.stride() and x.device==y.device,label
 elif isinstance(x,dict):
  for k in x:check_equal(x[k],y[k],label+'/'+str(k))
 elif isinstance(x,(tuple,list)):
  assert type(x)==type(y),label
  for i,(a,b) in enumerate(zip(x,y)):check_equal(a,b,label+'/'+str(i))
def main():
 assert not D.exists(),'Isolated diagnostic single use';D.mkdir(parents=True)
 contract=rt.js(Q/'CONTRACT.json');assert all(rt.sha(p)==h for p,h in contract['files'].items())
 before={str(p):rt.sha(p) for p in (Q/'seed41').glob('*.json')}
 srcout=rt.OUTPUT['S'];cp_hash={str(srcout/('stage%d.pth'%t)):rt.sha(srcout/('stage%d.pth'%t)) for t in range(6,10)}
 ledger.update(contract_sha256=rt.sha(Q/'CONTRACT.json'),checkpoint_sha256=cp_hash,scope='Every batch of one complete epoch per Task6/7/8/9, paired original/candidate from identical full state; not a repeat of accepted scientific epochs or a full 400-epoch duplicate.')
 persist();dist.init_process_group('nccl');torch.cuda.set_device(0)
 try:
  # build() only makes these private directories; restore OUTPUT before reading actual caches.
  rt.OUTPUT['S']=D/'build';a=rt.make('S');rt.OUTPUT['S']=srcout
  for stage in range(6,10):
   ds=rt.Dataset(stage);loader,sampler=rt.setup(a,ds)
   if stage>6:
    cp=torch.load(srcout/('stage%d.pth'%(stage-1)),map_location='cpu');a.policy.load_state_dict(cp['state_dict'],strict=True);a.step=cp['steps'];del cp
   a.policy.train();memory=load_memory('S',stage);scale=tr.scale_for(ds);info=rt.js(rt.O/('ra_coreset/ifa_task%d.json'%stage))['info'];pr=ft.setup_probes(ds,memory) if stage>6 else None
   sampler.set_epoch(0);iterator=iter(loader);nb=0;blocks=0;choices={'C':0,'Q':0};started=time.time();epoch_totals=None
   while True:
    batch=[]
    for _ in range(10 if stage>6 else 1):
     try:batch.append(next(iterator))
     except StopIteration:break
    if not batch:break
    select(original_cpu);base=rt.capture(a)
    def run(fn,oracle=None):
     select(fn);rt.restore(a,base)
     if stage==6:
      rows=[upd(a,batch[0],'S',memory,scale,info,stage)];end=rt.capture(a)
      result=dict(rows=rows,end=end)
     else:
      cr=[upd(a,b,'C',memory,scale,info,stage,retain_weight=0.) for b in batch];cs=rt.capture(a);cv=ft.probe(a,pr,memory)
      rt.restore(a,base);qr=[upd(a,b,'Q',memory,scale,info,stage) for b in batch];qs=rt.capture(a);qv=ft.probe(a,pr,memory)
      decision,checks=ft.choose(cv,qv)
      # Exercise both branch restore paths even when the gate chooses only one.
      for state in [cs,qs]:rt.restore(a,state);check_equal(state,rt.capture(a),'branch restore')
      rt.restore(a,cs if decision=='C' else qs);end=rt.capture(a)
      result=dict(C_rows=cr,Q_rows=qr,C_state=cs,Q_state=qs,C_probe=cv,Q_probe=qv,decision=decision,checks=checks,end=end)
     if oracle is not None:check_equal(oracle,result,'paired stage%d block%d'%(stage,blocks))
     return result
    ref=run(original_cpu);fast=run(cpu_snapshot,ref)
    if stage>6:choices[fast['decision']]+=1
    rows=fast['rows'] if stage==6 else fast[fast['decision']+'_rows']
    numeric={k:sum(row[k] for row in rows) for k in ['loss','bc','ifa','expert_loss','expert_route','retain_loss','retain_route','valid','invalid','old_eligible','anchor_slots','anchor_actual','anchor_empty','anchor_forwards']}
    if epoch_totals is None:epoch_totals={k:0. for k in numeric}
    for k,v in numeric.items():epoch_totals[k]+=v
    if blocks==0:
     selected=fast['end'];stream=io.BytesIO();torch.save(dict(state=selected,stage=stage,sampling_cursor=a.step),stream);stream.seek(0);loaded=torch.load(stream,map_location='cpu');check_equal(selected,loaded['state'],'serialization')
     select(original_cpu);rt.restore(a,selected);r1=upd(a,batch[0],'S' if stage==6 else 'Q',memory,scale,info,stage);s1=rt.capture(a)
     select(cpu_snapshot);rt.restore(a,loaded['state']);r2=upd(a,batch[0],'S' if stage==6 else 'Q',memory,scale,info,stage);s2=rt.capture(a)
     check_equal(r1,r2,'next step scalars');check_equal(s1,s2,'next step full state');rt.restore(a,selected)
     del stream,loaded,s1,s2,selected
     if stage>6:
      dg=[]
      for fn in [original_cpu,cpu_snapshot]:
       select(fn);x=ft.directional_diagnostic(a,batch[0],memory,pr,'Q');dg.append(x)
       if x is not None:ledger['extra_autograd_calls']+=x['extra_autograd_calls']
      check_equal(dg[0],dg[1],'directional diagnostics')
    nb+=len(batch);blocks+=1
    del ref,fast,base
    ledger['current']=dict(stage=stage,batches=nb,blocks=blocks,seconds=time.time()-started);persist()
   assert nb==len(loader)
   assert epoch_totals['valid']==rt.VALID[stage] and epoch_totals['invalid']==rt.POP[stage]-rt.VALID[stage] and epoch_totals['old_eligible']==rt.OLD[stage]
   ledger['stages'].append(dict(stage=stage,batches=nb,blocks=blocks,tail_batches=nb%10 if stage>6 else 1,choices=choices,seconds=time.time()-started,full_state_exact=True,all_rows_probes_decisions_exact=True,serialization_next_step_exact=True,totals=epoch_totals));persist()
  assert ledger['optimizer_calls_completed']==ledger['optimizer_calls_started']==3914
  assert all(rt.sha(p)==h for p,h in contract['files'].items()) and all(rt.sha(p)==h for p,h in cp_hash.items())
  assert all(rt.sha(p)==h for p,h in before.items())
  ledger.update(status='PASS_FULL_TRAINING_PATH_EXACT',ended=time.time(),original_sources_unchanged=True,completed_seed41_artifacts_unchanged=True,scientific_outputs_written=False,seed42_signals=0)
  persist();(E/'VALIDATION.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(ledger),flush=True)
 finally:dist.destroy_process_group()
if __name__=='__main__':
 try:main()
 except BaseException:
  ledger.update(status='FAIL',error=traceback.format_exc(),ended=time.time());persist();raise
