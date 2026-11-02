"""Isolated RTX3090 recovery and paired CPU/device branch replay; zero production writes."""
import os,sys,json,time,hashlib,copy,traceback,io
from pathlib import Path
D=Path(__file__).resolve().parent
Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');R=Q/'L005_41'
sys.path.insert(0,str(R))
import runtime as rt
def put(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def redirect(p,x):put(D/'side_effect_metadata'/Path(p).name,x)
rt.write=redirect
import train as tr
import gate_admission as ga
import torch.distributed as dist
torch=rt.torch
ledger=dict(status='STARTED',started=0,completed=0,extra_autograd=0,environments=0,production_epochs=0)
def record():put(D/'LEDGER.json',dict(ledger,time=time.time()))
orig=torch.optim.AdamW.step
def counted(self,*a,**kw):
 ledger['started']+=1;record();v=orig(self,*a,**kw);ledger['completed']+=1;record();return v
torch.optim.AdamW.step=counted
def device(x):
 if torch.is_tensor(x):return x.detach().clone()
 if isinstance(x,dict):return {k:device(v) for k,v in x.items()}
 if isinstance(x,list):return [device(v) for v in x]
 if isinstance(x,tuple):return tuple(device(v) for v in x)
 return copy.deepcopy(x)
def cap(a,dev):
 if not dev:return rt.capture(a)
 old=rt.cpu
 try:rt.cpu=device;v=rt._reference_capture(a)
 finally:rt.cpu=old
 assert rt.equal(rt.cpu(v),rt.capture(a));return v
def main():
 assert not (D/'VALIDATION.json').exists()
 record();rt.verify();rt.capacity()
 raw={str(p):rt.sha(p) for p in Q.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.exit']}
 cpfile=rt.OUTPUT['S']/'recovery.pth';assert rt.sha(cpfile)=='65b8d3b4c9f589fb543ba5b7a41991a8c1dbc50daae309c21c386471178b4070'
 cp=torch.load(cpfile,map_location='cpu');assert cp['history']==rt.js(R/'EPOCHS_S.json')
 dist.init_process_group('nccl');torch.cuda.set_device(0)
 try:
  assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
  a=rt.make('S');ds=rt.Dataset(8);loader,sampler=rt.setup(a,ds);memory=tr.load_memory('S',8);scale=tr.scale_for(ds);info=rt.js(rt.O/'ra_coreset/ifa_task8.json')['info'];pr=tr.setup_probes(ds,memory)
  assert json.loads(json.dumps(pr['metadata']))==rt.js(R/'PROBES_8.json')
  binding={str(t):dict(cache_sha256=rt.sha(rt.memory_path('S',t)),source_sha256=memory[t]['source_sha256'],count=len(memory[t]['rows']),support=memory[t]['support'],support_trajectories=memory[t]['support_trajectories']) for t in memory}
  assert binding==cp['memory_bindings']
  rt.restore(a,cp['state']);assert rt.equal(cp['state'],rt.capture(a))
  ga.STAGE=8;ledger['extra_autograd']=11;record();ad=ga.admission(a,ds,memory,scale,info,pr,'G');assert ad['optimizer_calls']==29 and rt.equal(cp['state'],rt.capture(a))
  old=[json.loads(s) for s in (R/'GATES.jsonl').read_text().splitlines()][cp['gate_blocks']:];assert len(old)==7
  trajectories=[]
  for dev in [False,True]:
   rt.restore(a,cp['state']);sampler.set_epoch(60);it=iter(loader);a.policy.train();rows=[]
   for bi in range(0,70,10):
    block=[next(it) for _ in range(10)];base=cap(a,dev)
    cr=[tr.update(a,b,'C',memory,scale,info,8,retain_weight=0.) for b in block];cs=cap(a,dev);cv=tr.probe(a,pr,memory)
    rt.restore(a,base);qr=[tr.update(a,b,'Q',memory,scale,info,8) for b in block];qs=cap(a,dev);qv=tr.probe(a,pr,memory)
    decision,checks=tr.choose(cv,qv);rt.restore(a,cs if decision=='C' else qs)
    gate=dict(stage=8,epoch=60,batch=bi,steps=10,start_step=base['step'],end_step=a.step,C=cv,Q=qv,accepted=decision,checks=checks)
    rows.append(dict(gate=gate,C_rows=cr,Q_rows=qr))
    if not dev:
     # Private temporary states stay remote, outside publishable evidence.
     torch.save(rt.capture(a),D/('state_%d.pth'%bi))
    else:assert rt.equal(torch.load(D/('state_%d.pth'%bi),map_location='cpu'),rt.capture(a))
    del base,cs,qs
   trajectories.append(rows)
  assert trajectories[0]==trajectories[1]
  rt.restore(a,cp['state']);assert rt.equal(cp['state'],rt.capture(a))
  assert ledger['started']==ledger['completed']==309
  assert all(rt.sha(Path(p))==h for p,h in raw.items()) and rt.sha(cpfile)== '65b8d3b4c9f589fb543ba5b7a41991a8c1dbc50daae309c21c386471178b4070'
  comparisons=[dict(block=i,all_gate_fields_exact=o==n['gate'],decision_equal=o['accepted']==n['gate']['accepted'],old=o,new=n['gate']) for i,(o,n) in enumerate(zip(old,trajectories[0]))]
  ledger['status']='PASS_GPU_RECOVERY_AND_PAIRED_BRANCH_REPLAY';record()
  put(D/'VALIDATION.json',dict(ledger,checkpoint_sha256=rt.sha(cpfile),history_epochs=260,full_state_restore_exact=True,paired_CPU_device_rows_states_probes_decisions_exact=True,restored_after_diagnostics=True,production_metadata_unchanged=True,hardware=torch.cuda.get_device_name(0),old_GPU_comparisons=comparisons,cross_hardware_bitwise_claim=all(x['all_gate_fields_exact'] for x in comparisons),old_partial_recorded_optimizer_calls=140,old_partial_unrecorded_cost='UNKNOWN',time=time.time()))
 finally:dist.destroy_process_group()
try:main()
except BaseException:
 ledger['status']='FAILED_REQUIRES_REVIEW';record();put(D/'FAILURE.json',dict(error=traceback.format_exc(),ledger=ledger));raise
