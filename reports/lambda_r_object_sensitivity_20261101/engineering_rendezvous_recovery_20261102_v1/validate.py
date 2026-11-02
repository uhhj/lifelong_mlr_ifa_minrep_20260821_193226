"""Isolated recovery of the saved epoch and partial gate prefix; zero environment."""
import os,sys,json,time,hashlib,copy,traceback,io
from pathlib import Path
D=Path(__file__).resolve().parent;Q=Path('/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_OBJECT41_42_43_20261101_v1');R=Q/'L005_43'
sys.path.insert(0,str(R))
import runtime as rt

def put(p,x):
 p=Path(p);p.parent.mkdir(exist_ok=True,parents=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
rt.write=lambda p,x:put(D/'side_effect_metadata'/Path(p).name,x)
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
 assert not (D/'VALIDATION.json').exists();record();rt.verify();rt.capacity();binding=rt.js(D/'BINDING.json')
 cpfile=rt.OUTPUT['S']/'recovery.pth';assert rt.sha(cpfile)==binding['checkpoint_sha256'];cp=torch.load(cpfile,map_location='cpu');assert cp['history']==rt.js(R/'EPOCHS_S.json') and len(cp['history'])==289
 original={str(p):rt.sha(p) for p in Q.rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.exit']}
 dist.init_process_group('nccl');torch.cuda.set_device(0)
 try:
  assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 3090'
  a=rt.make('S');ds=rt.Dataset(8);loader,sampler=rt.setup(a,ds);memory=tr.load_memory('S',8);scale=tr.scale_for(ds);info=rt.js(rt.O/'ra_coreset/ifa_task8.json')['info'];pr=tr.setup_probes(ds,memory)
  saved=rt.js(R/'PROBES_8.json');generated=json.loads(json.dumps(pr['metadata']));assert generated==saved
  original_scale=rt.js(R/'TRAIN_INPUT_S_8.json')['scale'];assert torch.equal(scale,torch.tensor(original_scale,device='cuda'))
  memory_binding={str(t):dict(cache_sha256=rt.sha(rt.memory_path('S',t)),source_sha256=memory[t]['source_sha256'],count=len(memory[t]['rows']),support=memory[t]['support'],support_trajectories=memory[t]['support_trajectories']) for t in memory}
  assert memory_binding==cp['memory_bindings'] and cp['probe_sha256']==rt.sha(R/'PROBES_8.json')
  rt.restore(a,cp['state']);assert rt.equal(cp['state'],rt.capture(a));stream=io.BytesIO();torch.save(cp['state'],stream);stream.seek(0);assert rt.equal(cp['state'],torch.load(stream,map_location='cpu'));del stream
  ga.STAGE=8;ledger['extra_autograd']=11;record();ad=ga.admission(a,ds,memory,scale,info,pr,'G');assert ad['optimizer_calls']==29 and rt.equal(cp['state'],rt.capture(a))
  old=[json.loads(s) for s in (R/'GATES.jsonl').read_text().splitlines()][cp['gate_blocks']:];assert len(old)==20
  sampler.set_epoch(89);it=iter(loader);a.policy.train();comparisons=[]
  for bi in range(0,210,10):
   block=[next(it) for _ in range(10)];start=rt.capture(a);paths=[];expected=None
   for dev in [False,True]:
    rt.restore(a,start);base=cap(a,dev)
    cr=[tr.update(a,b,'C',memory,scale,info,8,retain_weight=0.) for b in block];cs=cap(a,dev);cv=tr.probe(a,pr,memory)
    rt.restore(a,base);qr=[tr.update(a,b,'Q',memory,scale,info,8) for b in block];qs=cap(a,dev);qv=tr.probe(a,pr,memory)
    decision,checks=tr.choose(cv,qv);rt.restore(a,cs if decision=='C' else qs)
    gate=dict(stage=8,epoch=89,batch=bi,steps=10,start_step=base['step'],end_step=a.step,C=cv,Q=qv,accepted=decision,checks=checks)
    paths.append(dict(gate=gate,C_rows=cr,Q_rows=qr))
    if not dev:expected=dict(C=cs,Q=qs,selected=rt.capture(a))
    else:
     assert rt.equal(expected['C'],rt.cpu(cs)) and rt.equal(expected['Q'],rt.cpu(qs)) and rt.equal(expected['selected'],rt.capture(a))
    del base,cs,qs
   assert paths[0]==paths[1]
   if bi<200:assert paths[0]['gate']==old[bi//10]
   comparisons.append(dict(batch=bi,all_candidate_rows_states_probes_decisions_exact=True,original_gate_exact=bi<200,next_unsaved_block=bi==200));put(D/'REPLAY_PROGRESS.json',dict(blocks=len(comparisons),optimizer_completed=ledger['completed'],time=time.time()))
   del expected,start
  rt.restore(a,cp['state']);assert rt.equal(cp['state'],rt.capture(a));assert ledger['started']==ledger['completed']==869
  assert all(rt.sha(Path(p))==h for p,h in original.items()) and rt.sha(cpfile)==binding['checkpoint_sha256']
  ledger['status']='PASS_SAME_GPU_RECOVERY_PARTIAL_PREFIX_AND_NEXT_BLOCK';record()
  put(D/'VALIDATION.json',dict(ledger,checkpoint_sha256=rt.sha(cpfile),history_epochs=289,full_state_restore_exact=True,save_load_roundtrip_exact=True,restored_after_diagnostics=True,production_metadata_unchanged=True,hardware=torch.cuda.get_device_name(0),comparisons=comparisons,partial_recorded_optimizer_calls=400,partial_unrecorded_cost='UNKNOWN',time=time.time()))
 finally:dist.destroy_process_group()
try:main()
except BaseException:
 ledger['status']='FAILED_REQUIRES_REVIEW';record();put(D/'FAILURE.json',dict(error=traceback.format_exc(),ledger=ledger));raise
