import gc,gzip,time,os,sys
import numpy as np
from .common import *
from sample_task_qp_v1.measure import g,recovery,probes,anchor_gradients,project,sample_gradients,sample_project,inner,norm
from functional_geometry_diagnostic_v1.runtime_guard import memory,size
torch=g.torch
class Guard:
 def __init__(self,point):self.point=point;self.started=time.time();self.last=0
 def check(self,force=False):
  if not force and time.time()-self.last<1:return
  self.last=time.time();m=memory();r=read(OUT/'PROTOCOL_FROZEN.json')['resources'];import shutil
  assert m['rss']<r['host_rss_GiB']*2**30 and m['available']>r['host_available_min_GiB']*2**30,'Host memory budget'
  assert torch.cuda.memory_allocated()<r['cuda_allocated_GiB']*2**30 and torch.cuda.memory_reserved()<r['cuda_reserved_GiB']*2**30,'CUDA memory budget'
  assert time.time()-self.started<r['train_job_hours']*3600,'Train watchdog'
  assert shutil.disk_usage(ROOT).free>r['data_free_min_GiB']*2**30 and shutil.disk_usage('/root').free>r['root_free_min_GiB']*2**30,'Disk budget'
  write(OUT/'TRAIN_HEARTBEAT.json',dict(point=self.point,pid=os.getpid(),time=time.time(),seconds=time.time()-self.started,rss=m['rss'],cuda_allocated=torch.cuda.memory_allocated(),cuda_reserved=torch.cuda.memory_reserved()))

def assets(name):
 spec=read(OUT/'TRAINING_MEMBERSHIP.json')[name]
 with gzip.open(ROOT/'reports/gradient_relation_4090_v2/DESIGN.json.gz','rt') as f:design=json.load(f)
 recovery.OUT=OUT;algo,ds,plan=recovery.build_without_monitors(spec['point']['seed'],8,design)
 anchor=spec['origin']['anchor'];assert sha(anchor['path'])==anchor['sha256'];base=torch.load(anchor['path'],map_location='cpu');g.restore(algo,base)
 assert all(g.digest(base[k])==v for k,v in spec['origin']['witness'].items())
 assert all(g.compare_states(base,g.capture(algo)).values())
 P=probes(algo,ds,spec['probe']);Pold={k:v for k,v in P.items() if k[1]<8}
 lineage={(r['point'],r['eval_task'],r['half'],r['position_in_half']):r for r in read(OUT/'PROBE_LINEAGE.json')}
 return spec,algo,ds,plan,base,Pold,lineage

def update(algo,ds,plan,spec,Pold,lineage,name,arm,h,guard,check=False):
 start=time.time();stage=spec['point']['stage_step']+h-1
 if h>1:g.ensure_next_epoch_rng(algo,ds,plan,stage-1,stage)
 before=g.capture(algo) if arm!='MLR' else None
 H=None;K=None
 if arm=='TASK_STEP_QP':H,tasks,_=anchor_gradients(algo,Pold,fullcheck=check)
 elif arm=='SAMPLE_STEP_QP':
  H,rr=sample_gradients(algo,Pold,lineage,name,guard);K,_=inner(H,gram=True)
 g.native(algo,g.batch(ds,spec['batches'][h-1]),8,stage//plan['batches_per_epoch'])
 rec=dict(point=name,arm=arm,horizon=h,stage_step=stage,lr=algo.optimizer.param_groups[0]['lr'],batch_indices=spec['batches'][h-1])
 if H is not None:
  native=g.capture(algo);d=g.displacement(algo,before).cuda()
  if arm=='TASK_STEP_QP':c,sol=project(H,d,fullcheck=False)
  else:c,sol=sample_project(H,K,d,independent=False)
  g.assign_displacement(algo,native,-c);actual=g.displacement(algo,before).cuda();after=g.capture(algo)
  for key in ['optimizer','scheduler','rng','optimizer_step','scheduler_step','training_modes']:
   assert g.digest(after[key])==g.digest(native[key]),('Displacement correction changed native state',arm,h,key)
  _,rates=inner(H,actual);_,native_rates=inner(H,d);dn=norm(d);an=norm(actual)
  hn=H.double().square().sum(dim=0).sqrt().cpu().numpy() if K is None else np.sqrt(np.maximum(np.diag(K),0))
  rec.update(trigger=bool(any(v>0 for v in sol['lam'])),constraints=H.shape[1],active=len(sol['active']),kkt=sol['kkt_error'],projection_residual=sol['written_normalized_rate_max'],
   correction_norm=norm(c),actual_correction_norm=norm(actual-d),native_norm=dn,actual_norm=an,actual_norm_ratio=an/max(dn,1e-30),
   native_positive=int((native_rates>5e-6*hn*dn+1e-15).sum()),actual_positive=int((rates>5e-6*hn*an+1e-15).sum()),native_moments_preserved=True)
  del H,K,before,native,d,c,actual,after;gc.collect()
 rec['seconds']=time.time()-start;guard.check(force=True);return rec

def check_first(name,arm,state):
 expected=read(PRIOR/'measurements'/name/(arm+'_STATE_HASHES.json'))
 actual={k:g.digest(v) for k,v in state.items()}
 assert all(actual[k]==h for k,h in expected.items()),('Prior one-step witness',name,arm)
 return actual

def engineering():
 verify();p=read(OUT/'PROTOCOL_FROZEN.json');SCRATCH.mkdir(exist_ok=True);torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.cuda.get_device_name(0)==DEVICE
 g.init_ddp(OUT/'ddp_engineering');checks=[]
 try:
  for name in p['points']:
   spec,algo,ds,plan,base,Pold,lineage=assets(name);guard=Guard(name)
   for arm in ARMS:
    g.restore(algo,base);update(algo,ds,plan,spec,Pold,lineage,name,arm,1,guard,True);first=g.capture(algo);hh=check_first(name,arm,first)
    path=SCRATCH/'engineering_resume.pt';g.atomic_save(path,first)
    update(algo,ds,plan,spec,Pold,lineage,name,arm,2,guard,True);second=g.capture(algo)
    g.restore(algo,torch.load(path,map_location='cpu'));assert all(g.compare_states(first,g.capture(algo)).values())
    update(algo,ds,plan,spec,Pold,lineage,name,arm,2,guard,True);assert all(g.compare_states(second,g.capture(algo)).values()),('Resume step2 parity',name,arm)
    checks.append(dict(point=name,arm=arm,first_step_exact=True,serialized_resume_step2_exact=True,first_state_hashes=hh));write(OUT/'TRAIN_PREFLIGHT_PROGRESS.json',checks)
    del first,second;path.unlink();gc.collect()
   # Identical-model eval engineering uses original anchor, not a selected good endpoint.
   if name==p['points'][0]:
    path=SCRATCH/'engineering_anchor.pth';g.atomic_save(path,dict(state_dict=base['model']))
    write(OUT/'ENGINEERING_CHECKPOINT.json',dict(path=str(path),sha256=sha(path),model_hash=g.digest(base['model'])))
   del algo,ds,base,Pold,lineage;gc.collect();torch.cuda.empty_cache()
   print('ENGINEERING_ORIGIN_PASS',name,flush=True)
  write(OUT/'TRAIN_PREFLIGHT.json',dict(status='PASS',origins=3,arms=3,checks=checks))
 finally:torch.distributed.destroy_process_group()

def train(name,arm):
 verify();assert read(OUT/'PREFLIGHT_COMPLETE.json')['status']=='PASS';p=read(OUT/'PROTOCOL_FROZEN.json');assert name in p['points'] and arm in ARMS
 out=OUT/'training'/name/arm;dest=SCRATCH/name/arm;dest.mkdir(parents=True,exist_ok=True)
 if (out/'COMPLETE.json').exists():
  r=read(out/'COMPLETE.json');assert sha(r['checkpoint']['path'])==r['checkpoint']['sha256'];print('PRESERVED',name,arm,flush=True);return
 assert not out.exists(),'Interrupted branch requires documented faithful recovery';out.mkdir(parents=True)
 write(out/'STARTED.json',dict(time=time.time(),pid=os.getpid(),point=name,arm=arm))
 torch.set_num_threads(2);torch.set_num_interop_threads(1);assert torch.cuda.get_device_name(0)==DEVICE;g.init_ddp(out/'ddp');started=time.time()
 try:
  spec,algo,ds,plan,base,Pold,lineage=assets(name);guard=Guard(name)
  witnesses={r['horizon']:r['witness'] for r in rows(ROOT/f"reports/functional_geometry_diagnostic_v1/carrier/seed{spec['point']['seed']}/native_witnesses.jsonl") if r['point']==name}
  for h in range(1,301):
   rec=update(algo,ds,plan,spec,Pold,lineage,name,arm,h,guard,h==1)
   if h==1:check_first(name,arm,g.capture(algo))
   if arm=='MLR' and h in [1,5,20]:
    state=g.capture(algo);assert all(g.digest(state[k])==v for k,v in witnesses[h].items()),('Native old-horizon mismatch',h)
   emit(out/'events.jsonl',rec)
   if h%25==0:
    state=g.capture(algo);path=dest/'resume.pt';g.atomic_save(path,dict(state=state,next_horizon=h+1,point=name,arm=arm,protocol_sha256=sha(OUT/'PROTOCOL_FROZEN.json'),membership_sha256=sha(OUT/'TRAINING_MEMBERSHIP.json')))
    write(out/'PROGRESS.json',dict(step=h,total=300,point=name,arm=arm,seconds=time.time()-started,checkpoint=str(path),checkpoint_sha256=sha(path),time=time.time()))
    print('TRAIN_PROGRESS',name,arm,h,flush=True)
  state=g.capture(algo);path=dest/'endpoint.pth';g.atomic_save(path,dict(state_dict=state['model'],horizon=300,full_state_hashes={k:g.digest(v) for k,v in state.items()},protocol_sha256=sha(OUT/'PROTOCOL_FROZEN.json')))
  checkpoint=dict(path=str(path),sha256=sha(path),model_hash=g.digest(state['model']))
  write(out/'COMPLETE.json',dict(status='PASS',point=name,arm=arm,steps=300,checkpoint=checkpoint,full_state_hashes={k:g.digest(v) for k,v in state.items()},seconds=time.time()-started,event_sha256=sha(out/'events.jsonl')))
 finally:torch.distributed.destroy_process_group()
if __name__=='__main__':
 if sys.argv[1]=='engineering':engineering()
 else:train(sys.argv[1],sys.argv[2])
