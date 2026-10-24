import argparse,time,hashlib,io,pickle
from runtime import *
from losses import cost
from updates import update
import torch.distributed as dist

def prepare(arm,stage,ds,a):
 if not ds.ids:return {},torch.ones(3,device='cuda'),{},None
 saved=rng();modes={n:m.training for n,m in a.policy.named_modules()};tic=time.time();teacher=copy.deepcopy(a.policy.module).eval().requires_grad_(False);cache={}
 source=model_path(arm,stage-1);previous=torch.load(source,map_location='cpu')['state_dict']
 assert equal(a.policy.state_dict(),previous)
 try:
  with torch.no_grad():
   for k in range(0,len(ds.ids),32):
    ids=ds.ids[k:k+32];x=torch.stack([ds.native[i][0]['feature'].squeeze(0) for i in ids]).cuda();q=params(teacher,x)
    for j,i in enumerate(ids):cache[i]=tuple(v[j].cpu() for v in q)
   y=torch.stack([torch.as_tensor(ds.rows[i]['next_action']) for i in ds.ids]).float()
   scales=torch.stack([y[:,lo:hi].square().mean().sqrt().clamp_min(.1) for lo,hi in [(0,3),(3,6),(6,7)]]).cuda()
   q=tuple(torch.stack([cache[i][j] for i in ds.ids]).cuda() for j in range(3));ii=torch.arange(len(y),device='cuda')
   ct=cost(q[0],q[1],y.cuda(),scales)[ii,q[2].argmax(-1)].detach();tau=float(torch.quantile(ct.double(),.5).clamp_min(1e-8));weights=(tau/(tau+ct)).detach()
 finally:
  restore_rng(saved)
 assert equal(saved,rng()) and modes=={n:m.training for n,m in a.policy.named_modules()}
 p=teacher_path(arm,stage);assert not p.exists();save(p,dict(cache=cache,scales=scales.cpu(),weights=weights.cpu(),ids=ds.ids,tau=tau,source_sha256=sha(source)))
 rows=[dict(index=i,sample_sha256=ds.rows[i]['sample_sha256'],task=ds.rows[i]['task'],teacher_cost=float(ct[j]),weight=float(weights[j])) for j,i in enumerate(ds.ids)]
 meta=dict(status='PASS_FROZEN_STAGE_TEACHER',arm=arm,stage=stage,count=len(ds.ids),source_cp=str(source),source_sha256=sha(source),cache_sha256=sha(p),manifest_sha256=sha(O/('ra_coreset/dataset_manifests/task%d.json'%stage)),scales=scales.tolist(),tau=tau,rows=rows,forward_batches=(len(ds.ids)+31)//32,seconds=time.time()-tic,updates=0,environments=0)
 write(R/('TEACHER_%s_%d.json'%(arm,stage)),meta)
 return cache,scales,{i:float(weights[j]) for j,i in enumerate(ds.ids)},meta

def admit(arm,stage,ds,a,cache,scale,weights,info):
 from synthetic_checks import run
 saved=rng();synthetic=run();restore_rng(saved)
 ids=(ds.ids[:6]+[i for i,r in enumerate(ds.rows) if r['task']==stage][:4]) if ds.ids else list(range(10))
 batch=torch.utils.data.default_collate([ds[i] for i in ids]);initial=capture(a)
 s1=update(a,batch,arm,cache,scale,weights,info,stage);one=capture(a)
 stream=io.BytesIO();torch.save(dict(state=one,state_dict=one['model']),stream);size=stream.tell();assert size<350000000;stream.seek(0);loaded=torch.load(stream,map_location='cpu');assert equal(one,loaded['state'])
 s2=update(a,batch,arm,cache,scale,weights,info,stage);two=capture(a)
 restore(a,loaded['state']);s3=update(a,batch,arm,cache,scale,weights,info,stage);assert s2==s3 and equal(two,capture(a))
 calls=3;parity=None;measured=None
 if stage==6:
  restore(a,initial);a.caet_epoch_statistics=[];a.observe(batch,stage,0,info);parity=equal(one,capture(a));assert parity,'Original ALIGN first-stage update mismatch';calls+=1
 else:
  # Test with existing gradient tensors, not only a freshly constructed optimizer.
  restore(a,one);measured=update(a,batch,arm,cache,scale,weights,info,stage,measure=True);assert equal(two,capture(a)),'Stateful diagnostic changed update';calls+=1
  if arm in 'BC':
   restore(a,one);update(a,batch,arm,cache,scale,weights,info,stage,legacy=True);parity=equal(two,capture(a));assert parity;calls+=1
 restore(a,initial)
 p=a.policy.module;p.eval();x=batch[0]['feature'].cuda().squeeze(1)
 with torch.no_grad():
  for n in range(1,9):
   h=p.temporal_encode(x[:,:n])[:,-1];m,s,l=p.policy_head.forward_fn(h);k=l.argmax(-1);ii=torch.arange(len(k),device='cuda');d=p.policy_head(h,train=False)
   assert torch.equal(d.base_dist.loc,m[ii,k]) and torch.equal(d.base_dist.scale,s[ii,k])
 restore(a,initial);assert equal(initial,capture(a))
 write(R/('ADMISSION_%s_%d.json'%(arm,stage)),dict(status='PASS',arm=arm,stage=stage,optimizer_calls=calls,roundtrip_exact=True,next_step_exact=True,restore_initial_exact=True,stateful_gradient_measurement_exact=stage>6,gradient_measurement=measured['gradient_diagnostic'] if measured else None,baseline_or_legacy_update_exact=parity,strict_keys=len(p.state_dict()),parameters=sum(v.numel() for v in p.parameters()),serialized_bytes=size,synthetic=synthetic,lengths=list(range(1,9)),environments=0,contract=verify()))

def initialization(a,arm):
 r=rng();digest={k:hashlib.sha256(pickle.dumps(v) if k in ('python','numpy') else b''.join(x.cpu().numpy().tobytes() for x in (v if isinstance(v,list) else [v]))).hexdigest() for k,v in r.items()}
 tensor={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in a.policy.state_dict().items()}
 expected=js(O/'ra_coreset/INITIAL_STATE.json');assert tensor==expected['tensors'] and digest==expected['rng'],'Official initialization/RNG mismatch'
 write(R/('INITIAL_'+arm+'.json'),dict(status='PASS_OFFICIAL_INITIALIZATION',tensors=tensor,rng=digest,official_cp_sha256=sha(a.cfg.pretrain),parent_initial_sha256=sha(O/'ra_coreset/INITIAL_STATE.json')))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm',choices=list(ARMS));args=ap.parse_args();arm=args.arm;verify();capacity()
 dest=OUTPUT[arm];dest.mkdir(parents=True,exist_ok=True);assert not (dest/'recovery.pth').exists(),'No blind resume'
 dist.init_process_group('nccl');torch.cuda.set_device(0)
 try:
  a=make(arm);initialization(a,arm);history=[];gradients=[]
  for stage in range(6,10):
   write(R/'TRAIN_STATUS.json',dict(arm=arm,stage=stage,phase='prepare',time=time.time(),pid=os.getpid()))
   ds=Dataset(stage);loader,sampler=setup(a,ds);info=js(O/('ra_coreset/ifa_task%d.json'%stage))['info'];a.policy.train()
   cache,scale,weights,teacher=prepare(arm,stage,ds,a);admit(arm,stage,ds,a,cache,scale,weights,info)
   for ep in range(100):
    capacity();sampler.set_epoch(ep);order=list(iter(sampler));a.policy.train();tic=time.time();tot={k:0. for k in ['loss','bc','ifa','aux','exec_loss','route_loss','valid','invalid','old_eligible']};aux_tot={};lr=a.optimizer.param_groups[0]['lr'];measured=False
    for bi,batch in enumerate(loader):
     st=update(a,batch,arm,cache,scale,weights,info,stage,measure=stage>6 and ep in [0,9,49,99] and not measured)
     for k in tot:tot[k]+=st[k]
     for k,v in st['aux_stats'].items():aux_tot[k]=aux_tot.get(k,0)+v
     if st['gradient_diagnostic'] is not None:
      measured=True;gradients.append(dict(stage=stage,epoch=ep,batch=bi,steps=a.step,**st['gradient_diagnostic']));write(R/('GRADIENTS_'+arm+'.json'),gradients)
    assert tot['valid']==VALID[stage] and tot['invalid']==POP[stage]-VALID[stage] and tot['old_eligible']==OLD[stage]
    row=dict(stage=stage,epoch=ep,steps=a.step,batches=len(loader),seconds=time.time()-tic,lr_first=lr,lr_after=a.optimizer.param_groups[0]['lr'],sampler_sha256=hashlib.sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest(),totals=tot,aux_totals=aux_tot)
    history.append(row);state=capture(a)
    save(dest/'recovery.pth',dict(state=state,state_dict=state['model'],stage=stage,next_epoch=ep+1,steps=a.step,contract=verify(),teacher_sha256=teacher['cache_sha256'] if teacher else None,teacher_metadata_sha256=sha(R/('TEACHER_%s_%d.json'%(arm,stage))) if teacher else None,manifest_sha256=sha(O/('ra_coreset/dataset_manifests/task%d.json'%stage)),history=history))
    write(R/('EPOCHS_'+arm+'.json'),history);write(R/'TRAIN_STATUS.json',dict(arm=arm,stage=stage,phase='train',epoch_complete=ep+1,steps=a.step,time=time.time(),pid=os.getpid()));print(json.dumps(dict(arm=arm,**row)),flush=True)
   target=model_path(arm,stage);target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists()
   if stage==9:os.link(dest/'recovery.pth',target)
   else:save(target,dict(state_dict=cpu(a.policy.state_dict()),stage=stage,epoch=99,steps=a.step,contract=verify()))
   if stage==6:
    baseline=torch.load(O/'ra_coreset/models/mlrifa_6_model_ep99.pth',map_location='cpu')['state_dict'];same=equal(cpu(a.policy.state_dict()),baseline)
    write(R/('TASK6_PARITY_'+arm+'.json'),dict(equal=same,source_sha256=sha(O/'ra_coreset/models/mlrifa_6_model_ep99.pth'),actual_sha256=sha(target)));assert same,'Task6 must reproduce original ALIGN'
   write(R/('STAGE_%s_%d.json'%(arm,stage)),dict(status='PASS_TRAINING_PENDING_FINAL',model=str(target),sha256=sha(target),steps=a.step,epochs=100,stage=stage,teacher=teacher['cache_sha256'] if teacher else None))
  assert a.step==76100 and len(history)==400
  write(R/('TRAIN_'+arm+'.json'),dict(status='PASS',epochs=400,steps=a.step,summed_epoch_seconds=sum(r['seconds'] for r in history),gradient_autograd_calls=sum(r['extra_autograd_calls'] for r in gradients),checkpoints={str(t):sha(model_path(arm,t)) for t in range(6,10)}))
 finally:dist.destroy_process_group()
if __name__=='__main__':main()
