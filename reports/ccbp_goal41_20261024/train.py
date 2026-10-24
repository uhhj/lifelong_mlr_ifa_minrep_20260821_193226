import argparse,time,hashlib,io
from runtime import *
from losses import aux,cost,route
import torch.distributed as dist
def prepare(ds,a):
 teacher=copy.deepcopy(a.policy.module).eval().requires_grad_(False);cache={};ys=[];saved=rng();tic=time.time()
 with torch.no_grad():
  for k in range(0,len(ds.ids),32):
   ids=ds.ids[k:k+32];x=torch.stack([ds.native[i][0]['feature'].squeeze(0) for i in ids]).cuda()
   q=params(teacher,x)
   for j,i in enumerate(ids):cache[i]=tuple(v[j].cpu() for v in q);ys.append(torch.as_tensor(ds.rows[i]['next_action']))
 y=torch.stack(ys).float();scales=torch.stack([y[:,lo:hi].square().mean().sqrt().clamp_min(.1) for lo,hi in [(0,3),(3,6),(6,7)]])
 restore_rng(saved);assert all(not p.requires_grad for p in teacher.parameters())
 p=P/'teacher_cache.pth';assert not p.exists();save(p,dict(cache=cache,scales=scales))
 write(R/'TEACHER.json',dict(status='PASS',count=len(cache),forward_batches=(len(cache)+31)//32,seconds=time.time()-tic,cache_sha256=sha(p),cp_sha256=sha(O/'ra_coreset/models/mlrifa_8_model_ep99.pth'),scales=scales.tolist(),manifest_sha256=sha(O/'ra_coreset/dataset_manifests/task9.json'),label_sha256=sha(O/'effects/memory/task8.pth'),optimizer_updates=0))
def update(a,batch,arm,cache,scale,info):
 x=a.map_tensor_to_device(batch[0]);tasks=torch.tensor([int(v) for v in batch[1]],device='cuda');a.optimizer.zero_grad()
 h=hidden(a.policy.module,x['feature']);bc,valid=masked_bc(a.policy.module,h,x,True)
 mask=tasks==9;olds=[a.refs[j if k==9 else k] for j,k in info['selected_pairs']]
 ifa=angular_mean(h[mask],a.refs[9],olds,a.cfg.gamma) if mask.any() else h.sum()*0
 eligible=(tasks==8)&valid;extra=h.sum()*0;st=dict(count=0)
 if arm!='A' and eligible.any():
  saved=rng();modes={n:m.training for n,m in a.policy.named_modules()};a.policy.eval()
  try:
   q=params(a.policy.module,x['feature'][eligible]);ids=x['v3_index'][eligible].tolist()
   teacher=tuple(torch.stack([cache[i][j] for i in ids]).cuda() for j in range(3))
   extra,st=aux(arm,q,x['next_action'][eligible],teacher,scale)
  finally:
   for n,m in a.policy.named_modules():m.training=modes[n]
   restore_rng(saved)
 loss=bc+.1*ifa+.1*extra;(a.loss_scale*loss).backward()
 assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in a.policy.parameters())
 torch.nn.utils.clip_grad_norm_(a.policy.parameters(),a.cfg.train.grad_clip);a.optimizer.step();a.scheduler.step();a.step+=1
 return dict(loss=float(loss.detach()),bc=float(bc.detach()),ifa=float(ifa.detach()),aux=float(extra.detach()),valid=int(valid.sum()),invalid=int((~valid).sum()),old_eligible=int(eligible.sum()),aux_stats=st)
def admit(arm,ds,a,loader,cache,scales,info):
 selected=[i for i in ds.ids[:6]]+[i for i,r in enumerate(ds.rows) if r['task']==9][:4]
 batch=torch.utils.data.default_collate([ds[i] for i in selected]);a.policy.train();initial=capture(a)
 s1=update(a,batch,arm,cache,scales,info);one=capture(a)
 stream=io.BytesIO();torch.save(one,stream);stream.seek(0);loaded=torch.load(stream,map_location='cpu');assert equal(one,loaded)
 s2=update(a,batch,arm,cache,scales,info);two=capture(a)
 restore(a,loaded);s3=update(a,batch,arm,cache,scales,info);assert equal(two,capture(a)) and s2==s3
 parity=None
 if arm=='A':
  restore(a,initial);a.caet_epoch_statistics=[];a.observe(batch,9,0,info);parity=equal(one,capture(a));assert parity,'Original ALIGN update mismatch'
 # Synthetic membership semantics/gradients, no optimizer or environment calls.
 z=torch.tensor([[0.,1.,2.,3.,4.]],device='cuda',requires_grad=True)
 assert float(route(z,torch.ones_like(z,dtype=torch.bool)))==0
 rr=route(z,torch.tensor([[True,False,False,False,False]],device='cuda'));rr.sum().backward();assert z.grad[0,0]<0 and z.grad[0,4]>0
 yy=torch.zeros(2,7,device='cuda');mm=torch.ones(2,5,7,device='cuda',requires_grad=True);ss=torch.ones_like(mm,requires_grad=True);ll=torch.zeros(2,5,device='cuda',requires_grad=True)
 tt=(torch.zeros_like(mm),torch.ones_like(ss)*.01,torch.zeros_like(ll))
 loss,stats=aux('D',(mm,ss,ll),yy,tt,torch.ones(3,device='cuda'));assert stats['empty']==2;loss.backward();assert mm.grad.abs().sum()>0 and ss.grad.abs().sum()>0
 p=a.policy.module;p.eval();x=batch[0]['feature'].cuda().squeeze(1)
 with torch.no_grad():
  for n in range(1,9):
   h=hidden(p,x[:,:n]) if n==8 else p.temporal_encode(x[:,:n])[:,-1]
   m,s,l=p.policy_head.forward_fn(h);k=l.argmax(-1);ix=torch.arange(len(k),device='cuda');d=p.policy_head(h,train=False)
   assert torch.equal(d.base_dist.loc,m[ix,k]) and torch.equal(d.base_dist.scale,s[ix,k])
 restore(a,initial)
 write(R/('ADMISSION_'+arm+'.json'),dict(status='PASS',optimizer_calls=4 if arm=='A' else 3,extra_original_ALIGN_call=arm=='A',roundtrip_exact=True,next_step_exact=True,original_ALIGN_update_exact=parity,strict_keys=len(p.state_dict()),parameters=sum(p.numel() for p in p.parameters()),lengths=list(range(1,9)),routing_empty_allgood_gradients=True,environments=0,contract=verify()))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','admit','train']);ap.add_argument('arm',choices=list('ABCD'));args=ap.parse_args();verify();capacity()
 dist.init_process_group('nccl');torch.cuda.set_device(0)
 try:
  ds=Dataset();a,loader,sampler=make(args.arm,ds);info=js(O/'ra_coreset/ifa_task9.json')['info']
  if args.phase=='prepare':prepare(ds,a);return
  assert sha(P/'teacher_cache.pth')==js(R/'TEACHER.json')['cache_sha256']
  tc=torch.load(P/'teacher_cache.pth',map_location='cpu');cache=tc['cache'];scale=tc['scales'].cuda()
  if args.phase=='admit':admit(args.arm,ds,a,loader,cache,scale,info);return
  assert js(R/('ADMISSION_'+args.arm+'.json'))['status']=='PASS'
  dest=P/args.arm;assert not (dest/'recovery.pth').exists(),'Explicit versioned recovery only';history=[]
  init=capture(a);write(R/('INITIAL_'+args.arm+'.json'),dict(tensors={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in init['model'].items()},rng_cpu=hashlib.sha256(init['rng']['cpu'].numpy().tobytes()).hexdigest(),sampler_seed=sampler.seed,teacher_cache_sha256=sha(P/'teacher_cache.pth'),input_manifest_sha256=sha(O/'ra_coreset/dataset_manifests/task9.json')))
  for ep in range(100):
   capacity();sampler.set_epoch(ep);order=list(iter(sampler));a.policy.train();tic=time.time();tot={k:0. for k in ['loss','bc','ifa','aux','valid','invalid','old_eligible']};aux_tot={};lr=a.optimizer.param_groups[0]['lr']
   for batch in loader:
    st=update(a,batch,args.arm,cache,scale,info)
    for k in tot:tot[k]+=st[k]
    for k,v in st['aux_stats'].items():aux_tot[k]=aux_tot.get(k,0)+v
   row=dict(epoch=ep,steps=a.step,batches=len(loader),seconds=time.time()-tic,lr_first=lr,lr_after=a.optimizer.param_groups[0]['lr'],sampler_sha256=hashlib.sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest(),totals=tot,aux_totals=aux_tot)
   history.append(row)
   save(dest/'recovery.pth',dict(state=capture(a),next_epoch=ep+1,contract=verify(),teacher_sha256=sha(P/'teacher_cache.pth'),manifest_sha256=sha(O/'ra_coreset/dataset_manifests/task9.json')))
   write(R/('EPOCHS_'+args.arm+'.json'),history);print(json.dumps(dict(arm=args.arm,**row)),flush=True)
  assert a.step==33700
  save(dest/'stage9.pth',dict(state_dict=cpu(a.policy.state_dict()),contract=verify(),epoch=99,steps=a.step))
  write(R/('TRAIN_'+args.arm+'.json'),dict(status='PASS',epochs=100,steps=a.step,cp_sha256=sha(dest/'stage9.pth'),recovery_sha256=sha(dest/'recovery.pth'),summed_epoch_seconds=sum(r['seconds'] for r in history)))
 finally:dist.destroy_process_group()
if __name__=='__main__':main()
