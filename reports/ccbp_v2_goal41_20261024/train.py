import argparse,time,hashlib,io
from runtime import *
from losses import aux,cost
from legacy_losses import aux as legacy_aux
import torch.distributed as dist
def prepare(ds,a):
 prior=js(PRIOR_ROOT/'TEACHER.json');assert sha(TEACHER_CACHE)==prior['cache_sha256']
 tc=torch.load(TEACHER_CACHE,map_location='cpu');assert set(tc['cache'])==set(ds.ids)
 teacher=tuple(torch.stack([tc['cache'][i][j] for i in ds.ids]).cuda() for j in range(3));y=torch.stack([torch.as_tensor(ds.rows[i]['next_action']) for i in ds.ids]).cuda().float();scale=tc['scales'].cuda()
 ix=torch.arange(len(y),device='cuda');ct=cost(teacher[0],teacher[1],y,scale)[ix,teacher[2].argmax(-1)].detach();tau=float(torch.quantile(ct.double(),.5).clamp_min(1e-8));assert tau>0
 w=(tau/(tau+ct)).detach();assert torch.isfinite(w).all() and ((w>0)&(w<=1)).all()
 weights=dict(tau=tau,rows=[dict(index=i,teacher_cost=float(ct[j]),weight=float(w[j]),sample_sha256=ds.rows[i]['sample_sha256']) for j,i in enumerate(ds.ids)],cache_sha256=sha(TEACHER_CACHE),median_definition='prototype float64 quantile, average of two central values, numerical floor 1e-8')
 write(R/'WEIGHTS.json',weights);write(R/'TEACHER.json',dict(prior,prior_metadata_sha256=sha(PRIOR_ROOT/'TEACHER.json'),reused_readonly=True,new_forward_batches=0,weights_sha256=sha(R/'WEIGHTS.json'),tau=float(tau),weight_min=float(w.min()),weight_max=float(w.max()),weight_mean=float(w.mean())))
def update(a,batch,arm,cache,scale,weights,info,measure=False,legacy=False):
 x=a.map_tensor_to_device(batch[0]);tasks=torch.tensor([int(v) for v in batch[1]],device='cuda');a.optimizer.zero_grad()
 h=hidden(a.policy.module,x['feature']);bc,valid=masked_bc(a.policy.module,h,x,True)
 mask=tasks==9;olds=[a.refs[j if k==9 else k] for j,k in info['selected_pairs']]
 ifa=angular_mean(h[mask],a.refs[9],olds,a.cfg.gamma) if mask.any() else h.sum()*0
 eligible=(tasks==8)&valid;extra=h.sum()*0;ex=extra;rr=extra;st=dict(count=0)
 if eligible.any():
  saved=rng();modes={n:m.training for n,m in a.policy.named_modules()};a.policy.eval()
  try:
   q=params(a.policy.module,x['feature'][eligible]);ids=x['v3_index'][eligible].tolist();teacher=tuple(torch.stack([cache[i][j] for i in ids]).cuda() for j in range(3));w=torch.tensor([weights[i] for i in ids],device='cuda')
   extra,st,ex,rr=aux(arm,q,x['next_action'][eligible],teacher,scale,w)
   if legacy:extra,_=legacy_aux(arm,q,x['next_action'][eligible],teacher,scale)
  finally:
   for n,m in a.policy.named_modules():m.training=modes[n]
   restore_rng(saved)
 grad=None
 if measure and eligible.any():
  saved=rng();pars=[p for p in a.policy.parameters() if p.requires_grad];grad={};tic=time.time()
  for name,term in [('bc',bc),('ifa_weighted',.1*ifa),('exec_weighted',.1*ex),('route_weighted',.01*rr)]:
   gs=torch.autograd.grad(a.loss_scale*term,pars,retain_graph=True,allow_unused=True)
   grad[name]=dict(loss=float(term.detach()),l2=float(torch.sqrt(sum((g.double().square().sum() for g in gs if g is not None),torch.zeros((),device='cuda',dtype=torch.float64)))),nonzero_tensors=sum(g is not None and bool(g.abs().sum()>0) for g in gs))
  assert all(p.grad is None for p in pars) and equal(saved,rng());grad['seconds']=time.time()-tic;grad['extra_autograd_calls']=4
 loss=bc+.1*ifa+.1*extra;(a.loss_scale*loss).backward()
 assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in a.policy.parameters())
 torch.nn.utils.clip_grad_norm_(a.policy.parameters(),a.cfg.train.grad_clip);a.optimizer.step();a.scheduler.step();a.step+=1
 return dict(loss=float(loss.detach()),bc=float(bc.detach()),ifa=float(ifa.detach()),aux=float(extra.detach()),exec_loss=float(ex.detach()),route_loss=float(rr.detach()),valid=int(valid.sum()),invalid=int((~valid).sum()),old_eligible=int(eligible.sum()),aux_stats=st,gradient_diagnostic=grad)
def admit(arm,ds,a,loader,cache,scale,weights,info):
 from synthetic_checks import run
 saved=rng();synthetic=run();restore_rng(saved)
 ids=ds.ids[:6]+[i for i,r in enumerate(ds.rows) if r['task']==9][:4];batch=torch.utils.data.default_collate([ds[i] for i in ids]);a.policy.train();initial=capture(a)
 s1=update(a,batch,arm,cache,scale,weights,info);one=capture(a);stream=io.BytesIO();torch.save(dict(state=one,state_dict=one['model']),stream);serialized_bytes=stream.tell();assert serialized_bytes<350000000;stream.seek(0);pack=torch.load(stream,map_location='cpu');loaded=pack['state'];assert equal(one,loaded) and all(pack['state_dict'][n].data_ptr()==loaded['model'][n].data_ptr() for n in pack['state_dict'])
 s2=update(a,batch,arm,cache,scale,weights,info);two=capture(a);restore(a,loaded);s3=update(a,batch,arm,cache,scale,weights,info);assert equal(two,capture(a)) and s2==s3
 restore(a,initial);measured=update(a,batch,arm,cache,scale,weights,info,measure=True);assert equal(one,capture(a)),'Gradient measurement changed optimizer result'
 parity=None
 if arm in 'BC':
  restore(a,initial);update(a,batch,arm,cache,scale,weights,info,legacy=True);parity=equal(one,capture(a));assert parity
 p=a.policy.module;p.eval();x=batch[0]['feature'].cuda().squeeze(1)
 with torch.no_grad():
  for n in range(1,9):
   hh=hidden(p,x[:,:n]) if n==8 else p.temporal_encode(x[:,:n])[:,-1];m,s,l=p.policy_head.forward_fn(hh);k=l.argmax(-1);ii=torch.arange(len(k),device='cuda');d=p.policy_head(hh,train=False)
   assert torch.equal(d.base_dist.loc,m[ii,k]) and torch.equal(d.base_dist.scale,s[ii,k])
 restore(a,initial)
 write(R/('ADMISSION_'+arm+'.json'),dict(status='PASS',optimizer_calls=5 if arm in 'BC' else 4,roundtrip_exact=True,next_step_exact=True,shared_storage_serialization_bytes=serialized_bytes,shared_storage_alias_exact=True,gradient_measurement_update_exact=True,legacy_control_update_exact=parity,gradient_diagnostic=measured['gradient_diagnostic'],synthetic=synthetic,strict_keys=len(p.state_dict()),parameters=sum(p.numel() for p in p.parameters()),lengths=list(range(1,9)),environments=0,contract=verify()))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['prepare','admit','train']);ap.add_argument('arm',choices=list(ARMS));args=ap.parse_args();verify();capacity()
 dist.init_process_group('nccl');torch.cuda.set_device(0)
 try:
  ds=Dataset();a,loader,sampler=make(args.arm,ds);info=js(O/'ra_coreset/ifa_task9.json')['info']
  if args.phase=='prepare':prepare(ds,a);return
  assert sha(TEACHER_CACHE)==js(R/'TEACHER.json')['cache_sha256'];tc=torch.load(TEACHER_CACHE,map_location='cpu');cache=tc['cache'];scale=tc['scales'].cuda();wr=js(R/'WEIGHTS.json');weights={r['index']:r['weight'] for r in wr['rows']};assert set(weights)==set(ds.ids)
  if args.phase=='admit':admit(args.arm,ds,a,loader,cache,scale,weights,info);return
  assert js(R/('ADMISSION_'+args.arm+'.json'))['status']=='PASS';dest=OUTPUT[args.arm];dest.mkdir(parents=True,exist_ok=True);assert not (dest/'recovery.pth').exists();history=[];gd=[]
  init=capture(a);write(R/('INITIAL_'+args.arm+'.json'),dict(tensors={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in init['model'].items()},rng_cpu=hashlib.sha256(init['rng']['cpu'].numpy().tobytes()).hexdigest(),sampler_seed=0,teacher_cache_sha256=sha(TEACHER_CACHE),input_manifest_sha256=sha(O/'ra_coreset/dataset_manifests/task9.json'),weights_sha256=sha(R/'WEIGHTS.json')))
  for ep in range(100):
   capacity();sampler.set_epoch(ep);order=list(iter(sampler));a.policy.train();tic=time.time();tot={k:0. for k in ['loss','bc','ifa','aux','exec_loss','route_loss','valid','invalid','old_eligible']};aux_tot={};lr=a.optimizer.param_groups[0]['lr'];measured=False
   for batch_index,batch in enumerate(loader):
    st=update(a,batch,args.arm,cache,scale,weights,info,measure=ep in [0,9,49,99] and not measured)
    for k in tot:tot[k]+=st[k]
    for k,v in st['aux_stats'].items():aux_tot[k]=aux_tot.get(k,0)+v
    if st['gradient_diagnostic'] is not None:measured=True;gd.append(dict(epoch=ep,batch=batch_index,steps=a.step,**st['gradient_diagnostic']));write(R/('GRADIENTS_'+args.arm+'.json'),gd)
   history.append(dict(epoch=ep,steps=a.step,batches=len(loader),seconds=time.time()-tic,lr_first=lr,lr_after=a.optimizer.param_groups[0]['lr'],sampler_sha256=hashlib.sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest(),totals=tot,aux_totals=aux_tot))
   state=capture(a);save(dest/'recovery.pth',dict(state=state,state_dict=state['model'],next_epoch=ep+1,epoch=ep,steps=a.step,contract=verify(),teacher_sha256=sha(TEACHER_CACHE),weights_sha256=sha(R/'WEIGHTS.json'),manifest_sha256=sha(O/'ra_coreset/dataset_manifests/task9.json')))
   write(R/('EPOCHS_'+args.arm+'.json'),history);print(json.dumps(dict(arm=args.arm,**history[-1])),flush=True)
  assert a.step==33700;os.link(dest/'recovery.pth',dest/'stage9.pth');assert (dest/'stage9.pth').stat().st_ino==(dest/'recovery.pth').stat().st_ino
  write(R/('TRAIN_'+args.arm+'.json'),dict(status='PASS',epochs=100,steps=a.step,cp_sha256=sha(dest/'stage9.pth'),recovery_sha256=sha(dest/'recovery.pth'),storage='hardlinked final immutable recovery; state_dict alias shares tensor storage',summed_epoch_seconds=sum(r['seconds'] for r in history),gradient_autograd_calls=sum(r['extra_autograd_calls'] for r in gd)))
 finally:dist.destroy_process_group()
if __name__=='__main__':main()
