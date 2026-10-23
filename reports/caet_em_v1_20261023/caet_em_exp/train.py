import argparse,time
from torch.utils.data import DataLoader,DistributedSampler
from .common import *
from .model import Codec,Policy,nll,masked_mean,huber,strict_policy_restore
from .dataset import make,current,retain
from .transfer import build_cache,transfer_loss
from .retention import retain_loss
def load_codec():
 pack=torch.load(frozen_codec(),map_location='cpu');stats=torch.load(ROOT/'shared/normalization.pth',map_location='cpu');codec=Codec(stats);codec.load_state_dict(pack['model'],strict=True);codec=codec.cuda().eval().requires_grad_(False);return pack,codec
def optimizer(p,total):
 from libero.lifelong.utils import make_linear_schedule_with_warmup
 groups={1e-4:[],2.5e-5:[]};names={}
 for n,v in p.named_parameters():
  assert v.requires_grad;nrate=2.5e-5 if p.arm!='B0' and n.startswith('decoder.') else 1e-4;groups[nrate].append(v);names[n]=nrate
 assert len({id(v) for g in groups.values() for v in g})==len(list(p.parameters()))
 opt=torch.optim.AdamW([dict(params=g,lr=lr) for lr,g in groups.items() if g],weight_decay=.1,betas=(.9,.999),eps=1e-8)
 sched=torch.optim.lr_scheduler.LambdaLR(opt,make_linear_schedule_with_warmup(500,total));return opt,sched,names
@torch.no_grad()
def targets(ds,codec):
 for b in DataLoader(ds,32,shuffle=False,num_workers=0):
  b=to_device(b);e=codec(b).cpu()
  for j,i in enumerate(b['index'].tolist()):ds.rows[i]['effect']=e[j];ds.rows[i]['effect_mask']=cpu(b['action_mask'][j]&b['response_mask'][j]&b['visual_mask'][j])
def loss(p,b,codec,ifa_info,refs,task,cache=None):
 from coreset_exp.model import angular_mean
 q,z,hlang=p(b['feature']);bc=nll(q,b['action'],b['action_mask']);mask=b['task']==task
 pairs=ifa_info['selected_pairs'];oldrefs=[refs[j if k==task else k] for j,k in pairs]
 ifa=angular_mean(hlang[mask],refs[task],oldrefs,.3) if mask.any() else hlang.sum()*0
 result=bc+.1*ifa;zero=result*0;le=lt=lf=route=zero;anchors=transfer_steps=retained=switch=0
 if p.arm!='B0':
  target=(b['effect']-p.effect_mean)/p.effect_std;le=masked_mean(huber(z,target),b['effect_mask']);result=result+.1*le
 if cache is not None:
  lt,transfer_steps,anchors=transfer_loss(z,b,cache,task);lf,route,retained,switch=retain_loss(p,z,b,cache,task);result=result+.02*lt+.05*(lf+.1*route)
 stats=dict(bc=float(bc.detach()),ifa=float(ifa.detach()),effect=float(le.detach()),transfer=float(lt.detach()),retention=float(lf.detach()),route=float(route.detach()),action_valid=int(b['action_mask'].sum()),action_invalid=int(b['action_mask'].numel()-b['action_mask'].sum()),effect_valid=int(b['effect_mask'].sum()) if p.arm!='B0' else 0,transfer_anchors=anchors,transfer_steps=transfer_steps,retained_steps=retained,hard_mode_switches=switch,policy_windows=len(b['feature']),extra_retention_decoder_windows=2*len(b['feature']) if cache is not None else 0)
 return result,stats
def main():
 ap=argparse.ArgumentParser();ap.add_argument('arm',choices=ARMS);ap.add_argument('task',type=int,choices=range(6,10));ap.add_argument('--resume',action='store_true');args=ap.parse_args();a,t=args.arm,args.task;contract=verify();capacity();root=arm_root(a);assert js(ROOT/('ADMISSION_%s.json'%a))['status']=='PASS'
 pack,codec=load_codec();seed();p=Policy(a,pack['model'],pack['statistics']).cuda();root.mkdir(parents=True,exist_ok=True)
 if t>6:
  prior=torch.load(root/'models'/('task%d.pth'%(t-1)),map_location='cpu');strict_policy_restore(p,prior,contract,sha(frozen_codec()))
 ds=make(a,t);targets(ds,codec);rows=current(t)
 for r in ds.rows:
  if r['task']==t:rows[r['sample']]=r
 # Keep stage setup and frozen teacher inference out of the formal global RNG stream.
 sampler=DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=0);loader=DataLoader(ds,batch_size=10,num_workers=0,sampler=sampler);assert len(loader)==BATCHES[t]
 from libero.lifelong.algos.mlr import build_ifa_similarity_matrix
 saved=rng();info=build_ifa_similarity_matrix(t,str(PARENT/'features'),str(root/'buffer'));restore_rng(saved)
 refs={}
 for i,r in enumerate(ds.rows):
  if r['task'] not in refs:refs[r['task']]=torch.load(ds.file_paths[i],map_location='cpu')['feature'][0,0,0].cuda()
 manifest=dict(task=t,rows=[dict(index=i,task=r['task'],sample=r['sample'],demo=r['demo'],u=r['u'],sha256=r['sample_sha256'],action_mask=r['action_mask'].tolist(),effect_mask=r['effect_mask'].tolist()) for i,r in enumerate(ds.rows)],ifa={k:v.tolist() if torch.is_tensor(v) else v for k,v in info.items()})
 write(root/'dataset'/('task%d.json'%t),manifest)
 saved=rng();cache=build_cache(a,t,p,codec,ds) if a=='B2' and t>6 else None;restore_rng(saved)
 opt,sched,names=optimizer(p,100*len(loader));write(root/'optimizer'/('task%d.json'%t),dict(parameters=names,total_parameters=sum(v.numel() for v in p.parameters()),steps=100*len(loader)))
 path=root/'recovery/latest.pth';start=0;steps=sum(BATCHES[j]*100 for j in range(6,t));history=[]
 if args.resume:
  s=torch.load(path,map_location='cpu');assert s['contract']==contract and s['task']==t and s['codec_sha256']==sha(frozen_codec()) and s['dataset_sha256']==sha(root/'dataset'/('task%d.json'%t));restore(p,opt,sched,s);start=s['next_epoch'];steps=s['steps'];history=s['history']
 else:
  assert not (root/'models'/('task%d.pth'%t)).exists()
  if t==6:assert not list((root/'buffer').glob('*.pt'))
 for epoch in range(start,100):
  p.train();sampler.set_epoch(epoch);tic=time.time();totals={};order=[]
  for b in loader:
   b=to_device(b);order.extend(b['index'].tolist());opt.zero_grad();value,stat=loss(p,b,codec,info,refs,t,cache);assert torch.isfinite(value)
   value.backward();assert all(v.grad is None or torch.isfinite(v.grad).all() for v in p.parameters());torch.nn.utils.clip_grad_norm_(p.parameters(),100);opt.step();sched.step();steps+=1
   for k,v in stat.items():totals[k]=totals.get(k,0)+v
  assert sorted(order)==list(range(len(ds)))
  row=dict(arm=a,task=t,epoch=epoch,steps=steps,batches=len(loader),population=len(ds),order_sha256=hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest(),lr_after=[g['lr'] for g in opt.param_groups],seconds=time.time()-tic,**totals);history.append(row)
  state=capture(p,opt,sched,contract=contract,codec_sha256=sha(frozen_codec()),task=t,next_epoch=epoch+1,steps=steps,history=history,dataset_sha256=sha(root/'dataset'/('task%d.json'%t)),teacher_sha256=sha(root/'teachers'/('task%d.pth'%t)) if cache else None,cursor=0)
  save(path,state);write(root/'epochs'/('task%d.json'%t),history);event('main_epoch',**row)
 assert len(history)==100 and steps==sum(BATCHES[j]*100 for j in range(6,t+1))
 final=root/'models'/('task%d.pth'%t);save(final,dict(model=cpu(p.state_dict()),arm=a,task=t,epoch=99,contract=contract,codec_sha256=sha(frozen_codec()),statistics_sha256=sha(ROOT/'shared/normalization.pth'),steps=steps))
 retain(a,t,rows)
 write(root/'TASK_COMPLETE'/('task%d.json'%t),dict(status='PASS_TRAINING_NOT_CLOSED_LOOP',task=t,epochs=100,steps=steps,checkpoint_sha256=sha(final),seconds=sum(r['seconds'] for r in history)))
 if t==9:write(root/'TRAINING_COMPLETE.json',dict(status='PASS',epochs=400,steps=105500,codec_sha256=sha(frozen_codec()),checkpoints={str(j):sha(root/'models'/('task%d.pth'%j)) for j in range(6,10)}))
 event('task_complete',arm=a,task=t,steps=steps)
if __name__=='__main__':main()
