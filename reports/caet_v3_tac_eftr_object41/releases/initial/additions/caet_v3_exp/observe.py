import copy,json,time
import torch
from torch.nn import functional as F
from coreset_exp.common import O,SEED,sha,write,rng,restore_rng
from coreset_exp.train import PaperMLR
from coreset_exp.model import angular_mean
from .variant import VARIANT
from .model import Auxiliary,hidden,masked_bc,params,selected,kl
from . import eftr_optimizer as transaction

class V3MLR(PaperMLR):
 def configure_effect(self,t,ds):
  self.ds=ds;self.task=t;self.caet_epoch_statistics=[];self.cursor=0;self.aux=None;self.auxopt=None;self.head_steps=0;self.teacher={};self.teacher_hash={};self.old=[];self.disable_all=False
  self.support_sha256=sha(O/'effects/current'/('task%d'%t)/'SUPPORT.json');saved=rng()
  try:
   if VARIANT=='TAC':
    torch.manual_seed(930041);self.aux=Auxiliary().cuda();self.auxopt=torch.optim.AdamW(self.aux.parameters(),lr=3e-4,weight_decay=.01)
    if t>6:
     state=torch.load(O/'ra_coreset/aux_heads'/('task%d.pth'%(t-1)),map_location='cpu');self.load_aux(state)
   if VARIANT=='EFTR':
    for j in range(6,t):
     p=O/'teachers'/('task%d.pth'%j);meta=json.loads(p.with_suffix('.json').read_text())
     assert sha(p)==meta['sha256'] and sha(O/'ra_coreset/models'/('mlrifa_%d_model_ep99.pth'%j))==meta['checkpoint_sha256']
     self.teacher_hash[str(j)]=sha(p);self.teacher.update(torch.load(p,map_location='cpu'))
    tasks={j:sorted([i for i,r in enumerate(ds.rows) if r['task']==j],key=lambda i:ds.rows[i]['sample']) for j in range(6,t)}
    self.old=[v[k] for k in range(max([len(v) for v in tasks.values()] or [0])) for j,v in sorted(tasks.items()) if k<len(v)]
    for i in self.old:
     name='t%d__%s'%(ds.rows[i]['task'],ds.rows[i]['sample']);assert self.teacher[name]['sample_sha256']==ds.rows[i]['sample_sha256']
  finally:restore_rng(saved)
 def aux_state(self):
  if self.aux is None:return None
  return copy.deepcopy(dict(model=self.aux.state_dict(),optimizer=self.auxopt.state_dict(),grads={n:p.grad for n,p in self.aux.named_parameters()},modes={n:m.training for n,m in self.aux.named_modules()},requires_grad={n:p.requires_grad for n,p in self.aux.named_parameters()},steps=self.head_steps))
 def load_aux(self,s):
  self.aux.load_state_dict(s['model'],strict=True);self.auxopt.load_state_dict(s['optimizer']);self.head_steps=s['steps']
  for n,p in self.aux.named_parameters():p.grad=None if s['grads'][n] is None else s['grads'][n].to(p.device);p.requires_grad_(s['requires_grad'][n])
  for n,m in self.aux.named_modules():m.training=s['modes'][n]
 def export_v2_state(self):
  return dict(variant=VARIANT,support_sha256=self.support_sha256,teacher_hash=self.teacher_hash,cursor=self.cursor,aux=self.aux_state(),requires_grad={n:p.requires_grad for n,p in self.policy.named_parameters()})
 def import_v2_state(self,s):
  assert s['variant']==VARIANT and s['support_sha256']==self.support_sha256 and s['teacher_hash']==self.teacher_hash
  self.cursor=s['cursor']
  for n,p in self.policy.named_parameters():p.requires_grad_(s['requires_grad'][n])
  if self.aux is not None:self.load_aux(s['aux'])
 def teacher_batch(self,ids):
  rows=[self.teacher['t%d__%s'%(self.ds.rows[i]['task'],self.ds.rows[i]['sample'])] for i in ids]
  return tuple(torch.stack([r[k] for r in rows]).cuda() for k in ('means','std','logits'))
 def old_batch(self):
  if not self.old:return [],None,None
  ids=[self.old[(self.cursor+k)%len(self.old)] for k in range(min(32,len(self.old)))];self.cursor=(self.cursor+len(ids))%len(self.old)
  return ids,self.ds.features(ids),self.teacher_batch(ids)
 def cross(self,x,h):
  losses=[];exposures=0;anchor_ids=[];partners=[]
  for b,i in enumerate(x['v3_index'].tolist()):
   pos=self.ds.positive.get(i,[]);neg=self.ds.negative.get(i,[])
   if not pos or not neg:continue
   ps=[pos[self.step%len(pos)]];ns=[neg[(self.step+k)%len(neg)] for k in range(min(4,len(neg)))];ids=ps+ns
   partner_h=hidden(self.policy.module,self.ds.features(ids));z=self.aux.embedding(h[b:b+1]);zz=self.aux.embedding(partner_h)
   scores=(z@zz.T).squeeze(0)/.1;losses.append(torch.logsumexp(scores,0)-scores[0]);exposures+=len(ids)
   anchor_ids.append(i);partners.append(ids)
  return (torch.stack(losses).mean() if losses else h.sum()*0),exposures,anchor_ids,partners
 @torch.no_grad()
 def score(self,x,mask,oldx,teacher):
  h=hidden(self.policy.module,x['feature'])
  if mask.any():
   xx={k:v[mask] for k,v in x.items() if torch.is_tensor(v)};c=float(masked_bc(self.policy.module,h[mask],xx,False)[0])
  else:c=None
  ref=selected(teacher);now=selected(params(self.policy.module,oldx));kv=float(kl(ref,now).mean())/7;sw=float((ref[2]!=now[2]).float().mean());gr=float((ref[0][:,-1]-now[0][:,-1]).abs().mean())
  return dict(C=c,R=kv+.1*sw,KL_per_dim=kv,switch=sw,gripper_mean_abs=gr)
 def observe(self,data,current_task_id,epoch,ifa_info=None):
  tic=time.monotonic();step=self.step
  if self.disable_all or (VARIANT=='EFTR' and not self.old):
   loss=PaperMLR.observe(self,data,current_task_id,epoch,ifa_info)
   self.caet_epoch_statistics.append(dict(step=step,variant=VARIANT,batch_ids=data[0]['v3_index'].tolist(),native_exact=True,bc_valid=len(data[1]),bc_invalid=0,effect_samples=0,cross_anchors=0,extra_partner_forwards=0,head_updates=0,optimizer_trials=1,committed_updates=1,seconds=time.monotonic()-tic));return loss
  x=self.map_tensor_to_device(data[0]);self.optimizer.zero_grad();mask=data[1].to(device=x['feature'].device)==current_task_id
  if VARIANT=='EFTR':bc,h=self.policy.module.compute_loss(x,None,None);valid=torch.ones(len(h),device=h.device,dtype=torch.bool)
  else:h=hidden(self.policy.module,x['feature']);bc,valid=masked_bc(self.policy.module,h,x,True)
  olds=[self.refs[j if k==current_task_id else k] for j,k in ifa_info['selected_pairs']]
  ifa=angular_mean(h[mask],self.refs[current_task_id],olds,self.cfg.gamma) if mask.any() else h.sum()*0
  native=bc+.1*ifa;effect=None;cross=None;exposures=0;anchors=[];partners=[];head_calls=0
  stats=dict(step=step,variant=VARIANT,batch_ids=x['v3_index'].tolist(),bc=float(bc.detach()),ifa=float(ifa.detach()),bc_valid=int(valid.sum()),bc_invalid=int((~valid).sum()),bc_current_valid=int((valid&mask).sum()),effect_samples=0,cross_anchors=0,extra_partner_forwards=0,head_updates=0,committed_updates=1)
  if VARIANT=='TAC':
   self.auxopt.zero_grad();em=x['effect_valid'].bool();aux=h.sum()*0
   if em.any():
    scale=h.new_tensor([.01]*3+[torch.pi/18]*3+[.01]);effect=F.smooth_l1_loss(self.aux.predict(h[em],x['next_action'][em]),x['effect_target'][em]/scale);aux=aux+.03*effect
   saved=rng()
   try:
    torch.manual_seed(840000000+SEED*1000000+self.step);cross,exposures,anchors,partners=self.cross(x,h);aux=aux+.02*cross
   finally:restore_rng(saved)
   loss=native+aux;(self.loss_scale*loss).backward()
   assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(self.policy.parameters())+list(self.aux.parameters()))
   if self.cfg.train.grad_clip is not None:
    torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.train.grad_clip);torch.nn.utils.clip_grad_norm_(self.aux.parameters(),self.cfg.train.grad_clip)
   self.optimizer.step();self.scheduler.step();self.step+=1
   if em.any() or anchors:self.auxopt.step();self.head_steps+=1;head_calls=1
   stats.update(effect_samples=int(em.sum()),effect_loss=float(effect.detach()) if effect is not None else None,cross_loss=float(cross.detach()),cross_anchors=len(anchors),extra_partner_forwards=exposures,anchor_ids=anchors,partner_ids=partners,head_updates=head_calls,head_steps=self.head_steps,optimizer_trials=1)
  elif VARIANT=='ALIGN':
   loss=native;(self.loss_scale*loss).backward()
   if self.cfg.train.grad_clip is not None:torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.train.grad_clip)
   self.optimizer.step();self.scheduler.step();self.step+=1;stats.update(optimizer_trials=1)
  else:
   main_params=[p for p in self.policy.parameters() if p.requires_grad];ng=torch.autograd.grad(self.loss_scale*native,main_params,allow_unused=True)
   ids,oldx,teacher=self.old_batch();saved=rng();modes={n:m.training for n,m in self.policy.named_modules()};self.policy.eval()
   try:
    ref=selected(teacher);now=params(self.policy.module,oldx);ret=kl(ref,selected(now,ref[2])).mean()
    logits=now[2];winner=ref[2];ix=torch.arange(len(winner),device=winner.device);other=logits.clone();other[ix,winner]=-torch.inf
    mode=F.relu(.1+other.max(-1).values-logits[ix,winner]).mean();aux=.05*ret+.01*mode
    ag=torch.autograd.grad(self.loss_scale*aux,main_params,allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in [*ng,*ag])
    choice=transaction.candidates(self,ng,ag,lambda:self.score(x,mask,oldx,teacher))
   finally:
    for n,m in self.policy.named_modules():m.training=modes[n]
    restore_rng(saved)
   stats.update(choice,retention_loss=float(ret.detach()),mode_loss=float(mode.detach()),old_check_ids=ids,old_check_count=len(ids),cursor=self.cursor)
   loss=native+choice['scale']*aux
  assert self.step==step+1 and torch.isfinite(loss)
  stats['seconds']=time.monotonic()-tic;self.caet_epoch_statistics.append(stats);return float(loss.detach())
 @torch.no_grad()
 def finish_epoch(self,t,epoch):
  if VARIANT!='EFTR' or not self.old:
   write(O/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,epoch)),dict(active=False,variant=VARIANT));return
  saved=rng();modes={n:m.training for n,m in self.policy.named_modules()};self.policy.eval();total=0.;switch=0.;count=0
  try:
   for start in range(0,len(self.old),32):
    ids=self.old[start:start+32];ref=selected(self.teacher_batch(ids));now=selected(params(self.policy.module,self.ds.features(ids)))
    total+=float(kl(ref,now).double().sum())/7;switch+=float((ref[2]!=now[2]).double().sum());count+=len(ids)
  finally:
   for n,m in self.policy.named_modules():m.training=modes[n]
   restore_rng(saved)
  write(O/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,epoch)),dict(active=True,count=count,KL_per_dim=total/count,switch=switch/count,R=(total+.1*switch)/count,budget=.10,budget_satisfied=(total+.1*switch)/count<=.10,teacher_hash=self.teacher_hash))
