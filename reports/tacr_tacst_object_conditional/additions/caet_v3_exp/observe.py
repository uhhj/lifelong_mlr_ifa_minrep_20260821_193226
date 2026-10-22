import copy,json,time
import torch
from torch.nn import functional as F
from coreset_exp.common import O,SEED,sha,write,rng,restore_rng
from coreset_exp.train import PaperMLR
from coreset_exp.model import angular_mean
from .variant import VARIANT
from .model import Student,hidden,masked_bc
class V3MLR(PaperMLR):
 def configure_effect(self,t,ds):
  self.ds=ds;self.task=t;self.caet_epoch_statistics=[];self.cursor=0;self.head_steps=0;self.disable_all=False
  self.support_sha256=sha(O/'effects/current'/('task%d'%t)/'SUPPORT.json');self.teacher_hash={};self.keys={}
  saved=rng()
  try:
   torch.manual_seed(930000+SEED);self.student=Student().cuda()
   if t>6:self.student.load_state_dict(torch.load(O/'ra_coreset/aux_heads'/('task%d.pth'%(t-1)),map_location='cpu')['model'],strict=True)
   core=[p for p in self.policy.parameters() if p.requires_grad];ctrl=self.student.control_parameters();rep=self.student.representation_parameters()
   assert not set(map(id,core+ctrl))&set(map(id,rep)) and len(self.optimizer.param_groups)==1
   self.optimizer.param_groups[0]['params'].extend(ctrl)
   self.repopt=torch.optim.AdamW(rep,lr=.0003,weight_decay=.01)
   self.repscheduler=torch.optim.lr_scheduler.LambdaLR(self.repopt,lr_lambda=self.scheduler.lr_lambdas[0])
   if VARIANT=='TACRST' and t>6:
    from .teacher_keys import prepare_keys
    self.keys,self.teacher_hash=prepare_keys(self,t,ds)
  finally:restore_rng(saved)
 def aux_state(self):
  return copy.deepcopy(dict(model=self.student.state_dict(),optimizer=self.repopt.state_dict(),scheduler=self.repscheduler.state_dict(),grads={n:p.grad for n,p in self.student.named_parameters()},modes={n:m.training for n,m in self.student.named_modules()},requires_grad={n:p.requires_grad for n,p in self.student.named_parameters()},steps=self.head_steps))
 def load_aux(self,s):
  self.student.load_state_dict(s['model'],strict=True);self.repopt.load_state_dict(s['optimizer']);self.repscheduler.load_state_dict(s['scheduler']);self.head_steps=s['steps']
  for n,p in self.student.named_parameters():p.grad=None if s['grads'][n] is None else s['grads'][n].to(p.device);p.requires_grad_(s['requires_grad'][n])
  for n,m in self.student.named_modules():m.training=s['modes'][n]
 def export_v2_state(self):return dict(variant=VARIANT,support_sha256=self.support_sha256,teacher_hash=self.teacher_hash,cursor=self.cursor,aux=self.aux_state(),requires_grad={n:p.requires_grad for n,p in self.policy.named_parameters()})
 def import_v2_state(self,s):
  assert s['variant']==VARIANT and s['support_sha256']==self.support_sha256 and s['teacher_hash']==self.teacher_hash
  self.cursor=s['cursor'];self.load_aux(s['aux'])
  for n,p in self.policy.named_parameters():p.requires_grad_(s['requires_grad'][n])
 def cross(self,x,c):
  losses=[];anchors=[];partners=[];before=self.cursor
  for b,i in enumerate(x['v3_index'].tolist()):
   ps=self.ds.positive.get(i,[]);ns=self.ds.negative.get(i,[])
   if not ps or not ns:continue
   ids=[ps[self.cursor%len(ps)]]+[ns[(self.cursor+k)%len(ns)] for k in range(min(4,len(ns)))];self.cursor+=1
   assert all(j in self.keys for j in ids)
   keys=torch.stack([self.keys[j] for j in ids]).to(c.device);assert not keys.requires_grad
   z=F.normalize(self.student.condition(c[b:b+1],x['next_action'][b:b+1]),dim=-1)
   scores=(z@keys.T).squeeze(0)/.1;losses.append(torch.logsumexp(scores,0)-scores[0]);anchors.append(i);partners.append(ids)
  return (torch.stack(losses).mean() if losses else c.sum()*0),anchors,partners,before
 def observe(self,data,current_task_id,epoch,ifa_info=None):
  if self.disable_all:return PaperMLR.observe(self,data,current_task_id,epoch,ifa_info)
  tic=time.monotonic();step=self.step;x=self.map_tensor_to_device(data[0]);self.optimizer.zero_grad();self.repopt.zero_grad()
  h=hidden(self.policy.module,x['feature']);fused,c,gate,delta=self.student.deploy(h);bc,valid=masked_bc(self.policy.module,fused,x)
  current=torch.tensor([int(k) for k in data[1]],device=h.device)==current_task_id
  olds=[self.refs[j if k==current_task_id else k] for j,k in ifa_info['selected_pairs']]
  ifa=angular_mean(h[current],self.refs[current_task_id],olds,self.cfg.gamma) if current.any() else h.sum()*0
  em=x['effect_valid'].bool();effect=c.sum()*0
  if em.any():effect=F.smooth_l1_loss(self.student.D(self.student.condition(c[em],x['next_action'][em])),x['effect_target'][em]/h.new_tensor([.01]*3+[torch.pi/18]*3+[.01]))
  st=c.sum()*0;anchors=[];partners=[];cursor_before=self.cursor
  if VARIANT=='TACRST' and self.task>6:st,anchors,partners,cursor_before=self.cross(x,c)
  loss=bc+.1*ifa+.03*effect+.02*st;assert torch.isfinite(loss);(self.loss_scale*loss).backward()
  groups=[list(self.policy.parameters()),self.student.control_parameters(),self.student.representation_parameters()]
  assert all(p.grad is None or torch.isfinite(p.grad).all() for g in groups for p in g)
  if self.cfg.train.grad_clip is not None:
   for g in groups:torch.nn.utils.clip_grad_norm_(g,self.cfg.train.grad_clip)
  self.optimizer.step();self.repopt.step();self.scheduler.step();self.repscheduler.step();self.step+=1;self.head_steps+=1
  ratio=(delta.detach().square().mean(-1).sqrt()/h.detach().square().mean(-1).clamp_min(1e-16).sqrt()).max();assert float(ratio)<=.200001
  self.caet_epoch_statistics.append(dict(step=step,variant=VARIANT,batch_ids=x['v3_index'].tolist(),bc=float(bc.detach()),ifa=float(ifa.detach()),bc_valid=int(valid.sum()),bc_invalid=int((~valid).sum()),effect_samples=int(em.sum()),effect_loss=float(effect.detach()),st_loss=float(st.detach()),cross_anchors=len(anchors),anchor_ids=anchors,partner_ids=partners,old_key_reads=sum(map(len,partners)),cursor_before=cursor_before,cursor=self.cursor,teacher_hash=self.teacher_hash,head_updates=1,head_steps=self.head_steps,committed_updates=1,optimizer_trials=1,representation_optimizer_calls=1,gate_mean=float(gate.detach().mean()),residual_ratio_max=float(ratio),seconds=time.monotonic()-tic))
  return float(loss.detach())
 def finish_epoch(self,t,epoch):write(O/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,epoch)),dict(active=False,variant=VARIANT,no_EFTR=True))
