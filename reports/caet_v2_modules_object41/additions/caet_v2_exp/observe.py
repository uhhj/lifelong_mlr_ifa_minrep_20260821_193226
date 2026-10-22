import copy,json
import torch
from torch.nn import functional as F
from coreset_exp.common import O,SEED,sha,write,rng,restore_rng
from coreset_exp.train import PaperMLR
from coreset_exp.model import angular_mean
from .model import EffectModel
from .variant import VARIANT,feature,ActionSupport,phase
from . import functional_reference as reference

def safe_transfer(grad,bc):
 dot=sum((a.double()*b.double()).sum() for a,b in zip(grad,bc) if a is not None and b is not None)
 value=float(dot) if torch.is_tensor(dot) else float(dot)
 assert __import__('math').isfinite(value)
 return ([None]*len(grad) if value<0 else grad),value

class CAETv2MLR(PaperMLR):
 def configure_effect(self,t,ds):
  self.effect=None;self.ds=ds;self.mu={};self.bounds={};self.group_indices={};self.effect_sha256=None;self.caet_epoch_statistics=[]
  self.support_sha256=sha(O/'effects/current'/('task%d'%t)/'SUPPORT.json')
  if t==6:return
  root=O/'effects/models'/('task%d'%t);admit=json.loads((root/'ACCEPTANCE.json').read_text())
  if admit['status']=='FALLBACK_RA':return
  assert admit['status']=='PASS' and ds.support['stage_supported']
  assert sha(root/'final.pth')==admit['frozen_predictor_sha256']
  payload=torch.load(root/'final.pth',map_location='cpu');saved=rng()
  try:
   model=EffectModel().cuda();model.load_state_dict(payload['model'],strict=True)
   self.effect=model.eval().requires_grad_(False);self.effect_center=payload['center'].cuda();self.effect_scale=payload['scale'].cuda();self.effect_sha256=admit['frozen_predictor_sha256']
   self.group_indices,self.bounds=reference.build(self.policy.module,ds,t)
   self.mu={k:0. for k in self.bounds}
   if VARIANT=='RSET':
    self.action_support=ActionSupport(ds,torch.device('cuda'));self.reliable={k for k,v in admit['reliability'].items() if v['accepted']}
  finally:restore_rng(saved)
  write(O/'ra_coreset/effect_bindings'/('task%d.json'%t),dict(model_sha256=self.effect_sha256,support_sha256=self.support_sha256,reference_sha256=sha(O/'ra_coreset/functional_references'/('task%d.json'%t))))
 def export_v2_state(self):
  return dict(mu=copy.deepcopy(self.mu),bounds=copy.deepcopy(self.bounds),effect_sha256=self.effect_sha256,support_sha256=self.support_sha256)
 def import_v2_state(self,state):
  assert self.effect_sha256==state['effect_sha256'] and self.support_sha256==state['support_sha256'] and self.bounds==state['bounds']
  assert set(state['mu'])==set(self.bounds) and all(0<=v<=1 for v in state['mu'].values());self.mu=copy.deepcopy(state['mu'])
 def finish_epoch(self,t,epoch):
  if self.effect is None:
   write(O/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,epoch)),dict(active=False,reason='TASK6' if t==6 else 'STAGE_ADMISSION_FALLBACK'));return
  values=reference.score(self.policy.module,self.ds,self.group_indices);before=dict(self.mu);self.mu=reference.dual(self.mu,values,self.bounds)
  write(O/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,epoch)),dict(active=True,values=values,bounds=self.bounds,mu_before=before,mu_after=self.mu,epsilon=.05))
 def observe(self,data,current_task_id,epoch,ifa_info=None):
  if self.effect is None:
   step=self.step;value=PaperMLR.observe(self,data,current_task_id,epoch,ifa_info)
   self.caet_epoch_statistics.append(dict(step=step,active=False,effect_loss=None,transfer_supported=0,transfer_dot=None,transfer_applied=False,retention_loss=0.,retention_groups=0))
   return value
  ids=data[0]['v2_index'].tolist();data[0]=self.map_tensor_to_device(data[0]);x=data[0];self.optimizer.zero_grad()
  bc,last=self.policy.module.compute_loss(x,None,None);mask=torch.tensor([int(k)==current_task_id for k in data[1]],device=last.device)
  olds=[self.refs[j if k==current_task_id else k] for j,k in ifa_info['selected_pairs']]
  ifa=angular_mean(last[mask],self.refs[current_task_id],olds,self.cfg.gamma) if mask.any() else last.sum()*0
  native=bc+.1*ifa;params=[p for p in self.policy.parameters() if p.requires_grad]
  supported=mask & x['effect_valid'].bool() & x['transfer_supported'].bool() & ~x['effect_holdout'].bool()
  transferred=[None]*len(params);dot=None;aux_value=None;applied=False;retain_value=0.;retained=0;retloss=None;gate_counts=None
  with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
   torch.manual_seed(830000000+1000000*SEED+self.step)
   if supported.any():
    # Current BC gradient uses only current samples and the native BC formula.
    # Recompute with an auxiliary RNG only; never advance native dropout stream.
    cur={k:v[mask] for k,v in x.items() if torch.is_tensor(v)}
    cur_bc,_=self.policy.module.compute_loss(cur,None,None)
    bcgrad=torch.autograd.grad(cur_bc,params,allow_unused=True)
    feat=feature(x['feature'][supported]);z=self.policy.module.temporal_encode(feat)[:,-1]
    means,std,logits=self.policy.module.policy_head.forward_fn(z);winner=logits.detach().argmax(-1);index=torch.arange(len(winner),device=last.device)
    action=means[index,winner]+std[index,winner]*torch.randn_like(means[index,winner]);pred,_=self.effect(feat,action)
    weights=torch.ones(len(action),device=action.device,dtype=torch.bool)
    if VARIANT=='RSET':
     action_ok=self.action_support.mask(feat,action.detach())
     rel_ok=torch.tensor([phase(self.ds.rows[ids[i]]) in self.reliable for i in torch.nonzero(supported).flatten().tolist()],device=action.device)
     weights=action_ok & rel_ok
     gate_counts=dict(cross=int(supported.sum()),action=int(action_ok.sum()),reliability=int(rel_ok.sum()),joint=int(weights.sum()))
    per=F.smooth_l1_loss(pred,(x['effect_target'][supported]-self.effect_center)/self.effect_scale,reduction='none').mean(-1)
    aux=(per*weights).sum()/weights.sum().clamp_min(1)
    auxgrad=torch.autograd.grad(.05*aux,params,allow_unused=True);transferred,dot=safe_transfer(auxgrad,bcgrad);applied=dot>=0 and bool(weights.any());aux_value=float(aux.detach())
   # Mini-batch stochastic group estimates; full-group constraints update mu only
   # at complete epochs. Heldout refers only to effect fitting, not RA protection.
   oldvalid=(~mask)&x['effect_valid'].bool()
   if oldvalid.any() and any(v>0 for v in self.mu.values()):
    modes={n:m.training for n,m in self.policy.module.named_modules()}
    self.policy.module.eval()
    try:
     values=reference.nll(self.policy.module,feature(x['feature'][oldvalid]),x['next_action'][oldvalid])
     keys=[str(self.ds.rows[ids[i]]['task'])+'|'+self.ds.rows[ids[i]]['group'] for i in torch.nonzero(oldvalid).flatten().tolist()]
     terms=[]
     for key in sorted(set(keys)&set(self.bounds)):
      selected=torch.tensor([k==key for k in keys],device=last.device);excess=F.relu(values[selected].mean()-self.bounds[key]-.05)
      terms.append(self.mu[key]*excess);retained+=int(float(excess.detach())>0 and self.mu[key]>0)
     if terms:retloss=torch.stack(terms).sum()/len(self.bounds);retain_value=float(retloss.detach())
    finally:
     for n,m in self.policy.module.named_modules():m.training=modes[n]
  loss=native if retloss is None else native+retloss
  assert torch.isfinite(loss);(self.loss_scale*loss).backward()
  for p,g in zip(params,transferred):
   if g is not None:
    if p.grad is None:p.grad=g.detach()*self.loss_scale
    else:p.grad.add_(g.detach(),alpha=self.loss_scale)
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in params)
  assert all(p.grad is None for p in self.effect.parameters())
  if self.cfg.train.grad_clip is not None:torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.train.grad_clip)
  self.optimizer.step();self.scheduler.step()
  self.caet_epoch_statistics.append(dict(step=self.step,active=True,bc=float(bc.detach()),ifa=float(ifa.detach()),effect_loss=aux_value,transfer_supported=int(supported.sum()),transfer_dot=dot,transfer_applied=applied,retention_loss=retain_value,retention_groups=retained,selective_gate=gate_counts))
  self.step+=1;return float(loss.detach())+(0.05*aux_value if applied else 0.)

if VARIANT=='EFTR':
 from .eftr import EFTRMLR
 CAETv2MLR=EFTRMLR
