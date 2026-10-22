"""Native/candidate AdamW transactions. Local guards are not rollout guarantees."""
import copy,json,time
import torch
from torch.nn import functional as F
from coreset_exp.common import O,SEED,sha,write,rng,restore_rng
from coreset_exp.train import PaperMLR
from coreset_exp.model import angular_mean
from .model import EffectModel
from .variant import exec_params,exec_kl

def capture(a):
    return copy.deepcopy(dict(model=a.policy.state_dict(),optimizer=a.optimizer.state_dict(),scheduler=a.scheduler.state_dict(),
        grads={n:p.grad for n,p in a.policy.named_parameters()},modes={n:m.training for n,m in a.policy.named_modules()},rng=rng(),step=a.step))
def restore(a,x):
    a.policy.load_state_dict(x['model'],strict=True);a.optimizer.load_state_dict(x['optimizer']);a.scheduler.load_state_dict(x['scheduler']);a.step=x['step']
    for n,p in a.policy.named_parameters():p.grad=None if x['grads'][n] is None else x['grads'][n].clone()
    for n,m in a.policy.named_modules():m.training=x['modes'][n]
    restore_rng(x['rng'])
def decision(native,candidate):
    reasons=[]
    if candidate['bc'] is not None and candidate['bc']>native['bc']+.01:reasons.append('CURRENT_BC')
    if candidate['kl'] is not None:
        if candidate['kl']>.05:reasons.append('EXEC_KL')
        if candidate['switch']>.10:reasons.append('MODE_SWITCH')
        if candidate['gripper']>.05:reasons.append('GRIPPER')
    return not reasons,reasons

class EFTRMLR(PaperMLR):
    def configure_effect(self,t,ds):
        self.effect=None;self.reference=None;self.ds=ds;self.mu={};self.bounds={};self.caet_epoch_statistics=[]
        self.support_sha256=sha(O/'effects/current'/('task%d'%t)/'SUPPORT.json');self.effect_sha256=None;self.reference_sha256=None
        if t==6:return
        cp=O/'ra_coreset/models'/('mlrifa_%d_model_ep99.pth'%(t-1))
        self.reference_sha256=sha(cp);saved=rng()
        try:
            self.reference=copy.deepcopy(self.policy.module).eval().requires_grad_(False)
            raw=torch.load(cp,map_location='cpu')['state_dict']
            self.reference.load_state_dict({k[7:] if k.startswith('module.') else k:v for k,v in raw.items()},strict=True)
            root=O/'effects/models'/('task%d'%t);admit=json.loads((root/'ACCEPTANCE.json').read_text())
            if admit['status']=='PASS':
                assert sha(root/'final.pth')==admit['frozen_predictor_sha256']
                payload=torch.load(root/'final.pth',map_location='cpu');self.effect=EffectModel().cuda().eval().requires_grad_(False)
                self.effect.load_state_dict(payload['model'],strict=True);self.effect_center=payload['center'].cuda();self.effect_scale=payload['scale'].cuda();self.effect_sha256=admit['frozen_predictor_sha256']
            else:assert admit['status']=='FALLBACK_RA'
        finally:restore_rng(saved)
        write(O/'ra_coreset/effect_bindings'/('task%d.json'%t),dict(reference_stage=t-1,reference_sha256=self.reference_sha256,model_sha256=self.effect_sha256,support_sha256=self.support_sha256,retention_independent_of_transfer=True))
    def export_v2_state(self):
        return dict(mu={},bounds={},effect_sha256=self.effect_sha256,support_sha256=self.support_sha256,reference_sha256=self.reference_sha256)
    def import_v2_state(self,s):assert s==self.export_v2_state()
    @torch.no_grad()
    def metrics(self,x,mask):
        modes={n:m.training for n,m in self.policy.named_modules()};saved=rng();self.policy.eval()
        try:
            current={k:v[mask] for k,v in x.items() if torch.is_tensor(v)}
            bc=float(self.policy.module.compute_loss(current,None,None)[0]) if mask.any() else None
            if (~mask).any():
                ref=exec_params(self.reference,x['feature'][~mask]);now=exec_params(self.policy.module,x['feature'][~mask])
                kl=float(exec_kl(ref,now).mean());switch=float((ref[2]!=now[2]).float().mean());gripper=float((ref[0][:,-1]-now[0][:,-1]).abs().mean())
            else:kl=switch=gripper=None
            values=dict(bc=bc,kl=kl,switch=switch,gripper=gripper)
            assert all(v is None or __import__('math').isfinite(v) for v in values.values())
            return values
        finally:
            for n,m in self.policy.named_modules():m.training=modes[n]
            restore_rng(saved)
    def observe(self,data,current_task_id,epoch,ifa_info=None):
        if self.reference is None:
            step=self.step;v=PaperMLR.observe(self,data,current_task_id,epoch,ifa_info)
            self.caet_epoch_statistics.append(dict(step=step,active=False,effect_loss=None,transfer_supported=0,transfer_dot=None,transfer_applied=False,retention_loss=0.,retention_groups=0,candidate_optimizer_calls=0))
            return v
        tic=time.monotonic();data[0]=self.map_tensor_to_device(data[0]);x=data[0];self.optimizer.zero_grad()
        bc,last=self.policy.module.compute_loss(x,None,None);mask=torch.tensor([int(k)==current_task_id for k in data[1]],device=last.device)
        olds=[self.refs[j if k==current_task_id else k] for j,k in ifa_info['selected_pairs']]
        ifa=angular_mean(last[mask],self.refs[current_task_id],olds,self.cfg.gamma) if mask.any() else last.sum()*0
        native=bc+.1*ifa;params=[p for p in self.policy.parameters() if p.requires_grad]
        native_grad=torch.autograd.grad(self.loss_scale*native,params,allow_unused=True)
        aux_grad=[None]*len(params);effect_value=None;retention_value=0.;dot=None
        supported=mask & x['effect_valid'].bool() & x['transfer_supported'].bool() & ~x['effect_holdout'].bool()
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(830000000+1000000*SEED+self.step)
            aux=None
            if self.effect is not None and supported.any():
                feat=x['feature'][supported].squeeze(1);m,s,_=exec_params(self.policy.module,feat)
                act=m+s*torch.randn_like(m);pred,_=self.effect(feat,act)
                effect=F.smooth_l1_loss(pred,(x['effect_target'][supported]-self.effect_center)/self.effect_scale)
                aux=.05*effect;effect_value=float(effect.detach())
                cur={k:v[mask] for k,v in x.items() if torch.is_tensor(v)}
                cg=torch.autograd.grad(self.policy.module.compute_loss(cur,None,None)[0],params,allow_unused=True)
                eg=torch.autograd.grad(.05*effect,params,allow_unused=True,retain_graph=True)
                dot=float(sum((a.double()*b.double()).sum() for a,b in zip(cg,eg) if a is not None and b is not None))
            if (~mask).any():
                modes={n:m.training for n,m in self.policy.named_modules()};self.policy.eval()
                try:
                    with torch.no_grad():ref=exec_params(self.reference,x['feature'][~mask])
                    retention=exec_kl(ref,exec_params(self.policy.module,x['feature'][~mask])).mean()
                    retention_value=float(retention.detach());aux=.1*retention if aux is None else aux+.1*retention
                finally:
                    for n,m in self.policy.named_modules():m.training=modes[n]
            if aux is not None:aux_grad=torch.autograd.grad(self.loss_scale*aux,params,allow_unused=True)
        assert all(g is None or torch.isfinite(g).all() for g in [*native_grad,*aux_grad])
        start=capture(self)
        def step(extra):
            self.optimizer.zero_grad()
            for p,n,g in zip(params,native_grad,aux_grad):
                p.grad=None if n is None and (not extra or g is None) else (torch.zeros_like(p) if n is None else n.detach().clone())
                if extra and g is not None:p.grad.add_(g.detach())
            if self.cfg.train.grad_clip is not None:torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.train.grad_clip)
            self.optimizer.step();self.scheduler.step();self.step+=1
        step(False);native_metrics=self.metrics(x,mask);native_end=capture(self)
        restore(self,start);step(True);candidate_metrics=self.metrics(x,mask)
        accepted,reasons=decision(native_metrics,candidate_metrics)
        if not accepted:restore(self,native_end)
        assert self.step==start['step']+1 and self.scheduler.state_dict()==native_end['scheduler']
        self.caet_epoch_statistics.append(dict(step=start['step'],active=True,effect_loss=effect_value,transfer_supported=int(supported.sum()),transfer_dot=dot,
            transfer_applied=accepted and effect_value is not None,retention_loss=retention_value,retention_groups=int((~mask).sum()),
            candidate_optimizer_calls=2,committed_updates=1,candidate_accepted=accepted,rejection_reasons=reasons,native_metrics=native_metrics,candidate_metrics=candidate_metrics,seconds=time.monotonic()-tic))
        return float(native.detach())+(0.05*(effect_value or 0)+.1*retention_value if accepted else 0)
    @torch.no_grad()
    def finish_epoch(self,t,epoch):
        if self.reference is None:
            write(O/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,epoch)),dict(active=False,reason='TASK6'));return
        ids=[i for i,r in enumerate(self.ds.rows) if r['task']<t];modes={n:m.training for n,m in self.policy.named_modules()};saved=rng();self.policy.eval()
        sums=dict(kl=0.,switch=0.,gripper=0.)
        try:
            for j in range(0,len(ids),64):
                ix=ids[j:j+64];feat=torch.stack([self.ds.native[i][0]['feature'].squeeze(0) for i in ix]).cuda()
                ref=exec_params(self.reference,feat);now=exec_params(self.policy.module,feat)
                sums['kl']+=float(exec_kl(ref,now).double().sum());sums['switch']+=float((ref[2]!=now[2]).double().sum());sums['gripper']+=float((ref[0][:,-1]-now[0][:,-1]).abs().double().sum())
        finally:
            for n,m in self.policy.named_modules():m.training=modes[n]
            restore_rng(saved)
        write(O/'ra_coreset/constraints'/('task%d_epoch%02d.json'%(t,epoch)),dict(active=True,reference_sha256=self.reference_sha256,samples=len(ids),full_memory={k:v/len(ids) for k,v in sums.items()},not_a_global_guarantee=True))
