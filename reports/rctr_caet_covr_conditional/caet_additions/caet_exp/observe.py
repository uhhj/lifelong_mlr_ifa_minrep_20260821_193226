import json
import torch
from torch.nn import functional as F
from coreset_exp.common import O, SEED, sha, write
from coreset_exp.train import PaperMLR
from coreset_exp.model import angular_mean
from .model import EffectModel


class CAETMLR(PaperMLR):
    def configure_effect(self,task):
        self.effect=None
        self.caet_epoch_statistics=[]
        if task==6:
            return
        root=O/"effects/models"/("task%d" % task)
        acceptance=json.loads((root/"ACCEPTANCE.json").read_text())
        assert acceptance["status"]=="PASS"
        assert sha(root/"final.pth")==acceptance["frozen_predictor_sha256"]
        payload=torch.load(root/"final.pth",map_location="cpu")
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            model=EffectModel().cuda()
            model.load_state_dict(payload["model"],strict=True)
        self.effect=model.eval().requires_grad_(False)
        self.effect_center=payload["center"].cuda()
        self.effect_scale=payload["scale"].cuda()
        self.effect_sha256=acceptance["frozen_predictor_sha256"]
        write(O/"ra_coreset/effect_bindings"/("task%d.json"%task),
              dict(model_sha256=self.effect_sha256,acceptance_sha256=sha(root/"ACCEPTANCE.json")))

    def observe(self,data,current_task_id,epoch,ifa_info=None):
        data[0]=self.map_tensor_to_device(data[0])
        self.optimizer.zero_grad()
        bc,last=self.policy.module.compute_loss(data[0],None,None)
        mask=torch.tensor([int(x)==current_task_id for x in data[1]],device=last.device)
        olds=[self.refs[j if k==current_task_id else k] for j,k in ifa_info["selected_pairs"]]
        extra=angular_mean(last[mask],self.refs[current_task_id],olds,self.cfg.gamma) if mask.any() else last.sum()*0
        loss=bc+.1*extra
        value=None;valid_count=0
        if self.effect is not None:
            valid=data[0]["effect_valid"].bool()
            valid_count=int(valid.sum())
            if valid_count:
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    torch.manual_seed(830000000+1000000*SEED+self.step)
                    x=data[0]["feature"][valid].squeeze(1)
                    z=self.policy.module.temporal_encode(x)[:,-1]
                    means,std,logits=self.policy.module.policy_head.forward_fn(z)
                    winner=logits.detach().argmax(-1)
                    index=torch.arange(len(winner),device=last.device)
                    action=means[index,winner]+std[index,winner]*torch.randn_like(means[index,winner])
                    prediction,_=self.effect(x,action)
                    target=(data[0]["effect_target"][valid]-self.effect_center)/self.effect_scale
                    aux=F.smooth_l1_loss(prediction,target)
                    loss=loss+.05*aux
                    value=float(aux.detach())
        assert torch.isfinite(loss)
        (self.loss_scale*loss).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.policy.parameters())
        if self.effect is not None:
            assert all(p.grad is None for p in self.effect.parameters())
        if self.cfg.train.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.train.grad_clip)
        self.optimizer.step();self.scheduler.step()
        self.caet_epoch_statistics.append(dict(step=self.step,bc=float(bc.detach()),
            ifa=float(extra.detach()),effect_loss=value,valid_labels=valid_count))
        self.step+=1
        return float(loss.detach())
