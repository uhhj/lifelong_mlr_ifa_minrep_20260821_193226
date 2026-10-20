import json
import torch
from coreset_exp.common import O, SEED, sha, write
from coreset_exp.model import angular_mean
from coreset_exp.train import PaperMLR
from .loss import objective


class COVRMLR(PaperMLR):
    def configure_covr(self,task,epoch):
        self.covr_records=[]
        self.covr_epoch_statistics=[]
        if task==6 or epoch<50:
            return
        root=O/"covr"/("stage%d"%task)
        a=json.loads((root/"ACCEPTANCE.json").read_text())
        assert a["status"]=="PASS" and sha(root/"sidecar.pth")==a["sidecar_sha256"]
        self.covr_records=torch.load(root/"sidecar.pth",map_location="cpu")
        assert len(self.covr_records)==a["training_windows"]
        if self.covr_records:
            assert a["real_gradient"]["status"]=="PASS"
        write(O/"ra_coreset/covr_bindings"/("stage%d.json"%task),
              dict(sidecar_sha256=a["sidecar_sha256"],acceptance_sha256=sha(root/"ACCEPTANCE.json")))

    def observe(self,data,current_task_id,epoch,ifa_info=None):
        data[0]=self.map_tensor_to_device(data[0])
        self.optimizer.zero_grad()
        bc,last=self.policy.module.compute_loss(data[0],None,None)
        mask=torch.tensor([int(x)==current_task_id for x in data[1]],device=last.device)
        olds=[self.refs[j if k==current_task_id else k] for j,k in ifa_info["selected_pairs"]]
        extra=angular_mean(last[mask],self.refs[current_task_id],olds,self.cfg.gamma) if mask.any() else last.sum()*0
        loss=bc+.1*extra
        auxinfo=None
        if epoch>=50 and self.covr_records and (self.step+1)%10==0:
            n=len(self.covr_records)
            start=(self.step//10)*4
            selected=[self.covr_records[(start+i)%n] for i in range(4)]
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(840000000+1000000*SEED+self.step)
                aux,auxinfo=objective(self.policy.module,selected,last.device)
                loss=loss+aux
        assert torch.isfinite(loss)
        (self.loss_scale*loss).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.policy.parameters())
        if self.cfg.train.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(),self.cfg.train.grad_clip)
        self.optimizer.step();self.scheduler.step()
        self.covr_epoch_statistics.append(dict(step=self.step,bc=float(bc.detach()),
            ifa=float(extra.detach()),auxiliary=auxinfo))
        self.step+=1
        return float(loss.detach())
