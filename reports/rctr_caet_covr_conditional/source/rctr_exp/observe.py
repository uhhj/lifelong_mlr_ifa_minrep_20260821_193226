"""Normal RA batch and update count unchanged; one certified old sidecar per update."""
import json
from pathlib import Path
import torch
from coreset_exp.common import O, SEED, sha
from coreset_exp.model import angular_mean
from coreset_exp.train import PaperMLR
from .loss import objective


class RCTRMLR(PaperMLR):
    def configure_sidecars(self, current_task):
        self.rctr_records = []
        for task in range(6, current_task):
            root = O/"rctr"/("task%d" % task)
            acceptance = json.loads((root/"ACCEPTANCE.json").read_text())
            assert acceptance["status"] == "PASS"
            assert sha(root/"sidecar.pth") == acceptance["sidecar_sha256"]
            records = torch.load(root/"sidecar.pth", map_location="cpu")
            assert len(records) == acceptance["records"] <= 12
            for record in records:
                assert record["certified"] is True
                self.rctr_records.append(record)
        self.rctr_scale = torch.load(O/"pretraining_action_scale.pth", map_location="cpu").cuda()
        self.rctr_epoch_statistics = []

    def observe(self, data, current_task_id, epoch, ifa_info=None):
        data[0] = self.map_tensor_to_device(data[0])
        self.optimizer.zero_grad()
        bc, last = self.policy.module.compute_loss(data[0], None, None)
        mask = torch.tensor([int(x) == current_task_id for x in data[1]], device=last.device)
        pairs = ifa_info["selected_pairs"]
        olds = [self.refs[j if k == current_task_id else k] for j, k in pairs]
        extra = angular_mean(last[mask], self.refs[current_task_id], olds,
                             self.cfg.gamma) if mask.any() else last.sum()*0
        # Exactly the native loss expression when no certified old record exists.
        loss = bc + .1*extra
        aux_info = None
        if self.rctr_records:
            record = self.rctr_records[self.step % len(self.rctr_records)]
            # Counter-derived local stream: identical after complete-epoch recovery.
            # fork_rng restores both CPU and CUDA streams before the original update.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.manual_seed(600000000 + 1000000*SEED + self.step)
                aux, aux_info = objective(self.policy.module, record,
                                          self.rctr_scale, last.device)
                loss = loss + aux
        assert torch.isfinite(loss)
        (self.loss_scale*loss).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all()
                   for p in self.policy.parameters())
        if self.cfg.train.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.train.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.rctr_epoch_statistics.append(dict(step=self.step, bc=float(bc.detach()),
            ifa=float(extra.detach()), auxiliary=aux_info))
        self.step += 1
        return float(loss.detach())
