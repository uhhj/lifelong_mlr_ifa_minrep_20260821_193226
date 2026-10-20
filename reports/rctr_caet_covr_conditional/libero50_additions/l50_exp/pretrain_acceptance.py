"""Strictly reload newly reproduced full weights; no policy/environment execution."""
import json
import torch
from libero.lifelong.models.complete_policy import CompletePolicy
from .common import O,verify,sha,write
from .data import cfg
def main():
    verify()
    completion=json.loads((O/"PRETRAIN_COMPLETE.json").read_text())
    assert completion["epochs"]==100 and completion["steps"]>0
    cp=completion["final_checkpoint"];assert sha(cp)==completion["sha256"]
    raw=torch.load(cp,map_location="cpu")
    c=cfg("pretrain");c.pretraining_mode=False;c.pretrain=cp
    model=CompletePolicy(c,c.shape_meta)
    assert set(model.state_dict())==set(raw["state_dict"])
    assert all(torch.equal(v.cpu(),raw["state_dict"][k].cpu()) for k,v in model.state_dict().items())
    write(O/"PRETRAIN_STRICT_LOAD.json",dict(status="PASS",checkpoint=cp,sha256=sha(cp),
        tensors=len(raw["state_dict"]),origin="new reproduced shared 25-task checkpoint",
        epochs=100,steps=completion["steps"],policy_calls=0,environment_steps=0))
if __name__=="__main__":main()
