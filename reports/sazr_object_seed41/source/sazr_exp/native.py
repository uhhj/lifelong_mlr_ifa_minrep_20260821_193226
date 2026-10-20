from .common import *
from torch import nn
from torch.distributions import Independent,Normal

class Readout(nn.Module):
    def __init__(self,base,bank,enabled,ctx):
        super().__init__();self.base=base;self.bank=bank;self.enabled=enabled;self.ctx=ctx
    def forward(self,z,train=False):
        self.ctx["z"]=z.detach()
        if not self.enabled or self.bank is None:
            self.ctx["route_count"]=0
            return self.base(z,train=train)
        e=self.ctx["h"][:,-1,0]
        route=self.bank.route(e);self.ctx["route_count"]=int(route.sum())
        if not bool(route.any()):return self.base(z,train=train)
        b,s,l=raw(self.base,z);delta=self.bank(z,e)
        delta=torch.where(route[:,None],delta,torch.zeros_like(delta))
        out=corrected(b,s,l,delta)
        self.ctx["delta"]=delta.detach();self.ctx["out"]=out
        # Evaluation uses original argmax component Normal; no mean action shortcut.
        assert not train
        return out["dist"]

def install(p,bank=None,enabled=False,table=None):
    ctx={};spatial=p.spatial_encode
    def capture(data):
        h=spatial(data);ctx["h"]=h
        return h
    p.spatial_encode=capture;p.policy_head=Readout(p.policy_head,bank,enabled,ctx)
    if table is not None:
        # Explicit trajectory/time noise only for independent validation.
        def action(data):
            with torch.no_grad():
                p.eval()
                data=p.preprocess_input(data,train_mode=False)
                h=p.spatial_encode(data);z=p.temporal_encode(h)[:,-1]
                dist=p.policy_head(z)
                step=ctx["step"]
                return (dist.base_dist.loc+dist.base_dist.scale*table[:,step].to(z.device)).cpu().numpy()
        p.get_action=action
    return ctx
