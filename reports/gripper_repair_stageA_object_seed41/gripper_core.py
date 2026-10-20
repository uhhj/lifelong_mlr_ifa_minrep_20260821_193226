import numpy as np
import torch
from torch import nn

class GripperHead(nn.Module):
    def __init__(self, canonical):
        super().__init__()
        self.register_buffer("canonical", canonical.double()/canonical.double().norm(dim=-1,keepdim=True))
        self.linear=nn.Linear(512,1,bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    def forward(self,z):
        return self.linear(z).squeeze(-1)
    def route(self,e):
        n=e.double().norm(dim=-1,keepdim=True)
        matches=(e.double().div(n.clamp_min(1e-12))[:,None]-self.canonical[None]).abs().amax(-1)<=1e-6
        return (n[:,0]>1e-12)&(matches.sum(-1)==1)&matches[:,6]
    def apply(self,parent_actions,z,e):
        out=parent_actions.clone()
        logit=self(z)
        route=self.route(e)&(logit!=0)
        out[route,6]=torch.where(logit[route]>0,torch.ones_like(logit[route]),-torch.ones_like(logit[route]))
        return out

def install(policy,head):
    """Keep original sampling and first six actions, override gripper afterward."""
    ctx={}
    spatial=policy.spatial_encode
    def capture(data):
        h=spatial(data);ctx["h"]=h
        return h
    policy.spatial_encode=capture
    handle=policy.policy_head.register_forward_pre_hook(lambda m,args:ctx.update(z=args[0].detach()))
    original=policy.get_action
    def action(data):
        with torch.no_grad():
            old=original(data)
            z=ctx["z"];e=ctx["h"][:,-1,0]
            result=head.apply(torch.as_tensor(old,device=z.device),z,e)
            ctx["logit"]=head(z).detach()
            ctx["route"]=head.route(e).detach()
            return result.cpu().numpy()
    policy.get_action=action
    return ctx,original,handle
