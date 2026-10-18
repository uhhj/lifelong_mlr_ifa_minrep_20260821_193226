"""Continuation of integration: real get_action, zero optimizer/environment calls."""
import copy,json,time
import numpy as np
import torch
from torch.distributions import Independent,Normal
from torch.utils.data import DataLoader,Subset
import robomimic.utils.tensor_utils as TU
from .common import *
from .model import cfg,data_setup,complete,load_stage,readout
from .diagnostics import compare
from libero.lifelong.models.brief import BriefPolicy
from libero.lifelong.utils import control_seed
def main():
    assert not(O/'NATIVE_ACTION_CHECK.json').exists();control_seed(SEED);torch.backends.cudnn.benchmark=True
    c=cfg('eval');b,ds,shape=data_setup(c,True);c.shape_meta=shape;p=complete(c,shape);load_stage(p,O/'integration/stage7.pth',7);p.eval()
    brief=BriefPolicy(cfg('train'),None).cuda();load_stage(brief,O/'integration/stage7.pth',7);brief.eval()
    choices=json.loads((PROTO/'SAMPLE_IDS.json').read_text());rows=[]
    for t in (6,7):
        data=TU.map_tensor(next(iter(DataLoader(Subset(ds[t],choices['raw'][str(t)][:2]*5),batch_size=10))),lambda x:x.cuda())
        for length in (1,4,8):
            d=copy.deepcopy(data);d['obs']={k:v[:,:length].contiguous() for k,v in d['obs'].items()};d['actions']=d['actions'][:,:length]
            with torch.no_grad():
                h=p.spatial_encode(p.preprocess_input(copy.deepcopy(d),train_mode=False));out=readout(p,p.temporal_encode(h)[:,-1],h[:,-1,0]);ix=torch.arange(10,device='cuda');j=out['selected']
                bout=readout(brief,brief.temporal_encode(h)[:,-1],h[:,-1,0]);parity=compare(out,bout);assert parity['passed']
                before=rng();expected=Independent(Normal(out['means'][ix,j],out['scales'][ix,j]),1).sample().cpu().numpy();after=rng_digest();restore_rng(before)
                actual=p.get_action(copy.deepcopy(d));exact=np.array_equal(expected,actual);rng_equal=rng_digest()==after
            rows.append(dict(task=t,history=length,action_exact=exact,action_max=float(abs(actual-expected).max()),sampling_rng_exact=rng_equal,complete_brief=parity))
            assert exact and rng_equal
    write(O/'NATIVE_ACTION_CHECK.json',dict(status='PASS',rows=rows,optimizer_calls=0,environment_steps=0,numeric=numeric(),get_action_latency=p.action_latency))
    write(O/'NATIVE_PATH_PARITY.json',dict(status='PASS',rows=rows,strict_all_stage_residual_keys=True,gate_dtype='float64',environment_steps=0))
    print('NATIVE ACTION PASS; 0 updates, 0 environments')
if __name__=='__main__':main()
