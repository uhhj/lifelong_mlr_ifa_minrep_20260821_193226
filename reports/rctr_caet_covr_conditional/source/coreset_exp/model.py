import copy,json
import torch
import torch.nn.functional as F
from easydict import EasyDict
from .common import O,S,write,sha
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import get_dataset,SequenceVLDataset
from libero.lifelong.utils import get_task_embs,control_seed,torch_load_model
from libero.lifelong.models.complete_policy import CompletePolicy
from libero.lifelong.models.brief import BriefPolicy

def cfg(name):
    from .common import SEED
    c=EasyDict(json.loads((O/f'resolved_{name}.json').read_text()))
    assert c.benchmark_name=='LIBERO_OBJECT' and c.gamma==.3 and c.pretrain.endswith('object_multitask_model_ep99.pth')
    assert c.seed==SEED and all(c.policy[k] is True for k in ('use_vision','use_language','use_state'))
    return c
def data_setup(c,all_datasets=False):
    b=get_benchmark(c.benchmark_name)(c.data.task_order_index)
    datasets=[];shape=None
    # Same dataset and tokenization order as released exporter/evaluator.
    for t in range(10 if all_datasets else 1):
        d,shape=get_dataset(str(__import__('pathlib').Path(c.folder)/b.get_task_demonstration(t)),
            c.data.obs.modality,initialize_obs_utils=(t==0),seq_len=8,demo_limit=10,train=c.training)
        datasets.append(d)
    embs=get_task_embs(c,[b.get_task(t).language for t in range(10)]);b.set_task_embs(embs)
    return b,[SequenceVLDataset(d,embs['input_ids'][t],embs['attention_mask'][t]) for t,d in enumerate(datasets)],shape
def strict_downstream(model,path):
    sd=torch_load_model(path)[0];target=model.state_dict()
    wanted={k:v for k,v in sd.items() if k.startswith(('temporal_transformer.','policy_head.'))}
    expected={k for k in target if k.startswith(('temporal_transformer.','policy_head.'))}
    assert set(wanted)==expected,'Incomplete downstream checkpoint'
    for k in expected:assert target[k].shape==wanted[k].shape,k
    result=model.load_state_dict(wanted,strict=False)
    assert not result.unexpected_keys and not (set(result.missing_keys)&expected)
    for k in expected:assert torch.equal(model.state_dict()[k].cpu(),wanted[k].cpu()),k
    return len(expected)
def complete(c,shape):
    p=CompletePolicy(c,shape).cuda() # Native constructor strict=True on full official weight.
    raw=torch_load_model(c.pretrain)[0];actual=p.state_dict()
    assert set(raw)==set(actual)
    for k in raw:assert torch.equal(raw[k].cpu(),actual[k].cpu()),k
    return p
def angular_mean(global_current,own_reference,old_references,gamma):
    if not old_references:return global_current.sum()*0
    g=F.normalize(global_current,dim=-1,eps=1e-8)
    own=F.normalize(own_reference,dim=-1,eps=1e-8)
    theta_pos=torch.acos((g*own).sum(-1).clamp(-1+1e-6,1-1e-6))
    losses=[]
    for ref in old_references:
        old=F.normalize(ref,dim=-1,eps=1e-8)
        neg=torch.acos((g*old).sum(-1).clamp(-1+1e-6,1-1e-6))
        phi=torch.acos((own*old).sum(-1).clamp(-1+1e-6,1-1e-6))
        losses.append(torch.relu(theta_pos-neg+gamma*phi).mean())
    return torch.stack(losses).mean()
