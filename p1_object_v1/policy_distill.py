"""FP32 full-mixture teacher -> student MC KL; private CPU sampling stream."""
import copy
from contextlib import contextmanager

import torch
from torch.distributions import Categorical, Independent, MixtureSameFamily, Normal
from e1_object_v1.control import rng_state, restore_rng


@contextmanager
def deterministic_distribution(model):
    modes=[(m,m.training) for m in model.modules()]
    # Swap buffers, rather than copy_ while a native/student graph is alive.
    buffers=[(m,k,v) for m in model.modules() for k,v in m._buffers.items()]
    history={k:copy.deepcopy(getattr(model,k)) for k in ('latent_queue','history_buffer') if hasattr(model,k)}
    state=rng_state()
    try:
        for m,k,v in buffers:
            if v is not None:m._buffers[k]=v.detach().clone()
        model.training=True
        model.policy_head.training=True
        for m in model.modules():
            if isinstance(m,(torch.nn.modules.dropout._DropoutNd,torch.nn.modules.batchnorm._BatchNorm)):m.training=False
        with torch.cuda.amp.autocast(enabled=False):yield
    finally:
        for m,k,v in buffers:m._buffers[k]=v
        for m,flag in modes:m.training=flag
        for k,v in history.items():setattr(model,k,v)
        restore_rng(state)


def fields(distribution):
    out=dict(log_weights=distribution.mixture_distribution.logits,
             means=distribution.component_distribution.base_dist.loc,
             scales=distribution.component_distribution.base_dist.scale)
    assert out['means'].ndim==3 and out['means'].shape[-2:]==(5,7)
    assert all(v.dtype==torch.float32 and torch.isfinite(v).all() for v in out.values())
    assert (out['scales']>0).all()
    return out


def mixture(f):
    return MixtureSameFamily(Categorical(logits=f['log_weights']),Independent(Normal(f['means'],f['scales']),1))


def sample_teacher(f, generator, count=16):
    assert all(v.device.type=='cpu' and not v.requires_grad for v in f.values())
    with torch.no_grad():
        ids=torch.multinomial(f['log_weights'].exp(),count,replacement=True,generator=generator)
        index=ids.unsqueeze(-1).expand(-1,-1,f['means'].shape[-1])
        means=f['means'].gather(1,index);scales=f['scales'].gather(1,index)
        actions=(means+scales*torch.randn(means.shape,generator=generator,dtype=torch.float32)).transpose(0,1).contiguous()
        logp=mixture(f).log_prob(actions)
    return actions,logp


def loss_from_samples(student, actions, teacher_logp):
    assert not actions.requires_grad and not teacher_logp.requires_grad
    values=teacher_logp-student.log_prob(actions)
    assert torch.isfinite(values).all(), 'NONFINITE_MC_KL'
    per_window=values.mean(dim=0)
    return per_window.mean(),per_window


def forward_distribution(model,data):
    with deterministic_distribution(model):
        distribution,_=model(data['feature'].squeeze(1))
        fields(distribution)
        return distribution
