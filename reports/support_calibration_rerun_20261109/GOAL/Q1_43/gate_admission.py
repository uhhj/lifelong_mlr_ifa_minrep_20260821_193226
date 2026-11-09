"""One new-stage branch only. Full-state candidate comparison, finite budget."""
import argparse,io,math
from runtime import *
from support_memory import load_memory,weights
from updates import update
from legacy_updates import update as original_update
from fork_tools import setup_probes,probe,choose,directional_diagnostic
import torch.distributed as dist

def scale_for(ds):
    y=torch.stack([torch.as_tensor(ds.rows[i]['next_action']).float() for i in ds.ids])
    return torch.stack([y[:,lo:hi].square().mean().sqrt().clamp_min(.1) for lo,hi in [(0,3),(3,6),(6,7)]]).cuda()

def step(a,b,arm,m,scale,info):
    return update(a,b,arm,m,scale,info,STAGE,retain_weight=0. if arm=='C' else RETAIN_WEIGHT)

def admission(a,ds,m,scale,info,pr,arm):
    ids=ds.ids[:6]+[i for i,r in enumerate(ds.rows) if r['task']==STAGE][:4]
    b=torch.utils.data.default_collate([ds[i] for i in ids]);initial=capture(a);calls=0
    step(a,b,'C',m,scale,info);cc=capture(a);calls+=1
    restore(a,initial);original_update(a,b,'S',m,scale,info,STAGE,retain_weight=0.,legacy=True);assert equal(cc,capture(a));calls+=1
    restore(a,initial);step(a,b,'S',m,scale,info);ss=capture(a);calls+=1
    restore(a,initial);original_update(a,b,'S',m,scale,info,STAGE);assert equal(ss,capture(a));calls+=1
    ones=copy.deepcopy(m)
    for v in ones.values():v['support']=1.
    restore(a,initial);step(a,b,'Q',ones,scale,info);assert equal(ss,capture(a));calls+=1
    restore(a,initial);first=step(a,b,'Q' if arm=='G' else arm,m,scale,info);one=capture(a);calls+=1
    assert equal(one,ss),'Q1 actual-memory update must equal uniform legacy retention'
    stream=io.BytesIO();torch.save(one,stream);stream.seek(0);loaded=torch.load(stream,map_location='cpu');assert equal(one,loaded)
    step(a,b,'Q' if arm=='G' else arm,m,scale,info);two=capture(a);calls+=1
    restore(a,loaded);step(a,b,'Q' if arm=='G' else arm,m,scale,info);assert equal(two,capture(a));calls+=1
    restore(a,one);dg=directional_diagnostic(a,b,m,pr,arm);assert dg is not None
    step(a,b,'Q' if arm=='G' else arm,m,scale,info);assert equal(two,capture(a));calls+=1
    for chosen in [cc,ss]:restore(a,chosen);assert equal(chosen,capture(a))
    restore(a,initial);pv=probe(a,pr,m);assert equal(initial,capture(a))
    from fork_tools import auxiliary
    from support_memory import retention
    short=[]
    xx=a.map_tensor_to_device(b[0])['feature'].squeeze(1)[:1]
    with auxiliary(a):
        for length in range(1,9):
            x=xx[:,:length];qq=a.policy.module.policy_head.forward_fn(a.policy.module.temporal_encode(x)[:,-1]);kk=int(qq[2][0].argmax())
            fake={t:dict(rows=[dict(feature=x[0].detach().cpu(),mu=qq[0][0,kk].detach().cpu(),sd=qq[1][0,kk].detach().cpu())],scales=torch.ones(3),support=.1) for t in range(6,STAGE)}
            rx,rt,_=retention(a.policy.module,fake,STAGE,a.step,'Q');sx,st,_=retention(a.policy.module,fake,STAGE,a.step,'S')
            assert torch.equal(rx,sx) and torch.equal(rt,st)
            grads=torch.autograd.grad(rx+.1*rt,[p for p in a.policy.parameters() if p.requires_grad],allow_unused=True)
            assert all(v is None or torch.isfinite(v).all() for v in grads)
            if length==8:assert all(torch.equal(u,v) for u,v in zip(qq,params(a.policy.module,x)))
            short.append(dict(length=length,raw_q_point_one_ignored=True,effective_q_one_exact=True,finite_gradient=True))
    assert equal(initial,capture(a))
    assert choose(pv,pv)[0]=='C'
    better=dict(pv,retain_support=pv['retain_support']-1.);assert choose(pv,better)[0]=='Q'
    worse=dict(better,exec_new=pv['exec_new']+1.);assert choose(pv,worse)[0]=='C'
    # A 10-step branch must preserve complete replay, not only a single model step.
    base=capture(a)
    for _ in range(10):step(a,b,'C',m,scale,info)
    c10=capture(a);calls+=10
    restore(a,base)
    for _ in range(10):original_update(a,b,'S',m,scale,info,STAGE,retain_weight=0.,legacy=True)
    assert equal(c10,capture(a));calls+=10
    restore(a,initial);assert equal(initial,capture(a))
    return dict(status='PASS_OWN_GPU_ADMISSION',optimizer_calls=calls,roundtrip_next_step_exact=True,q_one_original_S_exact=True,lambda_zero_original_C_exact=True,ten_step_C_replay_exact=True,accepted_and_rejected_full_state_restore=True,measurement_state_exact=True,probe_all_state_exact=True,q1_actual_memory_uniform_exact=True,q1_short_history_exact=True,synthetic_gate_accept_reject=True,short_history=short,extra_autograd_calls=11,diagnostic=dg,probe=pv,environments=0)

