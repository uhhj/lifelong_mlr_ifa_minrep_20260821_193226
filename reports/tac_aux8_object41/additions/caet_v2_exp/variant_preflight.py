"""Additional isolated GPU mechanics; no formal steps or environments."""
import copy,json,math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from coreset_exp.common import O,SEED,write,save,verify,rng,restore_rng
from coreset_exp.model import cfg
from coreset_exp.train import build,PaperMLR,capture,restore
from libero.lifelong.algos.mlr import build_dataset_for_task
from .dataset import EffectDataset
from .observe import CAETv2MLR
from .preflight import equal
from .variant import VARIANT,exec_params,exec_kl,feature

@torch.no_grad()
def input_shift(policy,ds,name):
    ids=[i for i,r in enumerate(ds.rows) if r['valid'] and r['task']==ds.task and r['holdout']]
    ids=ids[:128];assert ids
    saved=rng();modes={n:m.training for n,m in policy.named_modules()};policy.eval();sums=dict(kl=0.,mode_switch=0.,mean_action_rms=0.,gripper_abs=0.)
    try:
        for j in range(0,len(ids),32):
            ix=ids[j:j+32];x=torch.stack([ds.native[i][0]['feature'].squeeze(0) for i in ix]).cuda()
            full=exec_params(policy,x);short=exec_params(policy,x[:,:7])
            sums['kl']+=float(exec_kl(full,short).double().sum());sums['mode_switch']+=float((full[2]!=short[2]).double().sum())
            sums['mean_action_rms']+=float((full[0]-short[0]).square().mean(-1).sqrt().double().sum());sums['gripper_abs']+=float((full[0][:,-1]-short[0][:,-1]).abs().double().sum())
    finally:
        for n,m in policy.named_modules():m.training=modes[n]
        restore_rng(saved)
    assert all(math.isfinite(v) for v in sums.values())
    out=dict(samples=len(ids),selection='first128 legal current holdout samples in native fixed order',metrics={k:v/len(ids) for k,v in sums.items()},interpretation='Distribution difference measured, not equivalence or rollout guarantee',model_forwards=2*math.ceil(len(ids)/32))
    write(O/'input_shift'/name,out);return out

def main():
    verify();assert not (O/'VARIANT_GPU_PREFLIGHT.json').exists()
    dist.init_process_group('nccl');torch.cuda.set_device(0);calls=0
    try:
        c=cfg('ra_coreset');c.run_root=str(O/'preflight/variant');c.buffer_dir=str(O/'preflight/variant/buffer');c.experiment_dir=str(O/'preflight/variant/models')
        a,_=build(c);a.__class__=CAETv2MLR
        native=build_dataset_for_task(c.feature_dir,6,c.buffer_dir,SEED,start=6);ds=EffectDataset(native,6);loader=DataLoader(ds,batch_size=10,num_workers=0)
        a.start_task(-1,loader);a.refs={6:native[0][0]['feature'][0,0,0].cuda()};a.configure_effect(6,ds)
        batch=next(iter(loader));initial=capture(a);result=dict(variant=VARIANT)
        if VARIANT=='TAC':
            result['input_shift']=input_shift(a.policy.module,ds,'initial_task6.json')
            ix=[i for i,r in enumerate(ds.rows) if r['valid']][:10]
            x=torch.stack([ds.native[i][0]['feature'].squeeze(0) for i in ix]).cuda();target=torch.stack([ds.rows[i]['next_action'] for i in ix]).cuda()
            a.policy.eval();z=a.policy.module.temporal_encode(feature(x))[:,-1];loss=-a.policy.module.policy_head(z).log_prob(target).mean()
            grads=torch.autograd.grad(loss,[p for p in a.policy.parameters() if p.requires_grad],allow_unused=True)
            assert all(g is None or torch.isfinite(g).all() for g in grads) and any(g is not None and float(g.abs().sum())>0 for g in grads)
            result['seven_step_gradient_finite_nonzero']=True
        elif VARIANT=='RSET':
            from .model import EffectModel
            m=EffectModel().cuda();x=batch[0]['feature'].cuda();action=batch[0]['next_action'].cuda();opt=torch.optim.AdamW(m.parameters(),lr=.001)
            before=copy.deepcopy(m.state_dict());order=x.flip(2) if x.ndim==5 else x.flip(1)
            assert not torch.equal(m(x,action)[0],m(order,action)[0]),'Order must matter with positional encoding'
            loss=m(x,action)[0].square().mean();opt.zero_grad();loss.backward();opt.step();calls+=1
            assert all(torch.isfinite(p).all() for p in m.parameters()) and any(not torch.equal(before[k],v) for k,v in m.state_dict().items())
            result['temporal_order_sensitive']=True;result['predictor_isolated_update']=True
        else:
            from .eftr import capture as cap,restore as res,decision
            a.reference=copy.deepcopy(a.policy.module).eval().requires_grad_(False)
            start=cap(a)
            PaperMLR.observe(a,copy.deepcopy(batch),6,0,{'selected_pairs':[]});calls+=1;native_end=cap(a)
            res(a,start);a.observe(copy.deepcopy(batch),6,0,{'selected_pairs':[]});calls+=2
            assert equal(native_end,cap(a)),'Zero auxiliary candidate must commit exact native state'
            first=cap(a);res(a,start);a.observe(copy.deepcopy(batch),6,0,{'selected_pairs':[]});calls+=2
            assert equal(first,cap(a)),'Candidate replay must be exact'
            # Construct a harmful real optimizer update on the native policy,
            # score it against the unchanged previous policy on old samples.
            res(a,start);x=a.map_tensor_to_device(copy.deepcopy(batch[0]));mask=torch.zeros(len(batch[1]),device='cuda',dtype=torch.bool)
            a.policy.eval();m,s,_=exec_params(a.policy.module,x['feature']);target=(m.detach()+1.)
            opt=torch.optim.SGD([p for p in a.policy.parameters() if p.requires_grad],lr=.1)
            loss=(m-target).square().mean();opt.zero_grad();loss.backward();opt.step();calls+=1
            bad=a.metrics(x,mask);ok,reasons=decision(dict(bc=None,kl=0.,switch=0.,gripper=0.),bad)
            assert not ok and reasons,('Harmful candidate did not trigger',bad)
            res(a,native_end);assert equal(cap(a),native_end),'Rejected candidate must restore native optimizer and all state'
            result.update(zero_aux_native_exact=True,candidate_replay_exact=True,harmful=bad,rejection=reasons,rejected_restoration_exact=True)
        restore(a,initial)
    finally:dist.destroy_process_group()
    verify();result.update(status='PASS',isolated_optimizer_calls=calls,formal_updates=0,environment_steps=0,gpu=torch.cuda.get_device_name())
    write(O/'VARIANT_GPU_PREFLIGHT.json',result)

if __name__=='__main__':main()
