import copy,io,json,time,types
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset
import robomimic.utils.tensor_utils as TU
from .common import *
from .train import build,stage
from .model import cfg,readout,attach,load_stage,complete,data_setup
from .diagnostics import compare
from coreset_exp.train import capture,restore
from libero.lifelong.models.brief import BriefPolicy
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def main():
    verify(False);assert not(O/'INTEGRATION.json').exists();assert not(O/'integration_optimizer_calls.jsonl').exists(),'Existing integration attempt: inspect budget before recovery'
    start=time.time();dist.init_process_group('nccl');torch.cuda.set_device(0)
    try:
        c=cfg('train');c.experiment_dir=str(O/'integration/models');c.buffer_dir=str(O/'integration/buffer')
        a,_=build(c);p=a.policy
        initial=dict(rng=rng_digest(),tensors={'module.'+k:v for k,v in digest(p.state_dict()).items()})
        assert initial==json.loads((P/'coreset/INITIAL_STATE.json').read_text())
        tests=dict(initial_public_and_rng_exact=True);birth6=stage(a,6);p=a.policy.module
        choices=json.loads((PROTO/'SAMPLE_IDS.json').read_text());payload=[]
        for t in (6,7):
            for n in choices['incremental'][str(t)]['probe'][:5]:payload.append((torch.load(P/'features'/str(t)/n,map_location='cpu'),str(t)))
        loader=DataLoader(payload,batch_size=10,num_workers=0);mixed=next(iter(loader));single=next(iter(DataLoader(payload[5:]*2,batch_size=10)));old=next(iter(DataLoader(payload[:5]*2,batch_size=10)))
        h=mixed[0]['feature'].squeeze(1).cuda();actions=mixed[0]['actions'].squeeze(1).cuda();r=rng();p.train()
        native=BriefPolicy.forward(p,h);native_loss=p.policy_head.loss_fn(native[0],actions,'mean');native_loss.backward()
        native_grad={n:v.grad.detach().clone() for n,v in p.named_parameters() if v.grad is not None};p.zero_grad();restore_rng(r)
        new=p(h);new_loss=p.policy_head.loss_fn(new[0],actions,'mean');new_loss.backward()
        assert torch.allclose(native_loss,new_loss,atol=1e-6,rtol=1e-6)
        gradmax=max(float((v-p.get_parameter(n).grad).abs().max()) for n,v in native_grad.items())
        assert all(torch.allclose(v,p.get_parameter(n).grad,atol=1e-6,rtol=1e-5) for n,v in native_grad.items())
        tests.update(zero_output_and_native_grad_parity=True,public_gradient_max_abs=gradmax,public_loss_abs=float((native_loss-new_loss).abs()))
        p.zero_grad()
        # Nonzero isolated old branch tests Jacobian and optimizer exclusion. Never used formally.
        with torch.no_grad():p.residual_bank['6'].U.fill_(.001)
        birth7=stage(a,7);p=a.policy.module;frozen=digest(p.residual_bank['6'].state_dict())
        z=p.temporal_encode(h)[:,-1].detach().requires_grad_(True);v=p.residual_bank['6'](z,h[:,-1,0]);g=torch.autograd.grad(v.sum(),z)[0]
        assert g.abs().max()>0 and not p.residual_bank['6'].U.requires_grad
        tests['old_branch_z_gradient_nonzero']=float(g.abs().max())
        a.start_task(7,loader);a.refs={t:payload[(t-6)*5][0]['feature'][0,0,0].cuda() for t in (6,7)};info={'selected_pairs':[(6,7)]}
        assert all(p.residual_bank['6'].U is not v and p.residual_bank['6'].V is not v for group in a.optimizer.param_groups for v in group['params'])
        # No-current batch path without an optimizer call.
        d=a.map_tensor_to_device(copy.deepcopy(old[0]));loss,z=p.compute_loss(d,None,None);assert torch.isfinite(loss)
        grads=[]
        def update(batch):
            append(O/'integration_optimizer_calls.jsonl',dict(event='before_call',number=len(grads)+1))
            lr=a.optimizer.param_groups[0]['lr'];loss=a.observe(copy.deepcopy(batch),7,0,info)
            grad={n:float(v.grad.norm()) if v.grad is not None else None for n,v in p.named_parameters() if n.startswith('residual_bank.')}
            grads.append(dict(loss=loss,lr=lr,grad=grad));assert digest(p.residual_bank['6'].state_dict())==frozen
            assert all(v.grad is None for v in p.residual_bank['6'].parameters())
            assert any(v.grad is not None and v.grad.abs().max()>0 for n,v in p.named_parameters() if n.startswith('temporal_transformer.'))
            assert any(v.grad is not None and v.grad.abs().max()>0 for n,v in p.named_parameters() if n.startswith('policy_head.'))
            return loss
        update(mixed);update(single);snap=capture(a);stream=io.BytesIO();torch.save(snap,stream);snap=None
        first=update(mixed);expected=capture(a);stream.seek(0);restore(a,torch.load(stream,map_location='cpu'));second=update(mixed)
        assert first==second and equal(expected,capture(a));assert len(grads)==4 and a.step==3
        assert grads[2]['grad']['residual_bank.7.U']>0 and grads[2]['grad']['residual_bank.7.V']>0
        tests.update(exact_serialized_optimizer_scheduler_rng_recovery=True,single_mixed_no_current_finite=True,old_branch_frozen=True,current_U_V_gradients=True)
        # Current train result reloaded into native frontend; no environment constructed.
        tmp=O/'integration/stage7.pth';save(tmp,dict(state_dict=cpu(a.policy.state_dict()),task=7,epoch_index=99,branches=[6,7]))
        ec=cfg('eval');b,ds,shape=data_setup(ec,True);ec.shape_meta=shape;online=complete(ec,shape);load_stage(online,tmp,7);online.eval();p.eval()
        torch.backends.cudnn.benchmark=True;native_rows=[]
        for t in (6,7):
            raw=TU.map_tensor(next(iter(DataLoader(Subset(ds[t],choices['raw'][str(t)][:2]*5),batch_size=10))),lambda v:v.cuda())
            for length in (1,4,8):
                d=copy.deepcopy(raw);d['obs']={k:v[:,:length].contiguous() for k,v in d['obs'].items()};d['actions']=d['actions'][:,:length]
                with torch.no_grad():
                    spatial=online.spatial_encode(online.preprocess_input(d,train_mode=False));aa=readout(online,online.temporal_encode(spatial)[:,-1],spatial[:,-1,0]);bb=readout(p,p.temporal_encode(spatial)[:,-1],spatial[:,-1,0])
                result=compare(aa,bb);assert result['passed'];native_rows.append(dict(task=t,length=length,**result))
        write(O/'NATIVE_PATH_PARITY.json',dict(status='PASS',rows=native_rows,strict_all_stage_residual_keys=True,gate_dtype='float64',environment_steps=0))
        write(O/'INTEGRATION.json',dict(status='PASS',tests=tests,updates=grads,optimizer_calls=4,effective_steps=3,formal_steps=0,environment_steps=0,branch_birth=[birth6,birth7],seconds=time.time()-start,numeric=numeric(),serialized_test_checkpoint_sha256=sha(tmp)))
        print('INTEGRATION PASS: 4 calls, 0 environment steps',flush=True)
    finally:dist.destroy_process_group()
if __name__=='__main__':main()
