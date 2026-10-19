"""Exactly four isolated optimizer calls, no environment construction/steps."""
from .model import *
from torch.utils.data import DataLoader,Subset
from coreset_exp.model import data_setup,complete
import robomimic.utils.tensor_utils as TU

def main():
    assert not (O/'INTEGRATION.json').exists() and not (O/'integration_calls.jsonl').exists()
    assert js(O/'ADMISSION.json')['status']=='PASS';setup();start=time.time()
    p,t0=new_public();load_stage(p,7);bank,geometry=make_bank(7)
    public_before=tensor_hash(p);t0_before=tensor_hash(t0)
    def data(t):
        # Task6 integration probes are retained parent memory, not discarded windows.
        files=sorted((P/PARM/'buffer').glob('t6__*.pt')) if t==6 else sorted((P/'features/7').glob('*.pt'))
        rows=[torch.load(f,map_location='cpu') for f in files[:10]]
        return torch.cat([r['feature'] for r in rows]).cuda(),torch.cat([r['actions'] for r in rows]).cuda()
    h6,a6=data(6);h7,a7=data(7)
    h=torch.cat([h6[:5],h7[:5]]);a=torch.cat([a6[:5],a7[:5]]);ids=torch.tensor([6]*5+[7]*5,device='cuda')
    with torch.no_grad():
        native=p.policy_head(p.temporal_encode(h)[:,-1]);zero=decode(p,t0,bank,h)
        assert torch.equal(native.component_distribution.base_dist.loc,zero['means'])
        assert torch.equal(native.component_distribution.base_dist.scale,zero['scales'])
        assert torch.equal(native.mixture_distribution.probs,zero['gmm'].mixture_distribution.probs)
        # Nonzero synthetic positive control, never enters formal initialization.
        positive=copy.deepcopy(bank);positive.branches['6'].U.fill_(.1)
        plus=decode(p,t0,positive,h);assert float((plus['means'][:5]-zero['means'][:5]).abs().max())>1e-5
        assert float((plus['means'][5:]-zero['means'][5:]).abs().max())<=1e-6
        assert float((plus['means']-zero['means']).abs().max())<=.100001
        assert torch.equal(plus['scales'],zero['scales']) and torch.equal(plus['logits'],zero['logits'])
        assert torch.equal(plus['selected'],zero['selected'])
    opt,sch=optimizer(bank,10*BATCHES[1])
    def call(hh,aa,ii):
        append(O/'integration_calls.jsonl',dict(event='before_optimizer_call',index=len((O/'integration_calls.jsonl').read_text().splitlines()) if (O/'integration_calls.jsonl').exists() else 0,time=time.time()))
        return update(p,t0,bank,opt,sch,hh,aa,ii)
    r0=call(h6,a6,torch.full((10,),6,device='cuda'))
    r1=call(h7,a7,torch.full((10,),7,device='cuda'))
    snap=dict(bank=cpu(bank.state_dict()),optimizer=cpu(opt.state_dict()),scheduler=cpu(sch.state_dict()),rng=cpu(rng()))
    save(O/'integration_recovery.pth',snap)
    r2=call(h,a,ids);after=cpu(bank.state_dict());after_opt=cpu(opt.state_dict())
    saved=torch.load(O/'integration_recovery.pth',map_location='cpu');bank.load_state_dict(saved['bank']);opt.load_state_dict(saved['optimizer']);sch.load_state_dict(saved['scheduler']);restore_rng(saved['rng'])
    r3=call(h,a,ids)
    for n,v in after.items():assert torch.equal(v,bank.state_dict()[n].cpu()),n
    for key,v in after_opt['state'].items():
        for field,value in v.items():assert torch.equal(value,opt.state_dict()['state'][key][field].cpu())
    assert tensor_hash(p)==public_before and tensor_hash(t0)==t0_before
    # Actual CompletePolicy.get_action, real demonstration images, no simulator.
    c=cfg('eval');b,_,shape=data_setup(c,False);c.shape_meta=shape
    from libero.lifelong.datasets import get_dataset,SequenceVLDataset
    d,_=get_dataset(str(Path(c.folder)/b.get_task_demonstration(7)),c.data.obs.modality,initialize_obs_utils=False,seq_len=8,demo_limit=10,train=False)
    tok,mask=b.get_task_emb(7);d=SequenceVLDataset(d,tok,mask)
    rawdata=TU.map_tensor(next(iter(DataLoader(Subset(d,[0]),batch_size=1))),lambda x:x.cuda())
    full=complete(c,shape).eval().requires_grad_(False);strict_downstream(full,js(O/'PARENT_BINDING.json')['checkpoints']['7']['path'])
    ctx=install_native(full,t0,bank);native_rows=[]
    with torch.no_grad():
        for length in [1,4,8]:
            batch=copy.deepcopy(rawdata);batch['obs']={k:v[:,:length].contiguous() for k,v in batch['obs'].items()}
            r=cpu(rng());action=full.get_action(batch);out=ctx['out'];hh=ctx['h'];separate=decode(p,t0,bank,hh)
            for key in ['means','scales','logits','selected']:assert torch.equal(out[key],separate[key]),(length,key)
            restore_rng(r);expected=selected_dist(separate).sample().cpu().numpy();assert np.array_equal(action,expected)
            native_rows.append(dict(length=length,distribution_exact=True,sampled_action_exact=True))
    assert ctx['spatial_calls']==3
    assert tensor_hash(p)==public_before and tensor_hash(t0)==t0_before
    write(O/'INTEGRATION.json',dict(status='PASS',optimizer_calls=4,environment_steps=0,real_history=native_rows,updates=[r0,r1,r2,r3],recovery_tensor_and_optimizer_exact=True,absent_branch_none_and_adam_unchanged=True,public_hash=public_before,T0_hash=t0_before,T0_bytes=sum(v.numel()*v.element_size() for v in t0.state_dict().values()),public_parameters_frozen=True,mean_bound=.1,seconds=time.time()-start))
    print('INTEGRATION PASS: four optimizer calls, zero environment steps',flush=True)
if __name__=='__main__':main()
