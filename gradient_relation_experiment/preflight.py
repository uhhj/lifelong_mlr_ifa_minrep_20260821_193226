"""Real-policy same-state checks before any scientific counterfactual label."""
import time
import traceback
from gradient_relation_experiment.kernel import *
from gradient_relation_experiment.prepare import OUT,batch_indices


def load_banks(algo,design,tasks):
    banks={}
    for task in tasks:
        for bank,rows in design['monitor_membership'][str(task)].items():
            samples=[]
            for row in rows:
                assert sha(ROOT/row['path'])==design['input_sha256'][row['path']]
                samples.append(torch.load(ROOT/row['path'],map_location='cpu'))
            banks[bank,task]=[algo.map_tensor_to_device(default_collate(samples[i:i+10])) for i in range(0,len(samples),10)]
    return banks


def toy_checks():
    # Known contribution-removal arithmetic without renormalizing the survivors.
    gnew=torch.tensor([2.,-4.]);gold=torch.tensor([-3.,1.]);q=torch.tensor([.1,.2]);a=.6;b=.2
    grads={'11':q+a*gnew+b*gold,'10':q+a*gnew,'01':q+b*gold,'00':q}
    assert torch.allclose(grads['11']-grads['10'],b*gold)
    assert torch.allclose(grads['11']-grads['01'],a*gnew)
    # Zero LR changes moments; zero gradient with established moments still updates parameters.
    p=torch.nn.Parameter(torch.tensor([1.,2.]));opt=torch.optim.AdamW([p],lr=0.,weight_decay=.1)
    p.grad=torch.ones_like(p);opt.step();assert torch.equal(p.detach(),torch.tensor([1.,2.]))
    assert float(opt.state[p]['exp_avg'].abs().sum())>0
    opt.param_groups[0]['lr']=1e-3;old=p.detach().clone();p.grad=torch.zeros_like(p);opt.step()
    assert not torch.equal(old,p.detach())
    # Reference zero is missing information, not zero conflict.
    assert cosine(torch.zeros(2),torch.ones(2)) is None
    # Clipping precedes optimizer; scheduler follows step.
    p.grad=torch.tensor([300.,400.]);torch.nn.utils.clip_grad_norm_([p],100)
    assert abs(float(p.grad.norm())-100)<1e-4
    return {'contribution_weights':'PASS','zero_lr_moments':'PASS','zero_gradient_not_skip':'PASS','zero_reference_missing':'PASS','clip':'PASS'}


def run():
    import gzip
    dest=OUT/'preflight';dest.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'PREFLIGHT_COMPLETE.json').exists()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    unit=toy_checks();init_ddp(dest/'ddp_store')
    with gzip.open(OUT/'DESIGN.json.gz','rt') as f:design=json.load(f)
    source=next(p for p in design['points'] if p['task']==8 and p['new_present'] and len(p['old_present'])==2)
    seed=source['seed'];plan=read(ROOT/design['source_plan_sha256'][f'{seed}/8']['path'])
    ds=CachedWindows(plan['immutable_feature_origins'],plan['record_sha256'])
    cfg=train_config(seed,dest);bootstrap_train(cfg);algo=build_train(cfg)
    sampler=torch.utils.data.DistributedSampler(ds,num_replicas=1,rank=0,shuffle=True,seed=0)
    dl=torch.utils.data.DataLoader(ds,batch_size=10,sampler=sampler,num_workers=0)
    with contextlib.redirect_stdout(io.StringIO()):algo.start_task(-1,dl)
    algo.policy.train()
    for i in range(4):native(algo,batch(ds,batch_indices(plan,i)),8,0)
    assert algo._optimizer_step_count==4 and algo.optimizer.param_groups[0]['lr']>0
    banks=load_banks(algo,design,(6,7,8))
    p=copy.deepcopy(source);p['point']='DISPOSABLE_GPU_PREFLIGHT';p['epoch']=0
    rows=[];begin=time.monotonic();witnesses=run_point(algo,ds,plan,p,banks,lambda kind,row:rows.append((kind,row)))
    # A second independently replayed native 20-step path must equal saved 11 witnesses.
    for h in range(1,21):
        step=p['stage_step']+h-1
        if h>1:ensure_next_epoch_rng(algo,ds,plan,step-1,step)
        native(algo,batch(ds,p['future20_record_indices'][h-1]),8,step//plan['batches_per_epoch'])
        if h in witnesses:
            s=capture(algo);assert all(digest(s[k])==v for k,v in witnesses[h].items()),('20step carrier parity',h)
    # Verify exact base_seed consumption at an epoch boundary against DataLoader.
    state=rng_state();iter(dl);expected=rng_state();restore_rng(state)
    torch.empty((),dtype=torch.int64).random_();assert digest(rng_state())==digest(expected)
    geometry=[r for k,r in rows if k=='geometry']
    assert len(geometry)==24 and {r['arm'] for r in geometry}=={'11','10','01','00'}
    # Explicit q=0 task7 test exercises the empty objective through real AdamW.
    indices=next(p for p in design['points'] if p['task']==7 and p['new_present'])
    plan7=read(ROOT/design['source_plan_sha256'][f'{indices["seed"]}/7']['path'])
    ds7=CachedWindows(plan7['immutable_feature_origins'],plan7['record_sha256'])
    state=capture(algo);fw=first_gradients(algo,batch(ds7,indices['future20_record_indices'][0]),7)
    assert norm(fw['gradients'][6]['00'])==0
    restore(algo,state);apply_first(algo,fw['gradients'][6]['00'],fw)
    assert norm(delta(algo,state))>0
    result={'status':'PASS','time':time.time(),'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,
        'unit':unit,'native_vs_diagnostic_11_bitwise_equal':True,'full20_native_carrier_parity':True,
        'full_state_restore':True,'monitor_repeat_exact':True,'epoch_rng_parity':True,
        'q_zero_real_adamw_not_skipped':True,'new_objective_renormalization':False,
        'scientific_points_evaluated':0,'disposable_full_point_seconds':time.monotonic()-begin,
        'real_parameter_count':sum(p.numel() for p in algo.policy.parameters()),'design_sha256':sha(OUT/'DESIGN.json.gz')}
    write(OUT/'PREFLIGHT_COMPLETE.json',result);print('PREFLIGHT_PASS',result,flush=True)
    torch.distributed.destroy_process_group()


if __name__=='__main__':
    try:run()
    except Exception:
        write(OUT/'PREFLIGHT_FAILED.json',{'time':time.time(),'traceback':traceback.format_exc()});raise
