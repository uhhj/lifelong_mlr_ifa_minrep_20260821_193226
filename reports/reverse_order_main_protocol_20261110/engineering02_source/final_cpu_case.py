"""Independent post-completion CPU reconstruction; no optimizer or environment."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import contextlib,copy,hashlib,io,json,math,random,sys,time
Q=Path(__file__).resolve().parent;case=sys.argv[1];R=Q/'cases'/case
J=lambda p:json.loads(Path(p).read_text())
inp=J(R/'INPUT.json');O=Path(inp['input_root']);out=Path(inp['output_root']);method=inp['method']
sys.path.insert(0,str(Q));sys.path.insert(0,str(R));sys.path.insert(0,str(R/'source') if method=='MLR' else inp['source_root'])
os.environ.update(CORESET_OUT=str(O),CORESET_SEED=str(inp['seed']),CAET_VARIANT='ALIGN')
import numpy as np
import torch
from easydict import EasyDict
from order import install,verify_mapping,ORIGINAL_TASKS
install()
from reverse_common import sha,write,verify_case
from libero.lifelong.models.brief import BriefPolicy
assert (Q/'controller.exit').read_text().strip()=='0'
assert (R/'train.exit').read_text().strip()==(R/'evaluate.exit').read_text().strip()=='0'
torch.set_num_threads(1)
assert not torch.cuda.is_initialized()
tic=time.time();contract=verify_case(R)
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def finite(x):
    if torch.is_tensor(x):assert torch.isfinite(x).all()
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,(list,tuple)):
        for v in x:finite(v)
def ah(x):return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()
class Model(torch.nn.Module):
    def __init__(self,cfg):
        super().__init__();self.module=BriefPolicy(copy.deepcopy(cfg),None)
def construct(cfg,sd):
    with contextlib.redirect_stdout(io.StringIO()):m=Model(cfg)
    assert len(m.state_dict())==len(sd)==100
    m.load_state_dict(sd,strict=True);assert equal(m.state_dict(),sd)
    m.eval();return m
@torch.no_grad()
def infer(m,feature):
    if feature.ndim==5:feature=feature.squeeze(1)
    h=m.module.temporal_encode(feature)[:,-1]
    y=m.module.policy_head.forward_fn(h);finite(y)
    return tuple(v.detach().cpu().clone() for v in y)
def lr(step,total):return step/500 if step<500 else max(0.,(total-step)/max(1,total-500))
cfg=EasyDict(copy.deepcopy(inp['train_config']));cfg.device='cpu'
mapping=verify_mapping(cfg)
assert J(R/'ACTUAL_ORDER.json')['mapping']==mapping
assert sha(cfg.pretrain)==inp['parent']['official_pretrain_sha256']
if method=='MLR':
    hist=J(out/'epochs.json');complete=J(out/'TRAINING_COMPLETE.json');initial=J(R/'INITIAL_BINDING.json')
    assert initial['status']=='PASS_OWN_INITIAL_TENSORS_AND_NONCUDA_RNG'
    cp_paths={s:out/'models'/('mlrifa_%d_model_ep99.pth'%s) for s in range(6,10)}
    recovery_path=out/'recovery/latest.pth'
else:
    hist=J(R/'EPOCHS_S.json');complete=J(R/'TRAIN_S.json');initial=J(R/'INITIAL_S.json')
    assert initial['status']=='PASS_OFFICIAL_INITIALIZATION'
    cp_paths={s:out/('stage%d.pth'%s) for s in range(6,10)}
    recovery_path=out/'recovery.pth'
assert len(hist)==complete['epochs']==400 and complete['steps']==inp['accepted_budget']
assert initial['tensors']==inp['initial_state']['tensors']
assert all(initial['rng'][k]==inp['initial_state']['rng'][k] for k in initial['rng'] if k!='cuda')
calls=J(R/'OPTIMIZER_CALLS.json');assert calls['status']=='ACTUAL_CALLS_COMPLETED'
assert calls['calls']==dict(production=inp['physical_budget'],admission=20 if method=='MLR' else 91)
assert calls['extra_autograd']['admission']==(0 if method=='MLR' else 41)
if method=='MLR':assert calls['extra_autograd']['production']==0
else:assert calls['extra_autograd']['production']<=36
checkpoints={};cumulative=0;model_calls=0;models={}
for stage in range(6,10):
    pop=inp['population'][str(stage)];batches=inp['batches'][str(stage)];total=batches*100
    rows=hist[(stage-6)*100:(stage-5)*100]
    for epoch,row in enumerate(rows):
        assert row['task' if method=='MLR' else 'stage']==stage and row['epoch']==epoch and row['batches']==batches
        assert row['steps']==cumulative+(epoch+1)*batches
        order=torch.randperm(pop,generator=torch.Generator().manual_seed(epoch)).numpy().astype(np.int64)
        assert row['sampler_sha256']==ah(order)
        assert math.isclose(row['lr_first'],cfg.train.optimizer.kwargs.lr*lr(epoch*batches,total),rel_tol=1e-12,abs_tol=1e-15)
        assert math.isclose(row['lr_after'],cfg.train.optimizer.kwargs.lr*lr((epoch+1)*batches,total),rel_tol=1e-12,abs_tol=1e-15)
        if method=='DEBR':
            counts=J(R/'INPUT_COUNTS.json')
            assert row['totals']['valid']==counts['valid'][str(stage)] and row['totals']['invalid']==pop-counts['valid'][str(stage)]
            assert row['totals']['old_eligible']==counts['old_valid'][str(stage)]
    raw=torch.load(cp_paths[stage],map_location='cpu');sd=raw['state_dict'];finite(sd)
    assert raw['task' if method=='MLR' else 'stage']==stage
    assert raw['epoch_index' if method=='MLR' else ('epoch' if stage<9 else 'next_epoch')]==(100 if method=='DEBR' and stage==9 else 99)
    if method=='MLR':
        assert raw['contract_sha256']==sha(O/'CONTRACT.json') and raw['modal_sha256']==sha(O/'MODAL_CONTRACT.json')
        assert complete['checkpoints'][cp_paths[stage].name]==sha(cp_paths[stage])
    else:
        assert raw['contract']==contract and complete['checkpoints'][str(stage)]==sha(cp_paths[stage])
        assert J(R/('STAGE_S_%d.json'%stage))['sha256']==sha(cp_paths[stage])
    model=construct(cfg,sd)
    stream=io.BytesIO();torch.save(sd,stream);stream.seek(0);loaded=torch.load(stream,map_location='cpu');assert equal(sd,loaded)
    again=construct(cfg,loaded)
    man=J(Path(inp['manifests'][str(stage)]['path']))
    sample=torch.cat([torch.load(man['rows'][i]['path'],map_location='cpu')['feature'] for i in [0,len(man['rows'])-1]],dim=0)
    before=torch.get_rng_state().clone();one=infer(model,sample);two=infer(again,sample)
    assert equal(one,two) and torch.equal(before,torch.get_rng_state());model_calls+=2
    checkpoints[str(stage)]=dict(sha256=sha(cp_paths[stage]),strict_keys=100,parameters=sum(p.numel() for p in model.parameters()),strict_load=True,cpu_serialization_and_forward_exact=True,original_current_task=ORIGINAL_TASKS[stage])
    if method=='DEBR':models[stage]=model
    elif stage==9:models[stage]=model
    if method=='MLR':
        admission=J(R/('ADMISSION_MLR_%d.json'%stage))
        assert admission['status']=='PASS_OWN_MAIN_MLR_GPU_ADMISSION' and admission['optimizer_calls']==5
        assert all(admission[k] for k in ['direct_first_update_exact','direct_second_update_exact','serialized_next_step_exact','all_state_restored','random_state_restored','actual_population_order_task_and_hash_exact'])
    else:
        admission=J(R/('ADMISSION_S_%d.json'%stage));assert admission['status']==('PASS' if stage==6 else 'PASS_OWN_GPU_ADMISSION')
        assert admission['optimizer_calls']==(4 if stage==6 else 29)
        if stage==6:assert admission['roundtrip_exact'] and admission['next_step_exact'] and admission['restore_initial_exact']
        else:assert admission['roundtrip_next_step_exact'] and admission['ten_step_C_replay_exact'] and admission['accepted_and_rejected_full_state_restore'] and admission['measurement_state_exact']
    cumulative+=total
    del raw,sd,again,stream,loaded,sample,one,two
assert cumulative==inp['accepted_budget']
rec=torch.load(recovery_path,map_location='cpu');state=rec['state']
assert equal(state['model'],models[9].state_dict())
assert state['step']==inp['accepted_budget']
finite(state['model']);finite(state['optimizer']);finite(state['grads'])
m=models[9]
for n,p in m.named_parameters():
    p.requires_grad_(state['requires'][n]);p.grad=None if state['grads'][n] is None else state['grads'][n].clone()
for n,v in m.named_modules():v.training=state['modes'][n]
opt=torch.optim.AdamW(m.parameters(),**dict(cfg.train.optimizer.kwargs));opt.load_state_dict(state['optimizer'])
total9=inp['batches']['9']*100
scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:lr(step,total9));scheduler.load_state_dict(state['scheduler']);opt.load_state_dict(state['optimizer'])
assert equal(opt.state_dict(),state['optimizer']) and equal(scheduler.state_dict(),state['scheduler']) and scheduler.last_epoch==total9
for value in state['optimizer']['state'].values():assert int(value['step'])==total9
random.setstate(state['rng']['python']);np.random.set_state(state['rng']['numpy']);torch.set_rng_state(state['rng']['cpu'])
assert equal(random.getstate(),state['rng']['python']) and equal(np.random.get_state(),state['rng']['numpy']) and torch.equal(torch.get_rng_state(),state['rng']['cpu'])
assert len(state['rng']['cuda'])==1 and state['rng']['cuda'][0].dtype==torch.uint8
assert all(equal(p.grad,state['grads'][n]) and p.requires_grad==state['requires'][n] for n,p in m.named_parameters())
assert {n:v.training for n,v in m.named_modules()}==state['modes']
if method=='MLR':assert rec['task']==9 and rec['next_epoch']==100 and rec['stage_complete']
else:assert rec['history']==hist and rec['stage']==9 and rec['next_epoch']==100
from libero.libero.benchmark import get_benchmark
with contextlib.redirect_stdout(io.StringIO()):benchmark=get_benchmark(cfg.benchmark_name)(21)
initials={}
for task in range(6,10):
    info=benchmark.get_task(task);path=Path(cfg.init_states_folder)/info.problem_folder/info.init_states_file
    values=torch.load(path,map_location='cpu');initials[task]=dict(file_sha256=sha(path),hashes=[ah(v) for v in values[:30]])
records=J(R/'NORMALIZED_OUTCOMES.json' if method=='MLR' else R/'OUTCOMES.json')
expected=[(s,t,i) for s in range(6,10) for t in range(6,s+1) for i in range(20)]
assert [(x['stage'],x['task'],x['initial_index']) for x in records]==expected
for x in records:
    s,t,i=x['stage'],x['task'],x['initial_index']
    assert x['attempt']==1 and type(x['success']) is bool and x['original_task']==ORIGINAL_TASKS[t]
    assert x['cp_sha256']==checkpoints[str(s)]['sha256']
    assert x['initial_file_sha256']==initials[t]['file_sha256'] and x['initial_sha256']==initials[t]['hashes'][i]
    if method=='MLR':
        base=out/'evaluation';native=[json.loads(v) for v in (base/('trials_stage%d_task%d.jsonl'%(s,t))).read_text().splitlines() if v]
        assert x['native_record_sha256']==sha(base/('trials_stage%d_task%d.jsonl'%(s,t)))
        assert x['success']==native[i]['success'] and x['vector_group_steps']==native[i]['vector_group_steps']
        assert x['cell_rng_sha256']==sha(base/('rng_before_stage%d_task%d.pth'%(s,t)))
        assert 1<=x['vector_group_steps']<=600
    else:
        eps=np.random.RandomState(20261024+inp['seed']*1000+t*100+i).standard_normal((600,7)).astype(np.float32)
        assert x['noise_sha256']==ah(eps) and x['env_seed']==inp['seed']+i//10
        assert 1<=x['steps']<=600 and x['termination']==('success' if x['success'] else 'timeout600')
        assert x['success'] or x['steps']==600
cache_checks={};gate_count=0
if method=='DEBR':
    gates=[json.loads(v) for v in (R/'GATES.jsonl').read_text().splitlines() if v]
    assert len(gates)==inp['gate_blocks']
    cursor=inp['batches']['6']*100
    for stage in range(7,10):
        for epoch in range(100):
            for start in range(0,inp['batches'][str(stage)],10):
                g=gates[gate_count];n=min(10,inp['batches'][str(stage)]-start)
                assert (g['stage'],g['epoch'],g['batch'],g['steps'],g['start_step'],g['end_step'])==(stage,epoch,start,n,cursor,cursor+n)
                c,q=g['C'],g['Q'];checks=dict(bc_new=q['bc_new']<=c['bc_new'],exec_new=q['exec_new']<=c['exec_new'],retain_improved=q['retain_support']<c['retain_support'])
                assert g['checks']==checks and g['accepted']==('Q' if all(checks.values()) else 'C')
                cursor+=n;gate_count+=1
    assert cursor==inp['accepted_budget']
    for task in range(6,10):
        meta=J(R/('MEMORY_S_%d.json'%task));path=out/'memory'/('task%d.pth'%task)
        assert sha(path)==meta['cache_sha256']
        cache=torch.load(path,map_location='cpu');assert len(cache['rows'])==meta['count']<=128
        assert cache['source_sha256']==meta['source_sha256']==checkpoints[str(task)]['sha256']
        outcomes=J(R/('COLLECTION_OUTCOMES_S_%d.json'%task));assert len(outcomes)==10
        success={i for i,x in enumerate(outcomes) if x['success']}
        assert len(success)==meta['collection_successes']
        assert {int(x['trajectory']) for x in cache['rows']}==success
        assert J(R/('COLLECTION_STATE_PARITY_S_%d.json'%task))['parent_all_state_exact']
        for j,x in enumerate(outcomes):
            i=j+20;assert x['initial_index']==i and x['original_task']==ORIGINAL_TASKS[task] and x['attempt']==1
            assert x['initial_file_sha256']==initials[task]['file_sha256'] and x['initial_sha256']==initials[task]['hashes'][i]
            eps=np.random.RandomState(20271025+inp['seed']*1000+task*100+i).standard_normal((600,7)).astype(np.float32)
            assert x['noise_sha256']==ah(eps) and x['cp_sha256']==checkpoints[str(task)]['sha256']
        model=models[task];model.eval();max_error=0.;calls_before=model_calls
        for row,public in zip(cache['rows'],meta['rows']):
            assert row['source']=='online' and row['trajectory'] in success
            assert all(ah(row[key])==public[label] for key,label in [('feature','feature_sha256'),('mu','mu_sha256'),('sd','sd_sha256')])
        for length in range(1,9):
            rows=[x for x in cache['rows'] if len(x['feature'])==length]
            for start in range(0,len(rows),16):
                batch=rows[start:start+16];mu,sd,z=infer(model,torch.stack([x['feature'] for x in batch]));model_calls+=1
                for j,row in enumerate(batch):
                    mode=int(row['mode']);assert float(z[j,mode])>=float(z[j].max())-1e-4
                    assert torch.allclose(mu[j,mode],row['mu'],atol=1e-4,rtol=1e-4) and torch.allclose(sd[j,mode],row['sd'],atol=1e-4,rtol=1e-4)
                    max_error=max(max_error,float((mu[j,mode]-row['mu']).abs().max()),float((sd[j,mode]-row['sd']).abs().max()))
        cache_checks[str(task)]=dict(rows=len(cache['rows']),successful_trajectories=len(success),support=len(success)/10,source_checkpoint_exact=True,cpu_teacher_replay_within_1e_minus4=True,max_abs_error=max_error,forward_calls=model_calls-calls_before)
assert verify_case(R)==contract and not torch.cuda.is_initialized()
result=dict(status='PASS_REVERSE_CPU_CASE',case=case,method=method,seed=inp['seed'],suite=inp['suite'],order=ORIGINAL_TASKS,contract=contract,checkpoints=checkpoints,recovery=dict(sha256=sha(recovery_path),model_optimizer_scheduler_grads_modes_requires_rng_step=True,accepted_steps=inp['accepted_budget'],saved_cuda_rng_preserved_not_executed=True),
    epochs=400,accepted_updates=inp['accepted_budget'],physical_optimizer=calls['calls'],extra_autograd=calls['extra_autograd'],formal=200,training_collection=inp['training_collection'],initial_physical_bytes=True,noise_definition='native_saved_global_RNG' if method=='MLR' else 'native_per_trajectory_Normal',gates=gate_count,caches=cache_checks,model_forward_calls=model_calls,new_optimizer=0,new_environments=0,seconds=time.time()-tic)
write(R/'FINAL_CPU.json',result);print(json.dumps(result))
