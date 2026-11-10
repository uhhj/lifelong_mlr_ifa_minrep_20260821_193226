"""Own real-state admission for the manuscript MLR source, restored before training."""
from pathlib import Path
import copy,hashlib,io,json,pickle,os
import numpy as np
import torch
import instrumentation as counts
from coreset_exp.common import rng,restore_rng,write,sha,jsonable
from coreset_exp.model import angular_mean
R=Path(os.environ['REVERSE_CASE_ROOT'])
INPUT=json.loads((R/'INPUT.json').read_text())
def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu(v) for v in x)
    return copy.deepcopy(x)
def capture(a):
    return dict(model=cpu(a.policy.state_dict()),optimizer=cpu(a.optimizer.state_dict()),scheduler=cpu(a.scheduler.state_dict()),grads={n:cpu(p.grad) if p.grad is not None else None for n,p in a.policy.named_parameters()},modes={n:m.training for n,m in a.policy.named_modules()},requires={n:p.requires_grad for n,p in a.policy.named_parameters()},rng=cpu(rng()),step=a.step)
def restore(a,x):
    a.policy.load_state_dict(x['model'],strict=True);a.optimizer.load_state_dict(x['optimizer']);a.scheduler.load_state_dict(x['scheduler']);a.step=x['step']
    for n,p in a.policy.named_parameters():
        p.grad=None if x['grads'][n] is None else x['grads'][n].to(p.device)
        p.requires_grad_(x['requires'][n])
    for n,m in a.policy.named_modules():m.training=x['modes'][n]
    restore_rng(x['rng'])
def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def record_initial(a,benchmark):
    from order import verify_built_mapping
    verify_built_mapping(benchmark,a.cfg,R)
    before=rng()
    digest={k:hashlib.sha256(pickle.dumps(v) if k in ('python','numpy') else b''.join(x.cpu().numpy().tobytes() for x in (v if isinstance(v,list) else [v]))).hexdigest() for k,v in before.items()}
    tensors={k:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for k,v in a.policy.state_dict().items()}
    expected=INPUT['initial_state']
    assert tensors==expected['tensors'] and all(digest[k]==expected['rng'][k] for k in digest if k!='cuda')
    assert equal(before,rng())
    write(R/'INITIAL_BINDING.json',dict(status='PASS_OWN_INITIAL_TENSORS_AND_NONCUDA_RNG',tensors=tensors,rng=digest,historical_cuda_rng_equal=digest['cuda']==expected['rng']['cuda'],pretrain_sha256=sha(a.cfg.pretrain),order=INPUT['order']))
def direct(a,b,stage,info):
    b[0]=a.map_tensor_to_device(b[0]);a.optimizer.zero_grad()
    bc,last=a.policy.module.compute_loss(b[0],None,None)
    mask=torch.tensor([int(x)==stage for x in b[1]],device=last.device)
    old=[a.refs[j if k==stage else k] for j,k in info['selected_pairs']]
    ifa=angular_mean(last[mask],a.refs[stage],old,a.cfg.gamma) if mask.any() else last.sum()*0
    loss=bc+.1*ifa;(a.loss_scale*loss).backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in a.policy.parameters())
    if a.cfg.train.grad_clip is not None:torch.nn.utils.clip_grad_norm_(a.policy.parameters(),a.cfg.train.grad_clip)
    a.optimizer.step();a.scheduler.step();a.step+=1
    return float(loss.detach())
def admit(a,ds,stage,info):
    assert jsonable(info)==json.loads((R/('EXPECTED_IFA_%d.json'%stage)).read_text())['info']
    from libero.lifelong.algos.mlr import parse_path
    expected=json.loads((R/'manifests'/('task%d.json'%stage)).read_text())['rows']
    assert len(expected)==len(ds.file_paths)
    for path,row in zip(ds.file_paths,expected):
        task=int(parse_path(path));name=Path(path).name
        if task<stage:name=name.split('__',1)[1]
        assert task==row['task'] and name==row['sample'] and sha(path)==row['sha256']
    current=next(i for i,p in enumerate(ds.file_paths) if int(parse_path(p))==stage)
    old=next((i for i,p in enumerate(ds.file_paths) if int(parse_path(p))<stage),current)
    indices=sorted(set([0,current,old,len(ds)-1]))
    batch=torch.utils.data.default_collate([ds[i] for i in indices])
    initial=capture(a);before=counts.CALLS.copy()
    with counts.admission():
        one_value=a.observe(copy.deepcopy(batch),stage,0,info);one=capture(a)
        restore(a,initial);assert direct(a,copy.deepcopy(batch),stage,info)==one_value and equal(one,capture(a))
        two_value=a.observe(copy.deepcopy(batch),stage,0,info);two=capture(a)
        stream=io.BytesIO();torch.save(one,stream);stream.seek(0);loaded=torch.load(stream,map_location='cpu');assert equal(one,loaded)
        restore(a,loaded);assert a.observe(copy.deepcopy(batch),stage,0,info)==two_value and equal(two,capture(a))
        restore(a,loaded);assert direct(a,copy.deepcopy(batch),stage,info)==two_value and equal(two,capture(a))
        restore(a,initial);assert equal(initial,capture(a))
    assert counts.CALLS['admission']-before['admission']==5 and counts.CALLS['production']==before['production']
    write(R/('ADMISSION_MLR_%d.json'%stage),dict(status='PASS_OWN_MAIN_MLR_GPU_ADMISSION',stage=stage,original_task=INPUT['order'][stage],optimizer_calls=5,production_calls_added=0,environments=0,direct_first_update_exact=True,direct_second_update_exact=True,serialized_next_step_exact=True,all_state_restored=True,random_state_restored=True,actual_population_order_task_and_hash_exact=True,manifest_sha256=INPUT['manifests'][str(stage)]['sha256'],calls=counts.CALLS.copy()))
