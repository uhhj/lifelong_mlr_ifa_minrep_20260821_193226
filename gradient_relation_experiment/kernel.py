"""Same-forward factorial contribution removal with native finite AdamW updates."""
import contextlib
import copy
import io
import math
from collections import Counter
from torch.utils.data._utils.collate import default_collate
from cb_ifa_experiment.gpu_runtime import *
from optimizer_signal_diagnostic.core import monitor_loss


def module(algo):return algo.policy.module if hasattr(algo.policy,'module') else algo.policy


def capture(algo):
    return {'model':cpu_state(algo.policy.state_dict()),'optimizer':cpu_state(algo.optimizer.state_dict()),
        'scheduler':copy.deepcopy(algo.scheduler.state_dict()),'rng':cpu_state(rng_state()),
        'optimizer_step':algo._optimizer_step_count,'scheduler_step':algo._scheduler_step_count,
        'training_modes':{n:m.training for n,m in algo.policy.named_modules()}}


def restore(algo,s):
    algo.policy.load_state_dict(s['model'],strict=True)
    algo.optimizer.load_state_dict(copy.deepcopy(s['optimizer']))
    algo.scheduler.load_state_dict(copy.deepcopy(s['scheduler']))
    algo._optimizer_step_count=s['optimizer_step'];algo._scheduler_step_count=s['scheduler_step']
    for n,m in algo.policy.named_modules():m.training=s['training_modes'][n]
    restore_rng(s['rng'])


def dot(x,y):
    return sum(float((a.detach().double()*b.detach().to(a.device).double()).sum()) for a,b in zip(x,y) if a is not None and b is not None)


def norm(x):return math.sqrt(max(0.,dot(x,x)))


def cosine(x,y):
    nx=norm(x);ny=norm(y)
    return dot(x,y)/nx/ny if nx>1e-12 and ny>1e-12 else None


def batch(ds,indices):return default_collate([ds[i] for i in indices])


def delta(algo,s):
    return tuple(p.detach().cpu()-s['model'][n] for n,p in algo.policy.named_parameters() if p.requires_grad)


def monitor(algo,banks):
    state=rng_state();modes={n:m.training for n,m in algo.policy.named_modules()}
    algo.policy.eval();result={}
    try:
        with torch.no_grad():
            for (bank,task),data in banks.items():
                values=[]
                for b in data:values.append(float(monitor_loss(module(algo),b)))
                result[f'{bank}/{task}']=sum(values)/len(values)
        assert all(math.isfinite(x) for x in result.values())
    finally:
        for n,m in algo.policy.named_modules():m.training=modes[n]
        restore_rng(state)
    return result


def first_gradients(algo,data,current):
    algo.optimizer.zero_grad();model=module(algo);ps=tuple(p for p in algo.policy.parameters() if p.requires_grad)
    assert algo.loss_scale==1. and model.policy_head.loss_coef==1.
    data[0]=algo.map_tensor_to_device(data[0]);ids=torch.tensor([int(x) for x in data[1]],device=ps[0].device)
    losses,_=model.compute_loss(data[0],None,None,reduction='none')
    assert losses.ndim==1 and bool(torch.isfinite(losses).all())
    counts=Counter(ids.tolist());N=len(ids);means={};contrib={}
    def grad(loss):return tuple(None if x is None else x.detach().clone() for x in torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True))
    for t in sorted(counts):means[t]=grad(losses[ids==t].mean())
    old=sorted(t for t in counts if t<current)
    for j in old:
        contrib[j]={}
        for arm,mask in [('10',ids!=j),('01',ids!=current),('00',(ids!=current)&(ids!=j))]:
            contrib[j][arm]=grad(losses[mask].sum()/N) if bool(mask.any()) else None
    losses.mean().backward()
    mixed=tuple(None if p.grad is None else p.grad.detach().clone() for p in ps)
    # Empty objective still participates in AdamW for all normally active parameters.
    for j in old:
        for arm in contrib[j]:
            if contrib[j][arm] is None:contrib[j][arm]=tuple(None if x is None else torch.zeros_like(x) for x in mixed)
        contrib[j]['11']=mixed
    if current not in means:means[current]=tuple(None if x is None else torch.zeros_like(x) for x in mixed)
    weighted=tuple(sum(means[t][i]*(counts[t]/N) for t in counts if means[t][i] is not None) if g is not None else None for i,g in enumerate(mixed))
    err=norm(tuple(None if g is None else g-w for g,w in zip(mixed,weighted)))
    assert err<=max(1e-6,2e-5*norm(mixed)),('Gradient decomposition mismatch',err,norm(mixed))
    return dict(gradients=contrib,means=means,mixed=mixed,counts=dict(counts),loss=float(losses.mean().detach()),
        after_rng=cpu_state(rng_state()),buffers={n:cpu_state(b) for n,b in algo.policy.named_buffers()},
        decomposition_relative_error=err/max(norm(mixed),1e-30))


def apply_first(algo,grads,forward):
    algo.optimizer.zero_grad()
    for p,g in zip((p for p in algo.policy.parameters() if p.requires_grad),grads):p.grad=None if g is None else g.clone()
    with torch.no_grad():
        for n,b in algo.policy.named_buffers():b.copy_(forward['buffers'][n])
    restore_rng(forward['after_rng'])
    pre=norm(tuple(p.grad for p in algo.policy.parameters() if p.requires_grad))
    torch.nn.utils.clip_grad_norm_(algo.policy.parameters(),algo.cfg.train.grad_clip)
    post=norm(tuple(p.grad for p in algo.policy.parameters() if p.requires_grad))
    algo.optimizer_step_and_maybe_schedule()
    return {'preclip_norm':pre,'postclip_norm':post}


def native(algo,b,current,epoch):
    with contextlib.redirect_stdout(io.StringIO()):return algo.observe(b,current,epoch,None)


def ensure_next_epoch_rng(algo,ds,plan,previous_step,next_step):
    bpe=plan['batches_per_epoch']
    if previous_step//bpe==next_step//bpe:return
    algo.epoch_schedule_step(previous_step//bpe);algo.policy.train()
    # Same DataLoader base_seed draw; sampler has its own generator and consumes no global RNG.
    torch.empty((),dtype=torch.int64).random_()


def compare_states(a,b):
    fields=['model','optimizer','scheduler','rng','optimizer_step','scheduler_step','training_modes']
    return {k:digest(a[k])==digest(b[k]) for k in fields}


def moments_distance(a,b):
    out={}
    for name in ('exp_avg','exp_avg_sq'):
        out[name]=math.sqrt(sum(float((v[name].double()-b['optimizer']['state'][k][name].double()).square().sum())
              for k,v in a['optimizer']['state'].items() if name in v))
    return out


def run_point(algo,ds,plan,point,banks,emit):
    before=capture(algo);start_rng=digest(before['rng']);initial=monitor(algo,banks);repeat=monitor(algo,banks)
    tolerance={key:max(2*abs(initial[key]-repeat[key]),32*torch.finfo(torch.float32).eps*max(1.,abs(value))) for key,value in initial.items()}
    restore(algo,before);forward=first_gradients(algo,batch(ds,point['future20_record_indices'][0]),point['task'])
    new=forward['means'][point['task']];scores={};stability={}
    for j,grads in forward['gradients'].items():
        old=forward['means'][j]
        scores[j]={'cosine':cosine(new,old),'new_dot_old':dot(new,old),'new_norm':norm(new),'old_norm':norm(old),
            'mixed_dot_old':dot(forward['mixed'],old),'mixed_norm':norm(forward['mixed']),
            'n_new':forward['counts'].get(point['task'],0),'n_old_j':forward['counts'][j],
            'n_batch':sum(forward['counts'].values()),'lr':algo.optimizer.param_groups[0]['lr'],
            'gradient_valid':norm(new)>1e-12 and norm(old)>1e-12,'decomposition_relative_error':forward['decomposition_relative_error']}
        cs=[]
        for ref in point['stability_references'][str(j)]:
            restore(algo,before);indices=list(point['future20_record_indices'][0]);it=iter(ref)
            for i,t in enumerate(point['first_task_ids']):
                if t==j:indices[i]=next(it)
            data=batch(ds,indices);data[0]=algo.map_tensor_to_device(data[0]);ids=torch.tensor([int(t) for t in data[1]],device='cuda:0')
            losses,_=module(algo).compute_loss(data[0],None,None,reduction='none')
            ref_grad=torch.autograd.grad(losses[ids==j].mean(),tuple(p for p in algo.policy.parameters() if p.requires_grad),allow_unused=True)
            cs.append(cosine(new,ref_grad));del ref_grad,losses
        stability[j]=cs
        emit('scores',dict(point=point['point'],seed=point['seed'],task=point['task'],phase=point['phase'],epoch=point['epoch'],
            stage_step=point['stage_step'],old_task=j,**scores[j],stability_cosines=cs,
            score_recorded_before_future_labels=True,start_rng_sha256=start_rng))
    if not forward['gradients']:
        emit('unobserved',dict(point=point['point'],reason='no old task present',**{k:point[k] for k in ['seed','task','phase']}))
    common11={};witnesses={}
    for j,grads in forward['gradients'].items():
        labels={};states11={};deltas1={};models1={};geometry=[]
        # Native duplicate is a genuine unchanged MLR update with the same full start.
        restore(algo,before);native(algo,batch(ds,point['future20_record_indices'][0]),point['task'],point['epoch'])
        native_state=capture(algo);native_monitor=monitor(algo,banks)
        for arm in ('11','10','01','00'):
            restore(algo,before);clip=apply_first(algo,grads[arm],forward)
            for h in range(1,21):
                if h>1:
                    a=point['stage_step']+h-1;ensure_next_epoch_rng(algo,ds,plan,a-1,a)
                    native(algo,batch(ds,point['future20_record_indices'][h-1]),point['task'],a//plan['batches_per_epoch'])
                if h not in (1,5,20):continue
                state=capture(algo);observed=monitor(algo,banks);labels[arm,h]=observed
                d=delta(algo,before);row={'point':point['point'],'seed':point['seed'],'task':point['task'],
                    'phase':point['phase'],'old_task':j,'arm':arm,'horizon':h,'displacement_norm':norm(d),
                    'optimizer_step':state['optimizer_step'],'scheduler_step':state['scheduler_step'],
                    'rng_sha256':digest(state['rng']),**clip}
                if arm=='11':
                    states11[h]=state
                    if h==1:
                        eq=compare_states(native_state,state);assert all(eq.values()),eq
                        assert observed==native_monitor
                        for key in tolerance:tolerance[key]=max(tolerance[key],2*abs(observed[key]-native_monitor[key]))
                    sig=digest(state['model'])
                    if h in common11:assert common11[h]==sig,'11 depends on selected old task'
                    common11[h]=sig
                    witnesses[h]={k:digest(state[k]) for k in ('model','optimizer','scheduler','rng','optimizer_step','scheduler_step','training_modes')}
                row['displacement_from_11']=math.sqrt(sum(float((state['model'][n].double()-states11[h]['model'][n].double()).square().sum()) for n in state['model']))
                row.update(moments_distance(state,states11[h]));geometry.append(row)
                if h==1:deltas1[arm]=d;models1[arm]=state['model']
        optimizer_scores={'old_dot_d11':dot(forward['means'][j],deltas1['11']),
            'old_dot_d11_minus_d01':dot(forward['means'][j],tuple(a-b for a,b in zip(deltas1['11'],deltas1['01']))),
            'new_dot_d11_minus_d10':dot(new,tuple(a-b for a,b in zip(deltas1['11'],deltas1['10'])))}
        for row in geometry:emit('geometry',row)
        for bank in ('A','B'):
            oldkey=f'{bank}/{j}';newkey=f'{bank}/{point["task"]}'
            for h in (1,5,20):
                emit('labels',{'point':point['point'],'seed':point['seed'],'task':point['task'],'phase':point['phase'],
                    'old_task':j,'bank':bank,'horizon':h,'E_new_to_old':labels['11',h][oldkey]-labels['01',h][oldkey],
                    'E_old_to_new':labels['11',h][newkey]-labels['10',h][newkey],
                    'H_MLR':labels['11',h][oldkey]-initial[oldkey],
                    'old_interaction':labels['11',h][oldkey]-labels['10',h][oldkey]-labels['01',h][oldkey]+labels['00',h][oldkey],
                    'new_interaction':labels['11',h][newkey]-labels['10',h][newkey]-labels['01',h][newkey]+labels['00',h][newkey],
                    'old_tolerance':2*tolerance[oldkey],'new_tolerance':2*tolerance[newkey],**optimizer_scores})
                for arm in ('11','10','01','00'):
                    for key,value in labels[arm,h].items():
                        if key.startswith(bank+'/'):emit('losses',{'point':point['point'],'old_task':j,'arm':arm,'horizon':h,'bank_task':key,'before':initial[key],'after':value,'delta':value-initial[key]})
        target=norm(deltas1['11'])
        for arm in ('11','10','01','00'):
            restore(algo,before);dn=norm(deltas1[arm]);valid=dn>1e-15 or target==0
            if valid:
                with torch.no_grad():
                    for (n,p),d in zip(((n,p) for n,p in algo.policy.named_parameters() if p.requires_grad),deltas1[arm]):
                        p.copy_(before['model'][n].to(p.device)+d.to(p.device)*(target/dn if dn else 0.))
                equal=monitor(algo,banks)
            else:equal={}
            emit('equal_norm',{'point':point['point'],'old_task':j,'arm':arm,'target_norm':target,'original_norm':dn,
                'valid':valid,'single_step_artificial_only':True,'losses':equal,'before':initial})
    restore(algo,before)
    assert all(compare_states(before,capture(algo)).values()),'Diagnostic contaminated carrier'
    emit('checks',{'point':point['point'],'start_model_sha256':digest(before['model']),'start_optimizer_sha256':digest(before['optimizer']),
        'carrier_restored_exact':True,'native_11_parity':True,'old_tasks':list(forward['gradients']),'tolerance':tolerance})
    return witnesses
