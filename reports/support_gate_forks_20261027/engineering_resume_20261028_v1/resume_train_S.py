import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'OBJECT_41'))
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
    return update(a,b,arm,m,scale,info,STAGE,retain_weight=0. if arm=='C' else .1)

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
            assert torch.allclose(rx,.1*sx,atol=1e-7,rtol=1e-6) and torch.allclose(rt,.1*st,atol=1e-7,rtol=1e-6)
            grads=torch.autograd.grad(rx+.1*rt,[p for p in a.policy.parameters() if p.requires_grad],allow_unused=True)
            assert all(v is None or torch.isfinite(v).all() for v in grads)
            if length==8:assert all(torch.equal(u,v) for u,v in zip(qq,params(a.policy.module,x)))
            short.append(dict(length=length,q_point_one_preserved=True,finite_gradient=True))
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
    return dict(status='PASS_OWN_GPU_ADMISSION',optimizer_calls=calls,roundtrip_next_step_exact=True,q_one_original_S_exact=True,lambda_zero_original_C_exact=True,ten_step_C_replay_exact=True,accepted_and_rejected_full_state_restore=True,measurement_state_exact=True,probe_all_state_exact=True,synthetic_gate_accept_reject=True,short_history=short,extra_autograd_calls=11,diagnostic=dg,probe=pv,environments=0)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=ARMS);arm=ap.parse_args().arm
    assert arm=='S' and CASE=='OBJECT_41' and STAGE==9
    verify();capacity();dest=OUTPUT[arm]
    eng=R.parent/'engineering_resume_20261028_v1'
    assert not (eng/'RESUME_STARTED.json').exists(),'Recovery already attempted; inspect, never retry blindly'
    write(eng/'RESUME_STARTED.json',dict(time=time.time(),pid=os.getpid(),isolated_calls_planned=31,extra_autograd_planned=11))
    recovery=dest/'recovery.pth'
    expected='df2c9101552a50f0478b36b1df942166c614b598a6d11af0d595fb4c9da7bb97'
    assert sha(recovery)==expected
    cp=torch.load(recovery,map_location='cpu')
    assert cp['history']==js(R/'EPOCHS_S.json') and cp['next_epoch']==7 and cp['state']['step']==71834
    assert cp['physical_optimizer_calls']==2534 and cp['contract']==verify()
    assert cp['origin_sha256']==sha(BINDING['checkpoint']) and cp['probe_sha256']==sha(R/'PROBES.json')
    assert not model_path(arm,STAGE).exists() and not (R/'train_S.exit').exists()
    # An immutable private copy retains the original interrupted checkpoint.
    import shutil
    archived=dest/'interrupted_20261028_epoch7.pth'
    assert not archived.exists()
    shutil.copyfile(recovery,archived);assert sha(archived)==expected
    write(eng/'INTERRUPTED_CHECKPOINT.json',dict(sha256=expected,bytes=archived.stat().st_size,next_epoch=7,partial_epoch_unlogged_cost='UNKNOWN',no_completed_epoch_repeated=True))
    dist.init_process_group('nccl');torch.cuda.set_device(0)
    try:
        a=make(arm);ds=Dataset(STAGE);loader,sampler=setup(a,ds)
        memory=load_memory(arm,STAGE);scale=scale_for(ds);info=js(O/('ra_coreset/ifa_task%d.json'%STAGE))['info'];pr=setup_probes(ds,memory)
        pp=R/'PROBES.json';assert js(pp)==json.loads(json.dumps(pr['metadata']))
        restore(a,cp['state']);assert equal(cp['state'],capture(a))
        ad=admission(a,ds,memory,scale,info,pr,arm)
        assert ad['optimizer_calls']==29 and equal(cp['state'],capture(a))
        write(eng/'GPU_ADMISSION.json',ad)
        sampler.set_epoch(7);it=iter(loader);batch=next(it);base=capture(a)
        write(eng/'REPLAY_ATTEMPT.json',dict(status='STARTED',isolated_optimizer_calls_planned=2,environments=0))
        step(a,batch,arm,memory,scale,info);once=capture(a)
        restore(a,base);step(a,batch,arm,memory,scale,info);assert equal(once,capture(a))
        restore(a,cp['state']);assert equal(cp['state'],capture(a))
        write(eng/'RECOVERY_ADMISSION.json',dict(status='PASS_GPU_FULL_STATE_RESTORE_AND_NEXT_STEP_REPLAY',checkpoint_sha256=expected,next_epoch=7,step=a.step,optimizer_calls=31,extra_autograd_calls=11,environments=0,full_state_exact=True,next_step_replay_exact=True,checkpoint_restored_after_admission=True,partial_epoch_unlogged_cost='UNKNOWN',device=torch.cuda.get_device_name(0)))
        del base,once,batch,it
        history=cp['history'];gradients=js(R/'GRADIENTS_S.json');gates=[];physical=cp['physical_optimizer_calls']
        assert all(x['epoch']<7 for x in gradients)
        del cp
        for ep in range(7,100):
            capacity();sampler.set_epoch(ep);order=list(iter(sampler));a.policy.train();tic=time.time();measured=False;rows=[];iterator=iter(loader);bi=0
            while True:
                block=[]
                for _ in range(10 if arm=='G' else 1):
                    try:block.append(next(iterator))
                    except StopIteration:break
                if not block:break
                if ep in [0,9,49,99] and not measured:
                    dg=directional_diagnostic(a,block[0],memory,pr,arm)
                    if dg is not None:measured=True;gradients.append(dict(epoch=ep,batch=bi,step=a.step,**dg));write(R/('GRADIENTS_'+arm+'.json'),gradients)
                if arm=='G':
                    base=capture(a);crows=[step(a,b,'C',memory,scale,info) for b in block];cstate=capture(a);cv=probe(a,pr,memory)
                    restore(a,base);qrows=[step(a,b,'Q',memory,scale,info) for b in block];qstate=capture(a);qv=probe(a,pr,memory)
                    decision,checks=choose(cv,qv);selected=cstate if decision=='C' else qstate;restore(a,selected)
                    assert equal(selected,capture(a));rows.extend(crows if decision=='C' else qrows)
                    gates.append(dict(epoch=ep,batch=bi,steps=len(block),start_step=base['step'],end_step=a.step,C=cv,Q=qv,accepted=decision,checks=checks))
                    with (R/'GATES.jsonl').open('a') as f:f.write(json.dumps(gates[-1])+'\n')
                    del base,cstate,qstate,selected;physical+=2*len(block)
                else:rows.append(step(a,block[0],arm,memory,scale,info));physical+=1
                bi+=len(block)
            assert bi==len(loader) and sum(x['valid'] for x in rows)==VALID[STAGE] and sum(x['old_eligible'] for x in rows)==OLD[STAGE]
            row=dict(epoch=ep,stage=STAGE,steps=a.step,batches=bi,physical_optimizer_calls=physical,seconds=time.time()-tic,sampler_sha256=hashlib.sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest(),totals={k:sum(x[k] for x in rows) for k in ['loss','bc','ifa','expert_loss','expert_route','retain_loss','retain_route','valid','invalid','old_eligible','anchor_slots','anchor_actual','anchor_empty','anchor_forwards']},training_probe=probe(a,pr,memory),gate_Q_accepts=sum(x['accepted']=='Q' for x in gates),gate_blocks=len(gates))
            history.append(row);state=capture(a);save(dest/'recovery.pth',dict(state=state,state_dict=state['model'],stage=STAGE,next_epoch=ep+1,history=history,contract=verify(),arm=arm,origin_sha256=sha(BINDING['checkpoint']),probe_sha256=sha(pp),physical_optimizer_calls=physical))
            write(R/('EPOCHS_'+arm+'.json'),history);write(R/'TRAIN_STATUS.json',dict(arm=arm,stage=STAGE,epoch_complete=ep+1,steps=a.step,physical_optimizer_calls=physical,time=time.time(),pid=os.getpid()));print(json.dumps(dict(arm=arm,epoch=ep,seconds=row['seconds'],steps=a.step)),flush=True)
        os.link(dest/'recovery.pth',model_path(arm,STAGE));assert a.step-BINDING['checkpoint_steps']==100*len(loader)
        write(R/('TRAIN_'+arm+'.json'),dict(status='PASS_EXECUTION_PENDING_FINAL',epochs=100,accepted_updates=100*len(loader),physical_optimizer_calls=physical,checkpoint=str(model_path(arm,STAGE)),sha256=sha(model_path(arm,STAGE)),epoch_seconds=sum(x['seconds'] for x in history),gate_blocks=len(gates),Q_accepts=sum(x['accepted']=='Q' for x in gates),formal_environments=0))
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
