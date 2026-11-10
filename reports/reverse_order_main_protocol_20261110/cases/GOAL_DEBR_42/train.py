import argparse,io,subprocess
from runtime import *
from updates import update
from memory import slots,retention
from support_memory import load_memory
from fork_tools import setup_probes,probe,choose,directional_diagnostic
import gate_admission
from initialization import initialization
import torch.distributed as dist

def scale_for(ds):
    if not ds.ids:return torch.ones(3,device='cuda')
    y=torch.stack([torch.as_tensor(ds.rows[i]['next_action']) for i in ds.ids]).float()
    return torch.stack([y[:,lo:hi].square().mean().sqrt().clamp_min(.1) for lo,hi in [(0,3),(3,6),(6,7)]]).cuda()

def admit(arm,stage,ds,a,memory,scale,info):
    ids=(ds.ids[:6]+[i for i,r in enumerate(ds.rows) if r['task']==stage][:4]) if ds.ids else list(range(10))
    batch=torch.utils.data.default_collate([ds[i] for i in ids]);initial=capture(a)
    first=update(a,batch,arm,memory,scale,info,stage);one=capture(a)
    stream=io.BytesIO();torch.save(dict(state=one,memory_sha256={str(t):sha(memory_path(arm,t)) for t in memory},sampling_cursor=a.step),stream);size=stream.tell();assert size<350000000
    stream.seek(0);loaded=torch.load(stream,map_location='cpu');assert equal(one,loaded['state'])
    second=update(a,batch,arm,memory,scale,info,stage);two=capture(a)
    restore(a,loaded['state']);third=update(a,batch,arm,memory,scale,info,stage);assert second==third and equal(two,capture(a))
    calls=3;grad=None
    if stage==6:
        restore(a,initial);a.caet_epoch_statistics=[];a.observe(batch,stage,0,info);assert equal(one,capture(a));calls+=1
    else:
        restore(a,one);measured=update(a,batch,arm,memory,scale,info,stage,measure=True);assert equal(two,capture(a));grad=measured['gradient_diagnostic'];calls+=1
        restore(a,one);update(a,batch,arm,memory,scale,info,stage,retain_weight=0.);off=capture(a)
        restore(a,one);update(a,batch,arm,memory,scale,info,stage,retain_weight=0.,legacy=True);assert equal(off,capture(a));calls+=2
    restore(a,initial);policy=a.policy.module;policy.eval();x=batch[0]['feature'].cuda().squeeze(1)
    with torch.no_grad():
        for length in range(1,9):
            h=policy.temporal_encode(x[:,:length])[:,-1];mu,sd,z=policy.policy_head.forward_fn(h);ix=torch.arange(len(mu),device='cuda');k=z.argmax(-1);actual=policy.policy_head(h,train=False)
            assert torch.equal(actual.base_dist.loc,mu[ix,k]) and torch.equal(actual.base_dist.scale,sd[ix,k])
    restore(a,initial);assert equal(initial,capture(a))
    if stage==6:
        # Versioned engineering check for the real retention path at lengths 1..8.
        policy.eval();saved=rng();checks=[];pars=[p for p in policy.parameters() if p.requires_grad]
        for length in range(1,9):
            xx=x[:1,:length].detach();q=policy.policy_head.forward_fn(policy.temporal_encode(xx)[:,-1]);kk=int(q[2][0].argmax())
            fake={6:dict(rows=[dict(feature=xx[0].cpu(),mu=q[0][0,kk].detach().cpu(),sd=q[1][0,kk].detach().cpu())],scales=torch.ones(3))}
            er,route,ct=retention(policy,fake,7,a.step)
            gs=torch.autograd.grad(er+.1*route,pars,allow_unused=True)
            assert all(g is None or torch.isfinite(g).all() for g in gs) and ct['actual']==4 and ct['slots']==4
            if length==8:
                old=params(policy,xx);assert all(torch.equal(u,v) for u,v in zip(q,old))
            checks.append(dict(length=length,finite_gradient=True,actual_slots=ct['actual']))
        assert equal(saved,rng());restore(a,initial);assert equal(initial,capture(a))
        write(R/('SHORT_HISTORY_ADMISSION_'+arm+'.json'),dict(status='PASS',lengths=checks,eight_frame_legacy_exact=True,all_state_restored=True,extra_autograd_calls=8,policy_calls=17,optimizer_calls=0,environments=0))
    assert loaded['sampling_cursor']==initial['step']+1
    write(R/('ADMISSION_%s_%d.json'%(arm,stage)),dict(status='PASS',arm=arm,stage=stage,optimizer_calls=calls,roundtrip_exact=True,next_step_exact=True,restore_initial_exact=True,baseline_or_legacy_update_exact=True,stateful_gradient_measurement_exact=stage>6,gradient_measurement=grad,retention_actual_slots_first_update=first['anchor_actual'],retention_empty_slots_first_update=first['anchor_empty'],memory_sha256=loaded['memory_sha256'],stateless_sampling_replay_exact=True,strict_keys=len(policy.state_dict()),parameters=sum(p.numel() for p in policy.parameters()),serialized_bytes=size,lengths=list(range(1,9)),environments=0,contract=verify()))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=list(ARMS));arm=ap.parse_args().arm
    verify();capacity();dest=OUTPUT[arm];dest.mkdir(parents=True,exist_ok=True);assert not (dest/'recovery.pth').exists(),'No blind resume'
    dist.init_process_group('nccl');torch.cuda.set_device(0)
    try:
        a=make(arm);initialization(a,arm);history=[];gradients=[];gates=[];physical=0
        for stage in range(6,10):
            write(R/'TRAIN_STATUS.json',dict(arm=arm,stage=stage,phase='prepare',time=time.time(),pid=os.getpid()))
            ds=Dataset(stage);loader,sampler=setup(a,ds);info=js(O/('ra_coreset/ifa_task%d.json'%stage))['info'];a.policy.train()
            memory=load_memory(arm,stage);scale=scale_for(ds)
            binding={str(t):dict(cache_sha256=sha(memory_path(arm,t)),source_sha256=memory[t]['source_sha256'],count=len(memory[t]['rows']),support=memory[t]['support'],support_trajectories=memory[t]['support_trajectories']) for t in memory}
            write(R/('TRAIN_INPUT_%s_%d.json'%(arm,stage)),dict(status='PASS_FROZEN_INPUT',manifest_sha256=sha(O/('ra_coreset/dataset_manifests/task%d.json'%stage)),scale=scale.tolist(),memory=binding,sampling='cyclic task slots; stateless independent RandomState per step/slot; empty tasks contribute zero'))
            pr=setup_probes(ds,memory) if stage>6 else None
            if stage==6:
                admit(arm,stage,ds,a,memory,scale,info)
            else:
                write(R/('PROBES_%d.json'%stage),pr['metadata'])
                gate_admission.STAGE=stage
                own=gate_admission.admission(a,ds,memory,scale,info,pr,'G')
                assert own['optimizer_calls']==29 and own['extra_autograd_calls']==11
                write(R/('ADMISSION_%s_%d.json'%(arm,stage)),dict(arm=arm,stage=stage,method='support_gate_full_S',contract=verify(),**own))
            for ep in range(100):
                capacity();sampler.set_epoch(ep);order=list(iter(sampler));a.policy.train();tic=time.time();lr=a.optimizer.param_groups[0]['lr'];measured=False
                tot={k:0. for k in ['loss','bc','ifa','expert_loss','expert_route','retain_loss','retain_route','valid','invalid','old_eligible','anchor_slots','anchor_actual','anchor_empty','anchor_forwards']}
                iterator=iter(loader);bi=0;retained_steps=0
                while True:
                    block=[]
                    for _ in range(10 if stage>6 else 1):
                        try:block.append(next(iterator))
                        except StopIteration:break
                    if not block:break
                    if stage>6 and ep in [0,9,49,99] and not measured:
                        dg=directional_diagnostic(a,block[0],memory,pr,'Q')
                        if dg is not None:
                            measured=True;gradients.append(dict(stage=stage,epoch=ep,batch=bi,steps=a.step,**dg));write(R/('GRADIENTS_'+arm+'.json'),gradients)
                    if stage>6:
                        base=capture(a)
                        crows=[update(a,b,'C',memory,scale,info,stage,retain_weight=0.) for b in block]
                        cstate=capture(a);cv=probe(a,pr,memory)
                        restore(a,base)
                        qrows=[update(a,b,'Q',memory,scale,info,stage) for b in block]
                        qstate=capture(a);qv=probe(a,pr,memory)
                        decision,checks=choose(cv,qv);selected=cstate if decision=='C' else qstate
                        restore(a,selected);assert equal(selected,capture(a))
                        rows=crows if decision=='C' else qrows
                        gate=dict(stage=stage,epoch=ep,batch=bi,steps=len(block),start_step=base['step'],end_step=a.step,C=cv,Q=qv,accepted=decision,checks=checks)
                        gates.append(gate)
                        with (R/'GATES.jsonl').open('a') as f:f.write(json.dumps(gate)+'\n')
                        if decision=='Q':retained_steps+=len(block)
                        physical+=2*len(block)
                        del base,cstate,qstate,selected
                    else:
                        rows=[update(a,block[0],arm,memory,scale,info,stage)]
                        physical+=1
                    for st in rows:
                        for k in tot:tot[k]+=st[k]
                    bi+=len(block)
                assert bi==len(loader)
                assert tot['valid']==VALID[stage] and tot['invalid']==POP[stage]-VALID[stage] and tot['old_eligible']==OLD[stage]
                assert tot['anchor_slots']==4*retained_steps and tot['anchor_slots']==tot['anchor_actual']+tot['anchor_empty']
                row=dict(stage=stage,epoch=ep,steps=a.step,batches=len(loader),seconds=time.time()-tic,lr_first=lr,lr_after=a.optimizer.param_groups[0]['lr'],sampler_sha256=hashlib.sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest(),totals=tot,physical_optimizer_calls=physical,gate_blocks=len(gates),gate_Q_accepts=sum(g['accepted']=='Q' for g in gates),training_probe=probe(a,pr,memory) if pr else None)
                history.append(row);state=capture(a)
                save(dest/'recovery.pth',dict(state=state,state_dict=state['model'],stage=stage,next_epoch=ep+1,steps=a.step,retention_sampling_cursor=a.step,contract=verify(),memory_bindings=binding,manifest_sha256=sha(O/('ra_coreset/dataset_manifests/task%d.json'%stage)),history=history,physical_optimizer_calls=physical,probe_sha256=sha(R/('PROBES_%d.json'%stage)) if pr else None,gate_blocks=len(gates)))
                write(R/('EPOCHS_'+arm+'.json'),history);write(R/'TRAIN_STATUS.json',dict(arm=arm,stage=stage,phase='train',epoch_complete=ep+1,steps=a.step,physical_optimizer_calls=physical,time=time.time(),pid=os.getpid()));print(json.dumps(dict(arm=arm,**row)),flush=True)
            target=model_path(arm,stage);assert not target.exists()
            if stage==9:os.link(dest/'recovery.pth',target)
            else:save(target,dict(state_dict=cpu(a.policy.state_dict()),stage=stage,epoch=99,steps=a.step,contract=verify(),memory_bindings=binding))
            write(R/('STAGE_%s_%d.json'%(arm,stage)),dict(status='PASS_TRAINING_PENDING_FINAL',model=str(target),sha256=sha(target),steps=a.step,epochs=100,stage=stage,memory_bindings=binding))
            before=capture(a);subenv=dict(os.environ)
            for k in ['RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE']:subenv.pop(k,None)
            write(R/'TRAIN_STATUS.json',dict(arm=arm,stage=stage,phase='training_collection',time=time.time(),pid=os.getpid()))
            with (R/('collect_%s_%d.log'%(arm,stage))).open('x') as log:
                run=subprocess.run([sys.executable,'-B',str(R/'collect.py'),arm,str(stage)],cwd=R,env=subenv,stdout=log,stderr=subprocess.STDOUT)
            (R/('collect_%s_%d.exit'%(arm,stage))).write_text(str(run.returncode));assert run.returncode==0
            assert equal(before,capture(a)),'Collection changed parent training state/RNG'
            write(R/('COLLECTION_STATE_PARITY_%s_%d.json'%(arm,stage)),dict(status='PASS',parent_all_state_exact=True,memory_sha256=sha(memory_path(arm,stage)),environments=10))
            del before
        assert a.step==90700 and len(history)==400 and physical==161700 and len(gates)==7200
        write(R/('TRAIN_'+arm+'.json'),dict(status='PASS_EXECUTION_PENDING_FINAL',epochs=400,steps=a.step,physical_optimizer_calls=physical,gate_blocks=len(gates),Q_accepts=sum(g['accepted']=='Q' for g in gates),summed_epoch_seconds=sum(x['seconds'] for x in history),gradient_autograd_calls=sum(x['extra_autograd_calls'] for x in gradients),training_collection_episodes=40,checkpoints={str(t):sha(model_path(arm,t)) for t in range(6,10)}))
    finally:dist.destroy_process_group()

if __name__=='__main__':main()
