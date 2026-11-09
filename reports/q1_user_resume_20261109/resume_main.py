def main():
    ap=argparse.ArgumentParser();ap.add_argument('arm',choices=list(ARMS));arm=ap.parse_args().arm
    verify();capacity();dest=OUTPUT[arm];dest.mkdir(parents=True,exist_ok=True);assert sha(dest/'recovery.pth')==binding['checkpoint_sha256'],'Interrupted checkpoint changed'
    dist.init_process_group('nccl');torch.cuda.set_device(0)
    try:
        a=make(arm);cp=torch.load(dest/'recovery.pth',map_location='cpu')
        assert cp['history']==js(R/'EPOCHS_S.json') and len(cp['history'])==binding['complete_epochs']
        history=cp['history'];gradients=js(R/'GRADIENTS_S.json');interrupted_gates=[json.loads(line) for line in (R/'GATES.jsonl').read_text().splitlines()]
        assert len(interrupted_gates)==binding['gate_rows_total'];gates=interrupted_gates[:cp['gate_blocks']]
        physical=cp['physical_optimizer_calls'];CALLS.update(production=physical,admission=91)
        for stage in [9]:
            write(R/'TRAIN_STATUS.json',dict(arm=arm,stage=stage,phase='prepare',time=time.time(),pid=os.getpid()))
            ds=Dataset(stage);loader,sampler=setup(a,ds);info=js(O/('ra_coreset/ifa_task%d.json'%stage))['info'];a.policy.train()
            memory=load_memory(arm,stage);scale=scale_for(ds)
            binding_input={str(t):dict(cache_sha256=sha(memory_path(arm,t)),source_sha256=memory[t]['source_sha256'],count=len(memory[t]['rows']),support=memory[t]['support'],effective_support=(1. if memory[t]['rows'] else 0.),support_trajectories=memory[t]['support_trajectories']) for t in memory}
            assert binding_input==cp['memory_bindings']
            assert scale.tolist()==js(R/'TRAIN_INPUT_S_9.json')['scale']
            pr=setup_probes(ds,memory);assert json.loads(json.dumps(pr['metadata']))==js(R/'PROBES_9.json')
            assert cp['probe_sha256']==sha(R/'PROBES_9.json')
            restore(a,cp['state']);assert equal(cp['state'],capture(a))
            write(ENG/(case+'_PRODUCTION_RESTORE.json'),dict(status='PASS_FULL_STATE_RESTORED',time=time.time(),checkpoint_sha256=binding['checkpoint_sha256'],step=a.step))
            set_count_mode('production')
            for ep in range(cp['next_epoch'],100):
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
                        if ep==cp['next_epoch'] and bi<10*len(binding['partial_gates']):
                            assert gate==binding['partial_gates'][bi//10],'Original partial gate differs'
                            write(ENG/(case+'_PREFIX_REPLAY.json'),dict(status='EXACT_RECONSTRUCTION',blocks=bi//10+1,total=len(binding['partial_gates']),time=time.time()))
                        else:
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
                assert bi==len(loader) and CALLS['production']==physical
                assert tot['valid']==VALID[stage] and tot['invalid']==POP[stage]-VALID[stage] and tot['old_eligible']==OLD[stage]
                assert tot['anchor_slots']==4*retained_steps and tot['anchor_slots']==tot['anchor_actual']+tot['anchor_empty']
                row=dict(stage=stage,epoch=ep,steps=a.step,batches=len(loader),seconds=time.time()-tic,lr_first=lr,lr_after=a.optimizer.param_groups[0]['lr'],sampler_sha256=hashlib.sha256(np.array(order,dtype=np.int64).tobytes()).hexdigest(),totals=tot,physical_optimizer_calls=physical,physical_calls_observed=dict(CALLS),gate_blocks=len(gates),gate_Q_accepts=sum(g['accepted']=='Q' for g in gates),training_probe=probe(a,pr,memory) if pr else None)
                history.append(row);state=capture(a)
                save(dest/'recovery.pth',dict(state=state,state_dict=state['model'],stage=stage,next_epoch=ep+1,steps=a.step,retention_sampling_cursor=a.step,contract=verify(),memory_bindings=binding_input,manifest_sha256=sha(O/('ra_coreset/dataset_manifests/task%d.json'%stage)),history=history,physical_optimizer_calls=physical,probe_sha256=sha(R/('PROBES_%d.json'%stage)) if pr else None,gate_blocks=len(gates)))
                write(R/('EPOCHS_'+arm+'.json'),history);write(R/'TRAIN_STATUS.json',dict(arm=arm,stage=stage,phase='train',epoch_complete=ep+1,steps=a.step,physical_optimizer_calls=physical,time=time.time(),pid=os.getpid()));print(json.dumps(dict(arm=arm,**row)),flush=True)
            target=model_path(arm,stage);assert not target.exists()
            if stage==9:os.link(dest/'recovery.pth',target)
            else:save(target,dict(state_dict=cpu(a.policy.state_dict()),stage=stage,epoch=99,steps=a.step,contract=verify(),memory_bindings=binding_input))
            if stage==6:
                baseline=torch.load(O/'ra_coreset/models/mlrifa_6_model_ep99.pth',map_location='cpu')['state_dict'];actual=cpu(a.policy.state_dict())
                sr=js(R/'S_REFERENCE.json')['stage6_checkpoint'];assert sha(sr['path'])==sr['sha256']
                baseline_s=torch.load(sr['path'],map_location='cpu')['state_dict'];assert actual.keys()==baseline_s.keys()
                assert all(actual[k].shape==baseline_s[k].shape and actual[k].dtype==baseline_s[k].dtype and torch.isfinite(actual[k]).all() for k in actual)
                task6_exact=equal(actual,baseline_s)
                write(R/'TASK6_HARDWARE_COMPARISON.json',dict(all_tensors_exact=task6_exact,baseline_cp_sha256=sr['sha256'],actual_cp_sha256=sha(target),different_tensors={k:float((actual[k].double()-baseline_s[k].double()).abs().max()) for k in actual if not torch.equal(actual[k],baseline_s[k])}))
                write(R/('TASK6_PARITY_'+arm+'.json'),dict(historical_ALIGN_equal=equal(actual,baseline),historical_equality_gate=False,actual_sha256=sha(target),historical_ALIGN_sha256=sha(O/'ra_coreset/models/mlrifa_6_model_ep99.pth'),same_seed_new_arm_comparator='AUTHORIZED_CROSS_GPU_COMPARISON_RECORDED',accepted_S_all_tensors_exact=task6_exact,accepted_S_cp_sha256=sr['sha256']))
            write(R/('STAGE_%s_%d.json'%(arm,stage)),dict(status='PASS_TRAINING_PENDING_FINAL',model=str(target),sha256=sha(target),steps=a.step,epochs=100,stage=stage,memory_bindings=binding_input))
            before=capture(a);subenv=dict(os.environ)
            for k in ['RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','MASTER_ADDR','MASTER_PORT','GROUP_RANK','ROLE_RANK','ROLE_WORLD_SIZE']:subenv.pop(k,None)
            write(R/'TRAIN_STATUS.json',dict(arm=arm,stage=stage,phase='training_collection',time=time.time(),pid=os.getpid()))
            with (R/('collect_%s_%d.log'%(arm,stage))).open('x') as log:
                run=subprocess.run([sys.executable,'-B',str(R.parent/'environment_locked.py'),R.name,'collect',arm,str(stage)],cwd=R,env=subenv,stdout=log,stderr=subprocess.STDOUT)
            (R/('collect_%s_%d.exit'%(arm,stage))).write_text(str(run.returncode));assert run.returncode==0
            assert equal(before,capture(a)),'Collection changed parent training state/RNG'
            write(R/('COLLECTION_STATE_PARITY_%s_%d.json'%(arm,stage)),dict(status='PASS',parent_all_state_exact=True,memory_sha256=sha(memory_path(arm,stage)),environments=10))
            del before
        assert CALLS=={'production':195300,'admission':91}
        assert a.step==105500 and len(history)==400 and physical==195300 and len(gates)==9100
        write(R/('TRAIN_'+arm+'.json'),dict(status='PASS_EXECUTION_PENDING_FINAL',epochs=400,steps=a.step,physical_optimizer_calls=physical,physical_calls_observed=dict(CALLS),gate_blocks=len(gates),Q_accepts=sum(g['accepted']=='Q' for g in gates),summed_epoch_seconds=sum(x['seconds'] for x in history),gradient_autograd_calls=sum(x['extra_autograd_calls'] for x in gradients),training_collection_episodes=40,checkpoints={str(t):sha(model_path(arm,t)) for t in range(6,10)}))
    finally:dist.destroy_process_group()
