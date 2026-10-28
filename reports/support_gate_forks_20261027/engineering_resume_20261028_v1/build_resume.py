"""Generate a bounded recovery wrapper without editing frozen science sources."""
from pathlib import Path
import ast, difflib, hashlib, json

E = Path(__file__).resolve().parent
R = E.parent
raw = (R / 'OBJECT_41/train.py').read_text(encoding='utf8')
assert hashlib.sha256((R / 'OBJECT_41/train.py').read_bytes()).hexdigest() == '74b85059abba7b54b5b11b081cc9967bc604d10d55de4590859e1a2806196249'
prefix = "import sys\nfrom pathlib import Path\nsys.path.insert(0,str(Path(__file__).resolve().parents[1]/'OBJECT_41'))\n"
start = raw.index('    verify();capacity();dest=')
end = raw.index('        for ep in range(100):', start)
replacement = '''    assert arm=='S' and CASE=='OBJECT_41' and STAGE==9
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
'''
out = prefix + raw[:start] + replacement + raw[end:].replace('for ep in range(100):', 'for ep in range(7,100):', 1)
ast.parse(out)
# The complete scientific epoch loop, accounting and final save are unmodified.
assert out[out.index('            capacity();sampler.set_epoch(ep)'):] == raw[raw.index('            capacity();sampler.set_epoch(ep)'):]
(E/'resume_train_S.py').write_text(out,encoding='utf8')
(E/'resume_train_S.diff').write_text(''.join(difflib.unified_diff(raw.splitlines(True),out.splitlines(True),fromfile='frozen_OBJECT_41_train.py',tofile='resume_train_S.py')),encoding='utf8')
(E/'BUILD_AUDIT.json').write_text(json.dumps(dict(status='PASS_STATIC_ONLY_GPU_PENDING',original_train_sha256=hashlib.sha256(raw.encode()).hexdigest(),resume_sha256=hashlib.sha256(out.encode()).hexdigest(),epoch_loop_exact=True,start_epoch=7,end_epoch_exclusive=100,extra_isolated_calls_planned=31,extra_autograd_planned=11,unlogged_interrupted_cost='UNKNOWN',scientific_parameters_changed=False),indent=2)+'\n',encoding='utf8')
print('PASS_STATIC_ONLY_GPU_PENDING')
