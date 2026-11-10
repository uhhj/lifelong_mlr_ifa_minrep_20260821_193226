"""Boundary handoff: keep active original evaluation untouched, cache future ones."""
from runtime_cached import *
import fcntl,signal,traceback,uuid
active={}
def launch(case,phase):
    prior.original.capacity();verify();assert not (E/(case+'_'+phase+'.exit')).exists()
    if phase=='evaluate':
        assert not (P/case/'random/evaluation/EVALUATION_COMPLETE.json').exists()
        assert not list((P/case/'random/evaluation').glob('trials*.jsonl'))
        cmd=[PY,'-B',str(E/'evaluate_cached.py'),case]
    else:
        assert case.startswith('GOAL_') and not (P/case/'train.exit').exists()
        assert not (P/case/'random/TRAINING_COMPLETE.json').exists()
        cmd=[PY,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+str(uuid.uuid4()),'--module','coreset_exp.train','random']
    with (E/(case+'_'+phase+'.log')).open('x') as f:p=subprocess.Popen(cmd,cwd=Q/'cases'/case/'source',env=env(case),stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    event('launch',case=case,phase=phase,pid=p.pid,command=cmd);return p
def reap(case,phase,p):
    rc=p.wait()
    with (E/(case+'_'+phase+'.exit')).open('x') as f:f.write(str(rc)+'\n')
    if phase=='train':
        with (P/case/'train.exit').open('x') as f:f.write(str(rc)+'\n')
    event('exit',case=case,phase=phase,pid=p.pid,code=rc);return rc
def check_prefix(g):
    for n,v in g['prefix_files'].items():
        raw=Path(n).read_bytes();assert len(raw)>=v['bytes'] and __import__('hashlib').sha256(raw[:v['bytes']]).hexdigest()==v['sha256'],n
    assert all(sha(Path(n))==h for n,h in g['immutable_files'].items())
def evaluate_suite(suite):
    for seed in [41,42,43]:
        case=suite+'_'+str(seed)
        if case in js(E/'HANDOFF_COMPLETE.json')['legacy_completed_cases']:
            assert js(P/case/'random/evaluation/EVALUATION_COMPLETE.json')['episodes']==200;continue
        assert js(P/case/'random/TRAINING_COMPLETE.json')['epochs']==400
        write(E/'STATUS.json',dict(status='EVALUATING_CACHED',case=case,observed=time.time()))
        p=launch(case,'evaluate');assert reap(case,'evaluate',p)==0
        assert js(P/case/'random/evaluation/EVALUATION_COMPLETE.json')['episodes']==200
    write(E/(suite+'_COMPLETE.json'),dict(epochs=1200,episodes=600,observed=time.time()))
def main():
    own=(E/'owner.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (E/'STARTED.json').exists();verify();g=js(E/'GUARD.json')
    assert same(g['old_controller'])['state'] in ['R','S','D'];assert same(g['evaluator'])['state'] not in ['T','Z','X']
    check_prefix(g);write(E/'STARTED.json',dict(identity=proc(os.getpid()),observed=time.time()))
    with (E/'watchdog.log').open('x') as f:w=subprocess.Popen([PY,'-B',str(E/'watchdog.py')],cwd=E,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    for _ in range(100):
        if (E/'WATCHDOG_READY.json').exists():break
        time.sleep(.1)
    assert (E/'WATCHDOG_READY.json').exists()
    same(g['old_controller']);os.kill(g['old_controller']['pid'],signal.SIGSTOP)
    for _ in range(100):
        if same(g['old_controller'])['state']=='T':break
        time.sleep(.02)
    assert same(g['old_controller'])['state']=='T'
    write(E/'STATUS.json',dict(status='ARMED_WAIT_ACTIVE_EVALUATOR_NATURAL_EXIT',case=g['case'],observed=time.time(),evaluation_signals=0))
    event('armed',old_controller=g['old_controller'],unchanged_evaluator=g['evaluator'],evaluation_signals=0)
    while True:
        ch=same(g['evaluator'])
        if ch['state']=='Z':break
        assert ch['state'] not in ['T','X'];time.sleep(2)
    rc=ch['exit_code']>>8 if ch['exit_code']&127==0 else -(ch['exit_code']&127)
    assert rc==0,('Active evaluation failed naturally; leave original evidence',rc)
    verify();check_prefix(g)
    legacy=[]
    for case in ['OBJECT_'+str(s) for s in [41,42,43]]:
        if (P/case/'random/evaluation/EVALUATION_COMPLETE.json').exists():
            assert js(P/case/'random/evaluation/EVALUATION_COMPLETE.json')['episodes']==200;legacy.append(case)
    assert g['case'] in legacy
    same(g['old_controller']);assert same(g['old_controller'])['state']=='T'
    os.kill(g['old_controller']['pid'],signal.SIGKILL)
    lock=(Q/'controller.lock').open('a+')
    for _ in range(100):
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:time.sleep(.1)
    else:raise RuntimeError('Root lock was not released')
    with (OLD/(g['case']+'_evaluate.exit')).open('x') as f:f.write(str(rc)+'\n')
    prior.event('exit',case=g['case'],phase='evaluate',pid=g['evaluator']['pid'],code=rc,source='natural kernel exit observed by authorized successor')
    preserved=g['immutable_files'].copy()
    for case in legacy:
        for p in (P/case/'random/evaluation').rglob('*'):
            if p.is_file():preserved[str(p)]=sha(p)
    write(E/'PRESERVED.json',dict(files=preserved))
    write(E/'HANDOFF_COMPLETE.json',dict(observed=time.time(),root_lock=True,adopted_natural_exit=ch,returncode=rc,legacy_completed_cases=legacy,evaluation_signals=0,old_controller_success_not_claimed=True))
    evaluate_suite('OBJECT')
    pending=['GOAL_'+str(s) for s in [41,42,43]];failed=False
    while pending or active:
        while pending and len(active)<2 and not failed:
            case=pending.pop(0);active[case]=launch(case,'train')
        for case,p in list(active.items()):
            if p.poll() is not None:
                if reap(case,'train',p)!=0:failed=True
                del active[case]
        write(E/'STATUS.json',dict(status='TRAINING' if not failed else 'FAILURE_WAITING_ACTIVE',active={c:p.pid for c,p in active.items()},pending=pending,observed=time.time()))
        if failed and not active:raise RuntimeError('Training failed; no retries')
        if active:time.sleep(5)
    evaluate_suite('GOAL');verify();assert not processes([P,Q/'cases'])
    write(E/'STATUS.json',dict(status='NATURAL_COMPLETE_PENDING_CPU_ACCEPTANCE',observed=time.time()))
    (E/'controller.exit').write_text('0\n')
if __name__=='__main__':
    try:main()
    except BaseException:
        write(E/'FAILURE.json',dict(observed=time.time(),error=traceback.format_exc()))
        for case,p in list(active.items()):reap(case,'train',p)
        (E/'controller.exit').write_text('1\n');raise
