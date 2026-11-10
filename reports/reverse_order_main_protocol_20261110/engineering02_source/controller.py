"""Finite reverse-order queue; never resumes or retries a scientific attempt."""
from reverse_common import *
import fcntl,subprocess,traceback,uuid
def launch(case):
    r=Q/'cases'/case
    assert not (r/'TRAIN_DISPATCH.json').exists() and not (r/'train.exit').exists()
    identity=uuid.uuid4().hex
    cmd=[PY,'-B','-m','torch.distributed.run','--nnodes=1','--nproc_per_node=1','--rdzv_backend=c10d','--rdzv_endpoint=localhost:0','--rdzv_id='+identity,str(Q/'run_case.py'),case,'train']
    with (r/'train.log').open('x') as log:
        p=subprocess.Popen(cmd,cwd=r,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    write(r/'TRAIN_DISPATCH.json',dict(time=time.time(),pid=p.pid,cmd=cmd,rdzv_id=identity));event('train_dispatch',case=case,pid=p.pid)
    return p
def train_suite(suite):
    pending=[c for c in read(Q/'SPEC.json')['cases'] if c.startswith(suite+'_')]
    active={};failed=[]
    while pending or active:
        while pending and len(active)<2 and not failed:
            verify();capacity();case=pending.pop(0);active[case]=launch(case)
        for case,p in list(active.items()):
            code=p.poll()
            if code is None:continue
            with (Q/'cases'/case/'train.exit').open('x') as f:f.write(str(code)+'\n')
            event('train_exit',case=case,pid=p.pid,returncode=code);del active[case]
            if code:failed.append(dict(case=case,returncode=code))
        write(Q/'STATUS.json',dict(status='PARTIAL_TRAINING' if not failed else 'FAILED_DRAINING',suite=suite,active={c:p.pid for c,p in active.items()},pending=pending,failures=failed,time=time.time()))
        if failed and not active:raise RuntimeError(('Reverse training failed; no automatic retry',failed))
        if active or pending:time.sleep(10)
def evaluate(case):
    r=Q/'cases'/case
    assert (r/'train.exit').read_text().strip()=='0'
    assert not (r/'EVAL_DISPATCH.json').exists()
    path=Path('/data/mlr_ifa/locks/reverse_main_protocol_environment.lock');path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        event('environment_acquired',case=case,purpose='formal',pid=os.getpid())
        cmd=[PY,'-B',str(Q/'run_case.py'),case,'evaluate']
        with (r/'evaluate.log').open('x') as log:
            p=subprocess.Popen(cmd,cwd=r,env=environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write(r/'EVAL_DISPATCH.json',dict(time=time.time(),pid=p.pid,cmd=cmd));event('evaluate_dispatch',case=case,pid=p.pid)
        write(Q/'STATUS.json',dict(status='PARTIAL_EVALUATING',case=case,pid=p.pid,time=time.time()))
        code=p.wait()
        with (r/'evaluate.exit').open('x') as f:f.write(str(code)+'\n')
        event('environment_released',case=case,purpose='formal',pid=os.getpid(),success=code==0)
        event('evaluate_exit',case=case,pid=p.pid,returncode=code)
        assert code==0,('Reverse formal attempt failed; preserve actual attempts; no retry',case,code)
def main():
    with (Q/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (Q/'controller.exit').exists()
        verify(full=True);capacity();event('controller_started',pid=os.getpid())
        for suite in ['OBJECT','GOAL']:
            train_suite(suite)
            for case in read(Q/'SPEC.json')['cases']:
                if case.startswith(suite+'_'):verify();capacity();evaluate(case)
            write(Q/(suite+'_COMPLETE.json'),dict(status='PASS_EXECUTION_PENDING_FINAL',formal=1200,training_runs=6,time=time.time()))
        event('controller_complete',pid=os.getpid());write(Q/'STATUS.json',dict(status='EXECUTION_COMPLETE_PENDING_CPU_ACCEPTANCE',time=time.time()))
if __name__=='__main__':
    code=1
    try:main();code=0
    except BaseException:write(Q/'FAILURE.json',dict(time=time.time(),traceback=traceback.format_exc()));raise
    finally:
        with (Q/'controller.exit').open('x') as f:f.write(str(code)+'\n')
