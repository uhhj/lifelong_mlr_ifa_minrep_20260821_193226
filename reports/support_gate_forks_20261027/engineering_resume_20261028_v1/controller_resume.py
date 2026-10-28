"""One bounded continuation: S epochs 8-100, virgin Q/G/U, then original evaluation."""
import os, sys, json, subprocess, time, fcntl, hashlib
from pathlib import Path
E=Path(__file__).resolve().parent
R=E.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text())
def write(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def verify():
    c=js(R/'CONTRACT.json');assert all(sha(p)==h for p,h in c['files'].items())
    e=js(E/'CONTRACT.json');assert all(sha(E/p)==h for p,h in e['files'].items())
def main():
    lock=open('/data/mlr_ifa/SUPPORT_GATE_FORKS_20261027.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (E/'STARTED.json').exists(),'Do not repeat a recovery attempt'
    verify();cases=js(R/'CASES.json')
    assert not (R/'controller.exit').exists() and not (R/'OBJECT_41/train_S.exit').exists()
    completed={}
    for name,c in cases.items():
        for arm in c['arms']:
            if name=='OBJECT_41' and arm!='C':continue
            assert (R/name/('train_'+arm+'.exit')).read_text().strip()=='0'
            assert len(js(R/name/('EPOCHS_'+arm+'.json')))==100
            completed[name+'/'+arm]=sha(R/name/('EPOCHS_'+arm+'.json'))
        assert not (R/name/'evaluate.log').exists() and not (R/name/'OUTCOMES.json').exists()
    assert len(completed)==13 and len(js(R/'OBJECT_41/EPOCHS_S.json'))==7
    for arm in ['Q','G','U']:
        assert not (R/'OBJECT_41'/('train_'+arm+'.log')).exists()
        assert not (Path('/data/mlr_ifa/experiments/SUPPORT_GATE_FORKS_20261027_v1/OBJECT_41')/arm).exists()
    write(E/'STARTED.json',dict(pid=os.getpid(),time=time.time(),completed_epoch_file_sha256=completed,interrupted_cost='UNKNOWN',original_controller_did_not_exit_normally=True))
    for arm in ['S','Q','G','U']:
        verify();write(R/'STATUS.json',dict(phase='train',case='OBJECT_41',arm=arm,time=time.time(),engineering_controller=str(E),resumed=(arm=='S')))
        source=E/'resume_train_S.py' if arm=='S' else R/'OBJECT_41/train.py'
        logfile=E/'train_S_resume.log' if arm=='S' else R/'OBJECT_41'/('train_'+arm+'.log')
        with logfile.open('x') as log:
            rc=subprocess.call([sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(source),arm],cwd=R/'OBJECT_41',stdout=log,stderr=subprocess.STDOUT)
        (E/('train_'+arm+'.exit')).write_text(str(rc))
        assert rc==0,(arm,rc)
        if arm=='S':assert js(E/'RECOVERY_ADMISSION.json')['status']=='PASS_GPU_FULL_STATE_RESTORE_AND_NEXT_STEP_REPLAY'
        assert len(js(R/'OBJECT_41'/('EPOCHS_'+arm+'.json')))==100
        # Logical arm completion marker; original interrupted log remains untouched.
        marker=R/'OBJECT_41'/('train_'+arm+'.exit');assert not marker.exists();marker.write_text('0')
        write(E/('COMPLETION_'+arm+'.json'),dict(exit=0,logical_arm=arm,log=str(logfile),resumed=(arm=='S'),time=time.time()))
    for key,h in completed.items():
        case,arm=key.split('/');assert sha(R/case/('EPOCHS_'+arm+'.json'))==h
    for name,c in cases.items():
        verify();write(R/'STATUS.json',dict(phase='evaluate',case=name,time=time.time(),engineering_controller=str(E)))
        with (R/name/'evaluate.log').open('x') as log:
            rc=subprocess.call([sys.executable,'-B',str(R/name/'evaluate.py')],cwd=R/name,stdout=log,stderr=subprocess.STDOUT)
        (R/name/'evaluate.exit').write_text(str(rc));assert rc==0,(name,rc)
    verify();write(R/'STATUS.json',dict(phase='complete',time=time.time(),engineering_controller=str(E),final_acceptance_pending=True))
if __name__=='__main__':
    rc=1
    try:main();rc=0
    finally:(E/'controller.exit').write_text(str(rc))
