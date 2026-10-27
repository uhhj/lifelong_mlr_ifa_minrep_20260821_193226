"""Finite dispatcher. No retries, no automatic scientific expansion."""
import os,sys,json,subprocess,time,fcntl
from pathlib import Path
R=Path(__file__).resolve().parent
def write(p,x):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(x,indent=2));os.replace(q,p)
def main():
    assert not (R/'STARTED.json').exists(),'No blind controller restart'
    lock=open('/data/mlr_ifa/SUPPORT_GATE_FORKS_20261027.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(R/'STARTED.json',dict(pid=os.getpid(),time=time.time()));cases=json.loads((R/'CASES.json').read_text())
    for name,c in cases.items():
        for arm in c['arms']:
            write(R/'STATUS.json',dict(phase='train',case=name,arm=arm,time=time.time()))
            with (R/name/('train_'+arm+'.log')).open('x') as log:
                rc=subprocess.call([sys.executable,'-B','-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(R/name/'train.py'),arm],cwd=R/name,stdout=log,stderr=subprocess.STDOUT)
            (R/name/('train_'+arm+'.exit')).write_text(str(rc));assert rc==0,(name,arm,rc)
    for name,c in cases.items():
        write(R/'STATUS.json',dict(phase='evaluate',case=name,time=time.time()))
        with (R/name/'evaluate.log').open('x') as log:rc=subprocess.call([sys.executable,'-B',str(R/name/'evaluate.py')],cwd=R/name,stdout=log,stderr=subprocess.STDOUT)
        (R/name/'evaluate.exit').write_text(str(rc));assert rc==0,(name,rc)
    write(R/'STATUS.json',dict(phase='complete',time=time.time()))
if __name__=='__main__':
    rc=1
    try:main();rc=0
    finally:(R/'controller.exit').write_text(str(rc))
