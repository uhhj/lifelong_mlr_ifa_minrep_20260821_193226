import os
import signal
import subprocess
import sys
import time
from confirmation_experiment.common import ROOT,environment,write,read
from confirmation_experiment.recovery.common import AUDIT

def run():
    folder=AUDIT/'launcher_test';folder.mkdir(exist_ok=False)
    children=[]
    def start(role):
        cmd=[sys.executable,'-m','torch.distributed.run','--rdzv_backend=c10d',
             '--rdzv_endpoint=localhost:0','--rdzv_id=isolation-'+role,'--nproc_per_node=1',
             '--module','confirmation_experiment.recovery.launcher_worker']
        env=dict(environment(),ISOLATION_TEST_DIR=str(folder),ISOLATION_TEST_ROLE=role)
        with (folder/(role+'.log')).open('w') as f:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(p);return p
    def waitfile(name,p):
        deadline=time.monotonic()+150
        while not (folder/name).exists():
            assert p.poll() is None,('launcher exited early',name,p.returncode)
            if time.monotonic()>deadline:raise TimeoutError(name)
            time.sleep(.2)
        return read(folder/name)
    try:
        a=start('owner');va=waitfile('owner_ready.json',a)
        b=start('keeper');vb=waitfile('keeper_ready.json',b)
        assert va['master_port']!=vb['master_port']
        write(folder/'owner_release',{});assert a.wait(timeout=90)==0
        write(folder/'probe_after_owner_exit',{})
        result=waitfile('keeper_survived.json',b);assert b.wait(timeout=90)==0
        write(AUDIT/'LAUNCHER_ISOLATION.json',dict(result,owner=va,keeper=vb,owner_exited_first=True,
              actual_nccl_collectives_after_peer_exit=True,no_training_or_trials=True))
    finally:
        for p in children:
            if p.poll() is None:os.killpg(p.pid,signal.SIGTERM)

if __name__=='__main__':run()
