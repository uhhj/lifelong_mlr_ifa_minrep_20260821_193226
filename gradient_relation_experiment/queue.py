"""One fixed serial seed queue, technical failure stop, no score-based decisions."""
import subprocess
import sys
import time
import traceback
from gradient_relation_experiment.run import verify_code,OUT,ROOT,read,write


def run():
    verify_code();assert not (OUT/'QUEUE_STARTED.json').exists()
    write(OUT/'QUEUE_STARTED.json',{'pid':__import__('os').getpid(),'time':time.time(),'seeds':[45,46,47],
        'carrier':'new_controlled_3090_from_official','points':120,'branches':['11','10','01','00'],'primary_horizon':20})
    try:
        for seed in (45,46,47):
            verify_code()
            with (OUT/f'seed{seed}.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-m','gradient_relation_experiment.run','--seed',str(seed)],
                    cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                write(OUT/'QUEUE_STATUS.json',{'status':'RUNNING','seed':seed,'child_pid':child.pid,'time':time.time()})
                assert child.wait()==0,f'seed{seed} failed; preserve all results, no automatic rerun'
        write(OUT/'QUEUE_STATUS.json',{'status':'SUMMARIZING','time':time.time()})
        subprocess.run([sys.executable,'-m','gradient_relation_experiment.summarize'],cwd=ROOT,check=True)
        write(OUT/'QUEUE_STATUS.json',{'status':'COMPLETE','time':time.time(),'points_completed':120})
    except Exception:
        write(OUT/'QUEUE_FAILED.json',{'time':time.time(),'traceback':traceback.format_exc()})
        write(OUT/'QUEUE_STATUS.json',{'status':'FAILED_INTEGRITY','time':time.time()});raise


if __name__=='__main__':run()
