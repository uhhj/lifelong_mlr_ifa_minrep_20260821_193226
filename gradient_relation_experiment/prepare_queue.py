"""Durable preparation and real-GPU preflight; no scientific label queue."""
import subprocess
import sys
import time
import traceback
from cb_ifa_experiment.common import write
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/gradient_relation_v1'


def run():
    try:
        for name in ('prepare','preflight'):
            with (OUT/(name+'.log')).open('x') as log:
                child=subprocess.Popen([sys.executable,'-m','gradient_relation_experiment.'+name],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                write(OUT/'PREPARATION_STATUS.json',{'phase':name,'child_pid':child.pid,'time':time.time()})
                assert child.wait()==0,name+' failed; no automatic retry'
        write(OUT/'PREPARATION_STATUS.json',{'phase':'READY','time':time.time()})
    except Exception:
        write(OUT/'PREPARATION_QUEUE_FAILED.json',{'time':time.time(),'traceback':traceback.format_exc()});raise


if __name__=='__main__':run()
