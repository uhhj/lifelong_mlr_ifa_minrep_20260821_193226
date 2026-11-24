"""Run both offline jobs sequentially in the experiment directory with restartable status."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def run(work,out):
    lock=(out/"driver.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    def status(**values):
        tmp=out/"driver_status.tmp"
        tmp.write_text(json.dumps(dict(time=time.time(),**values)))
        tmp.replace(out/"driver_status.json")
    for variant in ("shared","high"):
        if (out/variant/"epoch_50.pt").is_file():
            continue
        status(stage="training",variant=variant)
        env=dict(os.environ,HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",OMP_NUM_THREADS="2",OPENBLAS_NUM_THREADS="1")
        cmd=[sys.executable,"-Bu",str(out/"train_contrast.py"),str(work),str(out),variant]
        with (out/(variant+".log")).open("a") as log:
            result=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status(stage="failed",variant=variant,exit_code=result.returncode)
            return result.returncode
    status(stage="training_complete")
    return 0


if __name__=="__main__":
    sys.exit(run(*map(Path,sys.argv[1:3])))
