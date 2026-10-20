"""Finite phase ledger. Recovery requires a separate preserved-attempt invocation."""
import fcntl
import json
import os
import subprocess
import sys
import time
from .common import O,verify,write,sha

def main():
    c=verify()
    lock=(O/"pipeline.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (O/"pipeline.exit").exists()
    write(O/"pipeline.pid.json",dict(pid=os.getpid(),unix=time.time()))
    distributed=lambda module:[sys.executable,"-m","torch.distributed.run","--standalone",
        "--nproc_per_node=1","-m",module]
    phases=([("pretrain",distributed("l50_exp.pretrain")),
             ("strict_load",[sys.executable,"-m","l50_exp.pretrain_acceptance"])]
            if c["pretraining"] else
            [("export",[sys.executable,"-m","l50_exp.export"]),
             ("train",distributed("l50_exp.train")),
             ("evaluate",[sys.executable,"-m","l50_exp.evaluate"]),
             ("audit",[sys.executable,"-m","l50_exp.audit"])])
    rc=1
    try:
        for name,command in phases:
            assert not (O/(name+".exit")).exists()
            write(O/"RUN_STATUS.json",dict(status="RUNNING",phase=name,unix=time.time()))
            with (O/"commands.jsonl").open("a") as f:
                f.write(json.dumps(dict(event="start",phase=name,argv=command,
                    contract_sha256=sha(O/"CONTRACT.json"),unix=time.time()))+"\n")
            with (O/(name+".log")).open("x") as f:
                p=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                write(O/(name+".pid.json"),dict(pid=p.pid,unix=time.time()))
                value=p.wait()
            (O/(name+".exit")).write_text(str(value)+"\n")
            with (O/"commands.jsonl").open("a") as f:
                f.write(json.dumps(dict(event="exit",phase=name,exit=value,unix=time.time()))+"\n")
            if value:
                rc=value
                write(O/"RUN_STATUS.json",dict(status="BLOCKED_ENGINEERING",phase=name,
                    exit=value,unix=time.time(),failure_preserved=True))
                return rc
        rc=0
        return rc
    finally:(O/"pipeline.exit").write_text(str(rc)+"\n")
if __name__=="__main__":sys.exit(main())
