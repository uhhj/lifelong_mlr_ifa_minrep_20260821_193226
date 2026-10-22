"""Finite stage transactions. No blind restart of collection or evaluation."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from coreset_exp.common import O, verify, write, sha
from .variant import VARIANT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-training", action="store_true")
    args = ap.parse_args()
    verify()
    lock = (O/"pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (O/"pipeline.exit").exists(), "Preserve original exit; external recovery required"
    write(O/"pipeline.pid.json", dict(pid=os.getpid(), start=time.time()))
    phases = []
    for task in range(6,10):
        if task!=6:
            phases.append(("labels%d"%task,[sys.executable,"-m","caet_v3_exp.labels",str(task)]))
        command=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc_per_node=1",
                 "-m","caet_v3_exp.train","ra_coreset","--through-task",str(task)]
        if task!=6 or args.resume_training:
            command.append("--resume")
        phases.append(("train%d"%task,command))
        phases.append(("retain%d"%task,[sys.executable,"-m","caet_v3_exp.labels",str(task),"--retain"]))
    phases.extend([("evaluate",[sys.executable,"-m","caet_v3_exp.evaluate","ra_coreset"]),
                   ("audit",[sys.executable,"-m","caet_v3_exp.audit"])])
    result = 1
    try:
        for name, command in phases:
            assert not (O/(name+".exit")).exists(), "No repeated phase"
            write(O/"RUN_STATUS.json", dict(status="RUNNING", phase=name,
                                           unix=time.time(), pid=os.getpid()))
            with (O/"commands.jsonl").open("a") as ledger:
                ledger.write(json.dumps(dict(event="start", phase=name, argv=command,
                    unix=time.time(), contract_sha256=sha(O/"CONTRACT.json")))+"\n")
            with (O/(name+".log")).open("x") as log:
                p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                write(O/(name+".pid.json"), dict(pid=p.pid, argv=command, unix=time.time()))
                rc = p.wait()
            (O/(name+".exit")).write_text(str(rc)+"\n")
            with (O/"commands.jsonl").open("a") as ledger:
                ledger.write(json.dumps(dict(event="exit", phase=name, code=rc,
                                             unix=time.time()))+"\n")
            if rc:
                write(O/"RUN_STATUS.json", dict(status="BLOCKED_ENGINEERING",
                    phase=name, exit=rc, failed_evidence_preserved=True))
                result = rc
                return rc
        result = 0
        return 0
    finally:
        (O/"pipeline.exit").write_text(str(result)+"\n")


if __name__ == "__main__":
    sys.exit(main())
