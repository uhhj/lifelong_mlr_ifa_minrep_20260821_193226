"""Serial finite conditional scheduler. Missing admitted implementation waits; never guesses."""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from branch_gate import next_action
from queue_verifier import verify_worker, sha, live_workers


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+".tmp")
    with temp.open("w") as f:
        json.dump(obj,f,indent=2,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("root")
    args=ap.parse_args();root=Path(args.root).resolve()
    lock=(root/"controller.lock").open("a")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/"controller.exit").exists(), "Preserve exited controller; explicit recovery only"
    write(root/"controller.pid.json",dict(pid=os.getpid(),unix=time.time()))
    evidence={}
    rc=1
    try:
        while True:
            decision=next_action(evidence)
            write(root/"DECISION.json",dict(decision,unix=time.time()))
            if decision["action"]=="ALL_ACCEPTED":
                rc=0;return
            assert decision["action"]!="BLOCKED",decision
            key="%s_%s_%s"%(decision["method"],decision["suite"],decision["seed"])
            registry=json.loads((root/"REGISTRY.json").read_text())
            entry=registry.get(key)
            if entry is None or not entry.get("implementation_reviewed"):
                write(root/"RUN_STATUS.json",dict(status="WAITING_IMPLEMENTATION",
                    key=key,unix=time.time(),existing_runs_untouched=True))
                time.sleep(60);continue
            receipt=root/"receipts"/(key+".json")
            output=Path(entry["output"])
            if not (output/"supervisor.exit").exists():
                if live_workers(output):
                    write(root/"RUN_STATUS.json",dict(status="RUNNING",key=key,unix=time.time()))
                    time.sleep(60);continue
                # Any previous start without a live process or exit requires explicit
                # recovery, rather than relaunching a partially completed global-RNG run.
                start=root/"starts"/(key+".json")
                assert not start.exists(), "Started worker vanished; investigate retained evidence"
                assert not output.exists(), "Existing output without live worker/exit; no blind restart"
                launcher=Path(entry["launcher"])
                assert sha(launcher)==entry["launcher_sha256"]
                for p,h in entry["deployment_hashes"].items():
                    assert sha(p)==h,p
                write(start,dict(entry=entry,registry_sha256=sha(root/"REGISTRY.json"),unix=time.time()))
                with (root/(key+".launcher.log")).open("x") as log:
                    p=subprocess.Popen(["bash",str(launcher)],stdout=log,stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL)
                    launcher_rc=p.wait()
                write(root/(key+".launcher_exit.json"),dict(exit=launcher_rc,unix=time.time()))
                assert launcher_rc==0,"Launcher failed; preserve all phase attempts"
            assert decision["method"]!="RA_IFA", "LIBERO50 independent adapter must be registered before execution"
            result=verify_worker(entry)
            write(receipt,dict(result,method=decision["method"],suite=decision["suite"],
                               seed=decision["seed"],unix=time.time(),registry_sha256=sha(root/"REGISTRY.json")))
            evidence.setdefault(decision["method"],{}).setdefault(decision["suite"],{})[
                str(decision["seed"])]=result
            write(root/"VERIFIED_EVIDENCE.json",evidence)
    except Exception as exc:
        write(root/"BLOCKED.json",dict(unix=time.time(),error=repr(exc),
                                      traceback=traceback.format_exc(),evidence_preserved=True))
        raise
    finally:
        (root/"controller.exit").write_text(str(rc)+"\n")


if __name__=="__main__":
    main()
