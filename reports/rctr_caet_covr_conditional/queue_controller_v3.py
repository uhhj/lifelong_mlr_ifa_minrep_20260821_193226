"""Versioned serial scheduler with independent LIBERO50 verification.
Original method gates and running-worker sources unchanged.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from branch_gate_v3 import next_action
from queue_verifier import verify_worker, sha, live_workers
from libero50_verifier import verify_libero50


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
    assert not (root/"controller_v3.exit").exists(), "Preserve exited controller; explicit recovery only"
    write(root/"controller_v3.pid.json",dict(pid=os.getpid(),unix=time.time()))
    evidence={}
    rc=1
    try:
        while True:
            deferred=json.loads((root/"CAET_GOAL_DEFERRED.json").read_text())
            assert sha(root/"CAET_GOAL_FEASIBILITY_EVIDENCE.json")==deferred['feasibility_sha256']
            assert sha(root/"CAET_OBJECT_AGGREGATE.json")==deferred['aggregate_sha256']
            decision=next_action(evidence,deferred)
            write(root/"DECISION.json",dict(decision,unix=time.time()))
            if decision["action"]=="RUNNABLE_ACCEPTED_WITH_DEFERRED":
                write(root/"RUN_STATUS.json",dict(status="PARTIAL_BLOCKED",deferred=deferred,
                    runnable_branches_accepted=True,all_authorized_work_complete=False,unix=time.time()))
                rc=0;return
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
            result=verify_libero50(entry) if decision["method"]=="RA_IFA" else verify_worker(entry)
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
        (root/"controller_v3.exit").write_text(str(rc)+"\n")


if __name__=="__main__":
    main()
