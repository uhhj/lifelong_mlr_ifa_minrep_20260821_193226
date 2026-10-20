"""Replace only idle-waiting orchestration; keep the active experiment untouched."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
Q=Path("/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert not (Q/"CONTROLLER_V2_HANDOFF.json").exists()
assert json.loads((Q/"LIBERO50_REGISTRY_ACCEPTANCE.json").read_text())["status"]=="REGISTERED_GPU_GATED"
old=json.loads((Q/"controller.pid.json").read_text())["pid"]
proc=Path("/proc")/str(old)
cmd=(proc/"cmdline").read_bytes()
assert b"queue_controller.py" in cmd and b"queue_controller_v2.py" not in cmd
children=(proc/"task"/str(old)/"children").read_text().strip()
assert not children,"Controller is waiting for a launcher child; no interruption allowed"
decision=json.loads((Q/"DECISION.json").read_text())
assert (decision["method"],decision["suite"],decision["seed"])==("RCTR","OBJECT",42)
out=Path("/data/mlr_ifa/experiments/RCTR_OBJECT_seed42_full_20261021")
workers={}
for p in Path("/proc").glob("[0-9]*"):
    try:
        if ("CORESET_OUT="+str(out)).encode() in (p/"environ").read_bytes().split(b"\0"):
            workers[p.name]=dict(command_sha256=sha(p/"cmdline"),
                start_ticks=(p/"stat").read_text().rsplit(")",1)[1].split()[19])
    except OSError:pass
assert workers
before=dict(old_pid=old,old_command_sha256=sha(proc/"cmdline"),active_workers=workers,
    old_pid_receipt_sha256=sha(Q/"controller.pid.json"),unix=time.time())
(Q/"CONTROLLER_V2_HANDOFF_STARTED.json").write_text(json.dumps(before,indent=2)+"\n")
os.kill(old,signal.SIGTERM)
for _ in range(50):
    if not proc.exists() or (proc/"stat").read_text().rsplit(")",1)[1].split()[0]=="Z":break
    time.sleep(.1)
else:raise RuntimeError("Old controller has not exited; no duplicate scheduler")
for pid,info in workers.items():
    p=Path("/proc")/pid
    assert p.exists() and sha(p/"cmdline")==info["command_sha256"]
    assert (p/"stat").read_text().rsplit(")",1)[1].split()[19]==info["start_ticks"]
with (Q/"controller_v2.log").open("x") as log:
    child=subprocess.Popen(["/data/mlr_ifa/env/bin/python","-u",str(Q/"queue_controller_v2.py"),str(Q)],
        cwd=Q,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
time.sleep(2)
assert child.poll() is None,(Q/"controller_v2.log").read_text()
assert json.loads((Q/"controller_v2.pid.json").read_text())["pid"]==child.pid
receipt=dict(before,status="PASS",new_pid=child.pid,new_controller_sha256=sha(Q/"queue_controller_v2.py"),
    old_controller_termination_signal=15,original_exit_not_fabricated=True,
    experiment_workers_preserved=True,scientific_source_changes=0,unix_complete=time.time())
(Q/"CONTROLLER_V2_HANDOFF.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(status="PASS",old_pid=old,new_pid=child.pid,preserved_workers=list(workers))))
