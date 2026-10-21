"""Explicit idle-only scheduler handoff; never alter a worker or scientific contract."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from queue_verifier import live_workers

Q=Path('/data/mlr_ifa/experiment_workspaces/RCTR_CAET_COVR_conditional_20261020')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
assert not (Q/'CONTROLLER_V3_HANDOFF_STARTED.json').exists()
assert not (Q/'controller_v3.pid.json').exists()
registry=read(Q/'REGISTRY.json')
state=read(Q/'RUN_STATUS.json');decision=read(Q/'DECISION.json')
assert state['status']=='WAITING_IMPLEMENTATION' and state['key']=='CAET_GOAL_41'
assert (decision['method'],decision['suite'],decision['seed'])==('CAET','GOAL',41)
assert all(not live_workers(Path(e['output'])) for e in registry.values())
assert all(not (Q/'starts'/f'CAET_GOAL_{s}.json').exists() for s in (41,42,43))
assert not (Q/'starts/COVR_OBJECT_42.json').exists()
for seed in (41,42,43):
    receipt=read(Q/'receipts'/f'CAET_OBJECT_{seed}.json')
    assert receipt['engineering']=='PASS' and receipt['formal_records']==200
    assert not receipt['residual_workers']
proof=read(Q/'CAET_GOAL_DEFERRED.json')
assert sha(Q/'CAET_GOAL_FEASIBILITY_EVIDENCE.json')==proof['feasibility_sha256']
assert sha(Q/'CAET_OBJECT_AGGREGATE.json')==proof['aggregate_sha256']
feasibility=read(Q/'CAET_GOAL_FEASIBILITY_EVIDENCE.json')
assert all(feasibility['seeds'][str(s)]['stages']['7']['positive_pairs']==0 for s in (41,42,43))
from fractions import Fraction
delta=sum(Fraction(read(Q/'receipts'/f'CAET_OBJECT_{s}.json')['auc'])-Fraction(read(Q/'receipts'/f'CAET_OBJECT_{s}.json')['baseline_auc']) for s in (41,42,43))/3
assert delta==Fraction(11,720)
assert read(Q/'CAET_OBJECT_AGGREGATE.json')['summary']['paired_delta']['AUC']['mean_fraction']==str(delta)
old=read(Q/'controller_v2.pid.json')['pid'];proc=Path('/proc')/str(old)
assert b'queue_controller_v2.py' in (proc/'cmdline').read_bytes()
assert not (proc/'task'/str(old)/'children').read_text().strip()
assert not (Q/'controller_v2.exit').exists()
before=dict(old_pid=old,old_command_sha256=sha(proc/'cmdline'),old_source_sha256=sha(Q/'queue_controller_v2.py'),
    registry_sha256=sha(Q/'REGISTRY.json'),decision=decision,state=state,
    old_log_sha256=sha(Q/'controller_v2.log'),unix=time.time(),active_workers=0,
    deferred_sha256=sha(Q/'CAET_GOAL_DEFERRED.json'),scientific_source_changes=0)
(Q/'CONTROLLER_V3_HANDOFF_STARTED.json').write_text(json.dumps(before,indent=2)+'\n')
os.kill(old,signal.SIGTERM)
for _ in range(50):
    if not proc.exists() or (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':break
    time.sleep(.1)
else:raise RuntimeError('Old scheduler has not exited; no second scheduler started')
assert sha(Q/'REGISTRY.json')==before['registry_sha256']
with (Q/'controller_v3.log').open('x') as log:
    child=subprocess.Popen(['/data/mlr_ifa/env/bin/python','-u',str(Q/'queue_controller_v3.py'),str(Q)],
        cwd=Q,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
time.sleep(2)
assert child.poll() is None,(Q/'controller_v3.log').read_text()
assert read(Q/'controller_v3.pid.json')['pid']==child.pid
receipt=dict(before,status='PASS',new_pid=child.pid,new_controller_sha256=sha(Q/'queue_controller_v3.py'),
    old_termination_signal=15,original_exit_not_fabricated=True,unix_complete=time.time())
(Q/'CONTROLLER_V3_HANDOFF.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
