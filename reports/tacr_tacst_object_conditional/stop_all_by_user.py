"""Explicit user cancellation: stop only this finite queue; preserve every artifact."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

Q = Path('/data/mlr_ifa/experiment_workspaces/TACR_TACST_OBJECT_conditional_20261023').resolve()
assert Q.name == 'TACR_TACST_OBJECT_conditional_20261023'
assert Q.parent == Path('/data/mlr_ifa/experiment_workspaces')
my_pid = os.getpid()

def process(pid):
    try:
        p = Path('/proc')/str(pid)
        cwd = (p/'cwd').resolve()
        if pid == my_pid or not (cwd == Q or Q in cwd.parents):
            return None
        stat = (p/'stat').read_text().rsplit(')', 1)[1].split()
        if stat[0] == 'Z':
            return None
        return dict(pid=pid, ppid=int(stat[1]), start_ticks=int(stat[19]), cwd=str(cwd),
                    command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace'))
    except (OSError, ValueError, IndexError):
        return None

def census():
    return [r for p in Path('/proc').glob('[0-9]*') if (r := process(int(p.name))) is not None]

def signal_same(row, sig):
    actual = process(row['pid'])
    if actual and actual['start_ticks'] == row['start_ticks']:
        os.kill(row['pid'], sig)
        actions.append(dict(pid=row['pid'], start_ticks=row['start_ticks'], signal=sig.name, unix=time.time()))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

before = census()
actions = []
controllers = [r for r in before if '/controller.py' in r['command']]
for r in controllers:
    signal_same(r, signal.SIGSTOP)
# Once dispatcher is quiescent, no new entry can launch while its worker tree stops.
before = census()
for r in before:
    signal_same(r, signal.SIGSTOP)
contract = json.loads((Q/'ORCHESTRATION_CONTRACT.json').read_text())
hash_before = [p for p,h in contract['files'].items() if sha(p) != h]
original_queue_state = json.loads((Q/'RUN_STATUS.json').read_text())
for r in before:
    signal_same(r, signal.SIGTERM)
for r in before:
    signal_same(r, signal.SIGCONT)
deadline = time.time()+12
while time.time() < deadline and census():
    time.sleep(.25)
remaining = census()
for r in remaining:
    signal_same(r, signal.SIGKILL)
deadline = time.time()+5
while time.time() < deadline and census():
    time.sleep(.25)
after = census()
hash_after = [p for p,h in contract['files'].items() if sha(p) != h]
entries = json.loads((Q/'QUEUE.json').read_text())
artifacts = {}
for entry in entries:
    O = Path(entry['output'])
    if not O.exists():
        artifacts[entry['key']] = dict(status='CANCELLED_OR_SKIPPED_NOT_STARTED', output=str(O))
        continue
    ep = O/'ra_coreset/epochs.json'
    history = json.loads(ep.read_text()) if ep.exists() else []
    checkpoints = [dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(O.rglob('*.pth')) if p.is_file() and not p.is_symlink()]
    records = sum(len(p.read_text().splitlines()) for p in (O/'ra_coreset/evaluation').glob('trials*.jsonl'))
    complete = (Q/(entry['key']+'.receipt.json')).exists()
    artifacts[entry['key']] = dict(status='COMPLETED_PRESERVED' if complete else 'EARLY_TERMINATED_BY_USER',
        output=str(O), complete_epochs=len(history), main_steps=history[-1]['steps'] if history else 0,
        last_complete_epoch=history[-1] if history else None, formal_records=records,
        checkpoints=checkpoints, raw_exit_files={p.name:p.read_text() for p in O.glob('*.exit')},
        receipt_preserved=complete)
result = dict(status='USER_CANCELLED_ALL_TACR_TACRST_STOP_CONFIRMED' if not after else 'USER_CANCELLATION_RESIDUAL_PROCESS_REQUIRES_ACTION',
    observed_unix=time.time(), reason='User explicitly stopped training: TAC-R design has a problem.',
    queue=str(Q), original_queue_status_preserved=original_queue_state,
    before_processes=before, signal_actions=actions, residual_processes=after,
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],universal_newlines=True),
    frozen_files=len(contract['files']), frozen_hash_failures_before=hash_before, frozen_hash_failures_after=hash_after,
    entries=artifacts, no_artifacts_deleted=True, no_checkpoint_download=True,
    future_expansions_cancelled=True, no_training_or_evaluation_restart=True,
    original_exit_files_not_rewritten=True)
marker=Q/'USER_TERMINATION_BY_USER.json'
assert not marker.exists(), 'Retain existing cancellation evidence; do not overwrite'
marker.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
assert not after, after
