"""Recover the zero-update GOAL startup failure and keep the current OBJECT worker."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

import parallel_dispatch_v1 as previous
from reverse_common import PY, Q, capacity, environment, event, read, sha, verify, write

FAILED_CASE = 'GOAL_MLR_41'
STATE = Q / 'PARALLEL_REPAIR_HANDOFF.json'
READY = Q / 'PARALLEL_REPAIR_READY.json'
RELEASE = Q / 'PARALLEL_REPAIR_RELEASE.json'
ENTRY = Q / 'goal_mlr_entry.py'
GOAL_MLR = ('GOAL_MLR_41', 'GOAL_MLR_42', 'GOAL_MLR_43')


def require_failed_state():
    current = read(Q / 'STATUS.json')
    v1 = read(previous.RELEASE)
    inherited = read(previous.HANDOFF)
    assert current['controller'] == v1['pid'] and current['status'] == 'FAILED_DRAINING'
    assert current['failures'] == [dict(case=FAILED_CASE, returncode=1)]
    assert current['active'] == {previous.INHERITED_CASE: inherited['inherited_launcher']['pid']}
    previous.identity(inherited['original_controller'], stopped=True)
    previous.identity(inherited['inherited_launcher'])
    assert not (Q / 'controller.exit').exists()
    old_dispatcher = previous.process(v1['pid'])
    case = Q / 'cases' / FAILED_CASE
    failed_pid = read(case / 'TRAIN_DISPATCH.json')['pid']
    assert not (Path('/proc') / str(failed_pid)).exists()
    assert (case / 'train.exit').read_text().strip() == '1'
    counts = read(case / 'OPTIMIZER_CALLS.json')
    assert counts['status'] == 'ACTUAL_CALLS_FAILED_PARTIAL' and counts['environments'] == 0
    assert not any(counts['calls'].values()) and not any(counts['extra_autograd'].values())
    inputs = read(case / 'INPUT.json')
    output = Path(inputs['output_root'])
    assert not any((output / name).exists() for name in ('epochs.json', 'recovery', 'models', 'TRAINING_COMPLETE.json'))
    assert 'ORCHESTRATION_SHA256.json' in (case / 'train.log').read_text(errors='replace')
    assert not list(case.glob('ADMISSION_MLR_*.json'))
    return current, old_dispatcher, inherited


def prepare_checks():
    proofs = {}
    for case in GOAL_MLR:
        outcome = subprocess.run([PY, '-B', str(ENTRY), case, 'verify-only'], cwd=Q, env=environment(),
                                 capture_output=True, text=True)
        assert outcome.returncode == 0, (case, outcome.stderr[-3500:])
        result = json.loads(outcome.stdout.splitlines()[-1])
        assert result['status'] == 'PASS_ACTUAL_REVERSE_NAMESPACE_VERIFICATION'
        proofs[case] = result
    return proofs


def install(expected):
    verify()
    capacity()
    assert sha(Q / 'CONTRACT.json') == expected and not STATE.exists() and not RELEASE.exists()
    require_failed_state()
    proofs = prepare_checks()
    status, v1, inherited = require_failed_state()
    archived = []
    paused = False
    released = False
    runner = None
    try:
        os.kill(v1['pid'], signal.SIGSTOP)
        paused = True
        until = time.monotonic() + 5
        while previous.process(v1['pid'])['state'] not in ('T', 't') and time.monotonic() < until:
            time.sleep(0.02)
        previous.identity(v1, stopped=True)
        require_failed_state()
        case = Q / 'cases' / FAILED_CASE
        archive = Q / 'FAILED_GOAL_MLR_41_STARTUP_20261111_v1'
        archive.mkdir()
        for name in ('train.log', 'train.exit', 'TRAIN_DISPATCH.json', 'OPTIMIZER_CALLS.json', 'ACTUAL_ORDER.json'):
            source = case / name
            target = archive / name
            source.rename(target)
            archived.append((source, target))
        write(archive / 'FAILURE_STATUS.json', status)
        evidence = dict(status='PREPARED_ZERO_UPDATE_STARTUP_REPAIR', observed=time.time(), root_contract=expected,
                        retired_dispatcher=v1, inherited=inherited, preflight=proofs, entry_sha256=sha(ENTRY),
                        dispatcher_sha256=sha(__file__), previous_dispatcher_sha256=sha(Q / 'parallel_dispatch_v1.py'),
                        failed_attempt_hashes={target.name: sha(target) for _, target in archived},
                        new_optimizer=0, new_environments=0)
        write(STATE, evidence)
        with (Q / 'parallel_dispatch_v2.log').open('x') as log:
            runner = subprocess.Popen([PY, '-B', str(Path(__file__).resolve()), 'run'], cwd=Q, env=environment(),
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        until = time.monotonic() + 40
        while not READY.exists() and runner.poll() is None and time.monotonic() < until:
            time.sleep(0.1)
        assert read(READY)['pid'] == runner.pid and runner.poll() is None
        write(RELEASE, dict(pid=runner.pid, observed=time.time(), handoff_sha256=sha(STATE)))
        released = True
        previous.identity(v1, stopped=True)
        os.kill(v1['pid'], signal.SIGKILL)
    finally:
        if not released:
            try:
                if runner is not None and runner.poll() is None:
                    runner.terminate()
                    runner.wait(timeout=10)
                for source, target in reversed(archived):
                    assert not source.exists()
                    target.rename(source)
            finally:
                if paused:
                    previous.identity(v1, stopped=True)
                    os.kill(v1['pid'], signal.SIGCONT)
    return dict(status='REPAIRED_DISPATCHER_RELEASED', pid=runner.pid, observed=time.time(), handoff=evidence)


def launch_goal_mlr(case):
    directory = Q / 'cases' / case
    assert not (directory / 'TRAIN_DISPATCH.json').exists() and not (directory / 'train.exit').exists()
    identity = uuid.uuid4().hex
    command = [PY, '-B', '-m', 'torch.distributed.run', '--nnodes=1', '--nproc_per_node=1', '--rdzv_backend=c10d',
               '--rdzv_endpoint=localhost:0', '--rdzv_id=' + identity, str(ENTRY), case, 'train']
    with (directory / 'train.log').open('x') as log:
        worker = subprocess.Popen(command, cwd=directory, env=environment(), stdin=subprocess.DEVNULL,
                                  stdout=log, stderr=subprocess.STDOUT)
    write(directory / 'TRAIN_DISPATCH.json', dict(time=time.time(), pid=worker.pid, cmd=command, rdzv_id=identity,
          engineering_entry_sha256=sha(ENTRY)))
    event('train_dispatch', case=case, pid=worker.pid)
    return worker


def run():
    handoff = read(STATE)
    assert handoff['root_contract'] == sha(Q / 'CONTRACT.json') and handoff['dispatcher_sha256'] == sha(__file__)
    assert handoff['entry_sha256'] == sha(ENTRY) and handoff['previous_dispatcher_sha256'] == sha(Q / 'parallel_dispatch_v1.py')
    verify()
    previous.identity(handoff['inherited']['original_controller'], stopped=True)
    original_launch = previous.original.launch

    def launch(case):
        return launch_goal_mlr(case) if case in GOAL_MLR else original_launch(case)

    previous.original.launch = launch
    with (Q / 'parallel_dispatch_v2.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(READY, dict(pid=os.getpid(), observed=time.time()))
        until = time.monotonic() + 45
        while not RELEASE.exists() and time.monotonic() < until:
            time.sleep(0.1)
        release = read(RELEASE)
        assert release['pid'] == os.getpid() and release['handoff_sha256'] == sha(STATE)
        code = 1
        try:
            event('parallel_dispatch_started', pid=os.getpid(), original_controller=handoff['retired_dispatcher']['pid'])
            previous.train_remaining(handoff['inherited'])
            for suite in ('OBJECT', 'GOAL'):
                for case in read(Q / 'SPEC.json')['cases']:
                    if case.startswith(suite + '_'):
                        verify()
                        capacity()
                        previous.original.evaluate(case)
                write(Q / (suite + '_COMPLETE.json'), dict(status='PASS_EXECUTION_PENDING_FINAL', formal=1200, training_runs=6, time=time.time()))
            event('controller_complete', pid=os.getpid())
            write(Q / 'STATUS.json', dict(status='EXECUTION_COMPLETE_PENDING_CPU_ACCEPTANCE', time=time.time(), controller=os.getpid()))
            code = 0
        except BaseException:
            write(Q / 'FAILURE.json', dict(time=time.time(), traceback=traceback.format_exc()))
            raise
        finally:
            with (Q / 'controller.exit').open('x') as file:
                file.write(str(code) + '\n')


if __name__ == '__main__':
    assert sys.argv[1:] == ['run']
    run()
