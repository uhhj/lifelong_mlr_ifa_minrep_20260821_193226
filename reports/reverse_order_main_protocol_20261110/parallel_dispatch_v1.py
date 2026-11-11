"""Continue the frozen experiment with OBJECT and GOAL sharing two training slots."""
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import controller as original
from reverse_common import PY, Q, capacity, environment, event, read, sha, verify, write

HANDOFF = Q / 'PARALLEL_DISPATCH_HANDOFF.json'
READY = Q / 'PARALLEL_DISPATCH_READY.json'
RELEASE = Q / 'PARALLEL_DISPATCH_RELEASE.json'
INHERITED_CASE = 'OBJECT_DEBR_43'


def process(pid):
    path = Path('/proc') / str(pid) / 'stat'
    raw = path.read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return dict(pid=pid, state=fields[0], ppid=int(fields[1]), starttime=fields[19], wait_status=int(fields[49]))


def identity(expected, stopped=False):
    current = process(expected['pid'])
    assert current['starttime'] == expected['starttime'] and current['ppid'] == expected['ppid'], (expected, current)
    if stopped:
        assert current['state'] in ('T', 't'), current
    return current


def original_dispatcher():
    launched = read(Q / 'LAUNCH.json')
    assert launched['pid'] == 119466
    state = process(launched['pid'])
    args = (Path('/proc') / str(state['pid']) / 'cmdline').read_bytes().split(b'\0')
    assert str(Q / 'controller.py').encode() in args
    return state


def check_transition(parent):
    status = read(Q / 'STATUS.json')
    assert status['status'] == 'PARTIAL_TRAINING' and status['suite'] == 'OBJECT'
    assert not status['failures'] and not status['pending'] and set(status['active']) == {INHERITED_CASE}
    dispatch = read(Q / 'cases' / INHERITED_CASE / 'TRAIN_DISPATCH.json')
    worker = process(dispatch['pid'])
    assert worker['ppid'] == parent['pid'] and worker['pid'] == status['active'][INHERITED_CASE]
    cases = read(Q / 'SPEC.json')['cases']
    for case in cases:
        directory = Q / 'cases' / case
        assert not (directory / 'EVAL_DISPATCH.json').exists()
        if case.startswith('GOAL_'):
            assert not (directory / 'TRAIN_DISPATCH.json').exists() and not (directory / 'train.exit').exists()
        elif case != INHERITED_CASE:
            assert (directory / 'train.exit').read_text().strip() == '0'
    return worker


def install(expected_contract):
    verify()
    capacity()
    assert sha(Q / 'CONTRACT.json') == expected_contract
    assert not (Q / 'controller.exit').exists() and not HANDOFF.exists() and not RELEASE.exists()
    parent = original_dispatcher()
    inherited = check_transition(parent)
    paused = False
    runner = None
    released = False
    try:
        os.kill(parent['pid'], signal.SIGSTOP)
        paused = True
        until = time.monotonic() + 5
        while process(parent['pid'])['state'] not in ('T', 't') and time.monotonic() < until:
            time.sleep(0.02)
        identity(parent, stopped=True)
        assert check_transition(parent)['starttime'] == inherited['starttime']
        handoff = dict(status='USER_AUTHORIZED_PARALLEL_GOAL_TRAINING', observed=time.time(),
                       original_controller=parent, inherited_launcher=inherited, inherited_case=INHERITED_CASE,
                       root_contract=expected_contract, dispatcher_sha256=sha(__file__), max_training_slots=2,
                       evaluation_order=['OBJECT', 'GOAL'], formal_evaluation_after_all_training=True)
        write(HANDOFF, handoff)
        with (Q / 'parallel_dispatch_v1.log').open('x') as log:
            runner = subprocess.Popen([PY, '-B', str(Path(__file__).resolve()), 'run'], cwd=Q, env=environment(),
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        until = time.monotonic() + 40
        while not READY.exists() and runner.poll() is None and time.monotonic() < until:
            time.sleep(0.1)
        assert read(READY)['pid'] == runner.pid and runner.poll() is None
        write(RELEASE, dict(pid=runner.pid, observed=time.time(), handoff_sha256=sha(HANDOFF)))
        released = True
    finally:
        if not released:
            if runner is not None and runner.poll() is None:
                runner.terminate()
                runner.wait(timeout=10)
            if paused:
                identity(parent, stopped=True)
                os.kill(parent['pid'], signal.SIGCONT)
    return dict(status='DISPATCHER_RELEASED', pid=runner.pid, observed=time.time(), handoff=handoff, release=read(RELEASE))


def exit_code(status):
    if os.WIFEXITED(status):
        return os.WEXITSTATUS(status)
    assert os.WIFSIGNALED(status), status
    return -os.WTERMSIG(status)


def record_exit(case, pid, code):
    with (Q / 'cases' / case / 'train.exit').open('x') as file:
        file.write(str(code) + '\n')
    event('train_exit', case=case, pid=pid, returncode=code)


def finish_inherited(parent, child):
    code = exit_code(child['wait_status'])
    if code == 0:
        directory = Q / 'cases' / INHERITED_CASE
        case = read(Q / 'SPEC.json')['cases'][INHERITED_CASE]
        epochs = read(directory / 'EPOCHS_S.json')
        counters = read(directory / 'OPTIMIZER_CALLS.json')
        assert len(epochs) == 400 and epochs[-1]['steps'] == case['accepted_budget']
        assert counters['status'] == 'ACTUAL_CALLS_COMPLETED' and counters['calls']['production'] == case['physical_budget']
    write(Q / 'PARALLEL_INHERITED_TRAIN_EXIT.json', dict(case=INHERITED_CASE, observed=time.time(),
          original_controller=parent, kernel_process=child, returncode=code, completed_training_repeated=False))
    record_exit(INHERITED_CASE, child['pid'], code)
    identity(parent, stopped=True)
    os.kill(parent['pid'], signal.SIGKILL)
    event('retire_superseded_dispatcher', pid=parent['pid'], after_natural_training_exit=True)
    return code


def train_remaining(handoff):
    parent = handoff['original_controller']
    inherited = handoff['inherited_launcher']
    pending = [case for case in read(Q / 'SPEC.json')['cases'] if case.startswith('GOAL_')]
    active = {}
    failures = []
    while pending or active or inherited is not None:
        if inherited is not None:
            identity(parent, stopped=True)
            child = identity(inherited)
            if child['state'] == 'Z':
                code = finish_inherited(parent, child)
                inherited = None
                if code:
                    failures.append(dict(case=INHERITED_CASE, returncode=code))
        for case, worker in list(active.items()):
            code = worker.poll()
            if code is None:
                continue
            record_exit(case, worker.pid, code)
            del active[case]
            if code:
                failures.append(dict(case=case, returncode=code))
        while pending and len(active) + int(inherited is not None) < 2 and not failures:
            verify()
            capacity()
            case = pending.pop(0)
            active[case] = original.launch(case)
        live = {case: worker.pid for case, worker in active.items()}
        if inherited is not None:
            live[INHERITED_CASE] = inherited['pid']
        write(Q / 'STATUS.json', dict(status='FAILED_DRAINING' if failures else 'PARTIAL_TRAINING',
              suite='OBJECT+GOAL' if inherited is not None else 'GOAL', controller=os.getpid(), active=live,
              pending=pending, failures=failures, time=time.time()))
        if failures and not live:
            raise RuntimeError(('Training failed', failures))
        if live or pending:
            time.sleep(10)


def run():
    handoff = read(HANDOFF)
    assert handoff['root_contract'] == sha(Q / 'CONTRACT.json') and handoff['dispatcher_sha256'] == sha(__file__)
    verify()
    identity(handoff['original_controller'], stopped=True)
    lock_path = Q / 'parallel_dispatch_v1.lock'
    with lock_path.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(READY, dict(pid=os.getpid(), observed=time.time()))
        until = time.monotonic() + 45
        while not RELEASE.exists() and time.monotonic() < until:
            time.sleep(0.1)
        release = read(RELEASE)
        assert release['pid'] == os.getpid() and release['handoff_sha256'] == sha(HANDOFF)
        code = 1
        try:
            event('parallel_dispatch_started', pid=os.getpid(), original_controller=handoff['original_controller']['pid'])
            train_remaining(handoff)
            for suite in ('OBJECT', 'GOAL'):
                for case in read(Q / 'SPEC.json')['cases']:
                    if case.startswith(suite + '_'):
                        verify()
                        capacity()
                        original.evaluate(case)
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
