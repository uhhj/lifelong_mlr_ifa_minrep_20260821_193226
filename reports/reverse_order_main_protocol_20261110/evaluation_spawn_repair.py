"""Guard evaluation entry from multiprocessing spawn without changing the evaluator."""
from pathlib import Path
import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
import time
import traceback

ROOT = Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
sys.path.insert(0, str(ROOT))
from reverse_common import PY, capacity, environment, event, processes, read, sha, verify, write

ARCHIVE = ROOT / 'FAILED_OBJECT_DEBR_41_PRE_ROLLOUT'
RECEIPT = ROOT / 'EVALUATION_SPAWN_REPAIR.json'
COMPLETE = 'OBJECT_MLR_41'
FAILED = 'OBJECT_DEBR_41'
CONTRACT = '167eb6d15fab4f6d9f7e4db396f826f546160edacb0e2b3049bfa60d5ee80216'
CASE_FILES = ('evaluate.log', 'evaluate.exit', 'EVAL_DISPATCH.json', 'EVAL_ADMISSION.json', 'EVAL_EVENTS.jsonl', 'EVALUATION_STARTED.json')


def check_prefix(spec, complete_hashes, failed_expected):
    assert next(iter(spec)) == COMPLETE
    completed = ROOT / 'cases' / COMPLETE
    assert {name: sha(completed / name) for name in complete_hashes} == complete_hashes
    assert (completed / 'evaluate.exit').read_text().strip() == '0'
    assert len(list((completed / 'outcomes').glob('*.json'))) == 20
    assert len(read(completed / 'NORMALIZED_OUTCOMES.json')) == 200
    for name in spec:
        case = ROOT / 'cases' / name
        assert (case / 'train.exit').read_text().strip() == '0'
        if name == COMPLETE:
            continue
        assert not list((case / 'outcomes').glob('*'))
        assert not any((case / p).exists() for p in ('OUTCOMES.json', 'NORMALIZED_OUTCOMES.json', 'EVALUATION_COMPLETE.json'))
        if not failed_expected or name != FAILED:
            assert not any((case / p).exists() for p in CASE_FILES)


def install(source, expected):
    spec = verify()['cases']
    capacity()
    assert sha(ROOT / 'CONTRACT.json') == CONTRACT and not processes()
    assert not any(p.exists() for p in (ARCHIVE, RECEIPT, ROOT / 'FINAL_CPU_REQUEST.json'))
    check_prefix(spec, expected['completed_hashes'], failed_expected=True)
    case = ROOT / 'cases' / FAILED
    paths = {'root/' + n: ROOT / n for n in ('controller.exit', 'FAILURE.json', 'STATUS.json')}
    paths.update({'case/' + n: case / n for n in CASE_FILES})
    assert {key: sha(path) for key, path in paths.items()} == expected['failed_hashes']
    assert (ROOT / 'controller.exit').read_text().strip() == (case / 'evaluate.exit').read_text().strip() == '1'
    failed_log = (case / 'evaluate.log').read_text(errors='replace')
    assert failed_log.count('the following arguments are required: case, phase') >= 1
    assert 'self.parent_remote.send(["seed", seed])' in failed_log and 'ob,_,_,_=env.step' not in failed_log
    failed_vectors = [json.loads(line) for line in (case / 'EVAL_EVENTS.jsonl').read_text().splitlines()]
    assert [(row['kind'], row['run']) for row in failed_vectors] == [('start', 'S_s6_t6_g0')]
    events = [json.loads(line) for line in (ROOT / 'EVENTS.jsonl').read_text().splitlines()]
    assert events[-4:] == expected['failed_events']

    entry = ROOT / 'evaluation_spawn_repair.py'
    with entry.open('x') as file:
        file.write(source)
    probe = subprocess.run([PY, '-B', str(entry), 'spawn-check'], cwd=ROOT, env=environment(),
                           capture_output=True, text=True, timeout=30, check=True)
    assert probe.stdout.strip() == 'PASS_GUARDED_SPAWN_NO_ENVIRONMENT'
    assert not processes()
    ARCHIVE.mkdir()
    moved = []
    started = False
    try:
        for key, path in paths.items():
            target = ARCHIVE / key
            target.parent.mkdir(exist_ok=True)
            path.rename(target)
            moved.append((target, path))
            assert sha(target) == expected['failed_hashes'][key]
        proof = dict(status='PRE_ROLLOUT_SPAWN_ENTRY_REPAIRED', observed=time.time(), contract=CONTRACT,
                     case=FAILED, failed_vectors=failed_vectors, failed_archived_hashes=expected['failed_hashes'],
                     failed_events=expected['failed_events'], resets=0, steps=0, formal_outcomes=0,
                     new_training_updates=0, acquisition_repeated=0, archive=str(ARCHIVE),
                     completed_case=COMPLETE, completed_formal=200, completed_hashes=expected['completed_hashes'],
                     guarded_spawn_probe=probe.stdout.strip(), source_sha256=sha(entry))
        write(RECEIPT, proof)
        with (ROOT / 'evaluation_spawn_repair.log').open('x') as log:
            worker = subprocess.Popen([PY, '-B', str(entry), 'queue'], cwd=ROOT, env=environment(),
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = True
        return dict(proof, controller=worker.pid, receipt_sha256=sha(RECEIPT))
    finally:
        if not started:
            for target, path in reversed(moved):
                target.rename(path)


def run_queue():
    # Keep the frozen controller's lock, status, dispatch and exit accounting.
    # This process-local adapter only changes the DEBR worker's executable entry.
    import controller
    from types import SimpleNamespace

    entry = Path(__file__).resolve()

    def guarded_popen(command, **kwargs):
        assert command[:3] == [PY, '-B', str(ROOT / 'run_case.py')] and command[4:] == ['evaluate']
        if '_DEBR_' in command[3]:
            command[2:] = [str(entry), 'case', command[3]]
        return subprocess.Popen(command, **kwargs)

    controller.subprocess = SimpleNamespace(Popen=guarded_popen, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT)
    with (ROOT / 'evaluation_spawn_repair.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = 1
        try:
            proof = read(RECEIPT)
            assert sha(ROOT / 'CONTRACT.json') == CONTRACT == proof['contract'] and sha(entry) == proof['source_sha256']
            for key, digest in proof['failed_archived_hashes'].items():
                assert sha(ARCHIVE / key) == digest
            spec = verify()['cases']
            check_prefix(spec, proof['completed_hashes'], failed_expected=False)
            assert not (ROOT / 'controller.exit').exists()
            event('parallel_dispatch_started', pid=os.getpid(), evaluation_spawn_repair_sha256=sha(RECEIPT))
            for suite in ('OBJECT', 'GOAL'):
                for case in spec:
                    if spec[case]['suite'] == suite and case != COMPLETE:
                        verify()
                        capacity()
                        controller.evaluate(case)
                write(ROOT / (suite + '_COMPLETE.json'), dict(status='PASS_EXECUTION_PENDING_FINAL', formal=1200, training_runs=6, time=time.time()))
            event('controller_complete', pid=os.getpid())
            write(ROOT / 'STATUS.json', dict(status='EXECUTION_COMPLETE_PENDING_CPU_ACCEPTANCE', time=time.time(), controller=os.getpid()))
            code = 0
        except BaseException:
            write(ROOT / 'FAILURE.json', dict(time=time.time(), traceback=traceback.format_exc()))
            raise
        finally:
            with (ROOT / 'controller.exit').open('x') as file:
                file.write(str(code) + '\n')


if __name__ == '__main__':
    operation = sys.argv[1] if len(sys.argv) > 1 else None
    if operation == 'queue':
        run_queue()
    elif operation == 'case':
        case = sys.argv[2]
        assert read(ROOT / 'SPEC.json')['cases'][case]['method'] == 'DEBR'
        sys.argv = [str(ROOT / 'run_case.py'), case, 'evaluate']
        import run_case
    elif operation == 'spawn-check':
        sys.argv = ['evaluate.py']
        child = multiprocessing.get_context('spawn').Process(target=time.sleep, args=(0,))
        child.start()
        child.join(15)
        assert child.exitcode == 0
        print('PASS_GUARDED_SPAWN_NO_ENVIRONMENT')
