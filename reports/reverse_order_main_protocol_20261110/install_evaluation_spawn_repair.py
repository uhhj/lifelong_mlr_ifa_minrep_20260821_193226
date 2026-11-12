SOURCE='"""Guard evaluation entry from multiprocessing spawn without changing the evaluator."""\nfrom pathlib import Path\nimport fcntl\nimport json\nimport multiprocessing\nimport os\nimport subprocess\nimport sys\nimport time\nimport traceback\n\nROOT = Path(\'/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05\')\nsys.path.insert(0, str(ROOT))\nfrom reverse_common import PY, capacity, environment, event, processes, read, sha, verify, write\n\nARCHIVE = ROOT / \'FAILED_OBJECT_DEBR_41_PRE_ROLLOUT\'\nRECEIPT = ROOT / \'EVALUATION_SPAWN_REPAIR.json\'\nCOMPLETE = \'OBJECT_MLR_41\'\nFAILED = \'OBJECT_DEBR_41\'\nCONTRACT = \'167eb6d15fab4f6d9f7e4db396f826f546160edacb0e2b3049bfa60d5ee80216\'\nCASE_FILES = (\'evaluate.log\', \'evaluate.exit\', \'EVAL_DISPATCH.json\', \'EVAL_ADMISSION.json\', \'EVAL_EVENTS.jsonl\', \'EVALUATION_STARTED.json\')\n\n\ndef check_prefix(spec, complete_hashes, failed_expected):\n    assert next(iter(spec)) == COMPLETE\n    completed = ROOT / \'cases\' / COMPLETE\n    assert {name: sha(completed / name) for name in complete_hashes} == complete_hashes\n    assert (completed / \'evaluate.exit\').read_text().strip() == \'0\'\n    assert len(list((completed / \'outcomes\').glob(\'*.json\'))) == 20\n    assert len(read(completed / \'NORMALIZED_OUTCOMES.json\')) == 200\n    for name in spec:\n        case = ROOT / \'cases\' / name\n        assert (case / \'train.exit\').read_text().strip() == \'0\'\n        if name == COMPLETE:\n            continue\n        assert not list((case / \'outcomes\').glob(\'*\'))\n        assert not any((case / p).exists() for p in (\'OUTCOMES.json\', \'NORMALIZED_OUTCOMES.json\', \'EVALUATION_COMPLETE.json\'))\n        if not failed_expected or name != FAILED:\n            assert not any((case / p).exists() for p in CASE_FILES)\n\n\ndef install(source, expected):\n    spec = verify()[\'cases\']\n    capacity()\n    assert sha(ROOT / \'CONTRACT.json\') == CONTRACT and not processes()\n    assert not any(p.exists() for p in (ARCHIVE, RECEIPT, ROOT / \'FINAL_CPU_REQUEST.json\'))\n    check_prefix(spec, expected[\'completed_hashes\'], failed_expected=True)\n    case = ROOT / \'cases\' / FAILED\n    paths = {\'root/\' + n: ROOT / n for n in (\'controller.exit\', \'FAILURE.json\', \'STATUS.json\')}\n    paths.update({\'case/\' + n: case / n for n in CASE_FILES})\n    assert {key: sha(path) for key, path in paths.items()} == expected[\'failed_hashes\']\n    assert (ROOT / \'controller.exit\').read_text().strip() == (case / \'evaluate.exit\').read_text().strip() == \'1\'\n    failed_log = (case / \'evaluate.log\').read_text(errors=\'replace\')\n    assert failed_log.count(\'the following arguments are required: case, phase\') >= 1\n    assert \'self.parent_remote.send(["seed", seed])\' in failed_log and \'ob,_,_,_=env.step\' not in failed_log\n    failed_vectors = [json.loads(line) for line in (case / \'EVAL_EVENTS.jsonl\').read_text().splitlines()]\n    assert [(row[\'kind\'], row[\'run\']) for row in failed_vectors] == [(\'start\', \'S_s6_t6_g0\')]\n    events = [json.loads(line) for line in (ROOT / \'EVENTS.jsonl\').read_text().splitlines()]\n    assert events[-4:] == expected[\'failed_events\']\n\n    entry = ROOT / \'evaluation_spawn_repair.py\'\n    with entry.open(\'x\') as file:\n        file.write(source)\n    probe = subprocess.run([PY, \'-B\', str(entry), \'spawn-check\'], cwd=ROOT, env=environment(),\n                           capture_output=True, text=True, timeout=30, check=True)\n    assert probe.stdout.strip() == \'PASS_GUARDED_SPAWN_NO_ENVIRONMENT\'\n    assert not processes()\n    ARCHIVE.mkdir()\n    moved = []\n    started = False\n    try:\n        for key, path in paths.items():\n            target = ARCHIVE / key\n            target.parent.mkdir(exist_ok=True)\n            path.rename(target)\n            moved.append((target, path))\n            assert sha(target) == expected[\'failed_hashes\'][key]\n        proof = dict(status=\'PRE_ROLLOUT_SPAWN_ENTRY_REPAIRED\', observed=time.time(), contract=CONTRACT,\n                     case=FAILED, failed_vectors=failed_vectors, failed_archived_hashes=expected[\'failed_hashes\'],\n                     failed_events=expected[\'failed_events\'], resets=0, steps=0, formal_outcomes=0,\n                     new_training_updates=0, acquisition_repeated=0, archive=str(ARCHIVE),\n                     completed_case=COMPLETE, completed_formal=200, completed_hashes=expected[\'completed_hashes\'],\n                     guarded_spawn_probe=probe.stdout.strip(), source_sha256=sha(entry))\n        write(RECEIPT, proof)\n        with (ROOT / \'evaluation_spawn_repair.log\').open(\'x\') as log:\n            worker = subprocess.Popen([PY, \'-B\', str(entry), \'queue\'], cwd=ROOT, env=environment(),\n                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\n        started = True\n        return dict(proof, controller=worker.pid, receipt_sha256=sha(RECEIPT))\n    finally:\n        if not started:\n            for target, path in reversed(moved):\n                target.rename(path)\n\n\ndef run_queue():\n    # Keep the frozen controller\'s lock, status, dispatch and exit accounting.\n    # This process-local adapter only changes the DEBR worker\'s executable entry.\n    import controller\n    from types import SimpleNamespace\n\n    entry = Path(__file__).resolve()\n\n    def guarded_popen(command, **kwargs):\n        assert command[:3] == [PY, \'-B\', str(ROOT / \'run_case.py\')] and command[4:] == [\'evaluate\']\n        if \'_DEBR_\' in command[3]:\n            command[2:] = [str(entry), \'case\', command[3]]\n        return subprocess.Popen(command, **kwargs)\n\n    controller.subprocess = SimpleNamespace(Popen=guarded_popen, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT)\n    with (ROOT / \'evaluation_spawn_repair.lock\').open(\'a+\') as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n        code = 1\n        try:\n            proof = read(RECEIPT)\n            assert sha(ROOT / \'CONTRACT.json\') == CONTRACT == proof[\'contract\'] and sha(entry) == proof[\'source_sha256\']\n            for key, digest in proof[\'failed_archived_hashes\'].items():\n                assert sha(ARCHIVE / key) == digest\n            spec = verify()[\'cases\']\n            check_prefix(spec, proof[\'completed_hashes\'], failed_expected=False)\n            assert not (ROOT / \'controller.exit\').exists()\n            event(\'parallel_dispatch_started\', pid=os.getpid(), evaluation_spawn_repair_sha256=sha(RECEIPT))\n            for suite in (\'OBJECT\', \'GOAL\'):\n                for case in spec:\n                    if spec[case][\'suite\'] == suite and case != COMPLETE:\n                        verify()\n                        capacity()\n                        controller.evaluate(case)\n                write(ROOT / (suite + \'_COMPLETE.json\'), dict(status=\'PASS_EXECUTION_PENDING_FINAL\', formal=1200, training_runs=6, time=time.time()))\n            event(\'controller_complete\', pid=os.getpid())\n            write(ROOT / \'STATUS.json\', dict(status=\'EXECUTION_COMPLETE_PENDING_CPU_ACCEPTANCE\', time=time.time(), controller=os.getpid()))\n            code = 0\n        except BaseException:\n            write(ROOT / \'FAILURE.json\', dict(time=time.time(), traceback=traceback.format_exc()))\n            raise\n        finally:\n            with (ROOT / \'controller.exit\').open(\'x\') as file:\n                file.write(str(code) + \'\\n\')\n\n\nif __name__ == \'__main__\':\n    operation = sys.argv[1] if len(sys.argv) > 1 else None\n    if operation == \'queue\':\n        run_queue()\n    elif operation == \'case\':\n        case = sys.argv[2]\n        assert read(ROOT / \'SPEC.json\')[\'cases\'][case][\'method\'] == \'DEBR\'\n        sys.argv = [str(ROOT / \'run_case.py\'), case, \'evaluate\']\n        import run_case\n    elif operation == \'spawn-check\':\n        sys.argv = [\'evaluate.py\']\n        child = multiprocessing.get_context(\'spawn\').Process(target=time.sleep, args=(0,))\n        child.start()\n        child.join(15)\n        assert child.exitcode == 0\n        print(\'PASS_GUARDED_SPAWN_NO_ENVIRONMENT\')\n'
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

print(json.dumps(install(SOURCE, {'failed_hashes': {'root/controller.exit': '4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865', 'root/FAILURE.json': '9d94b46445c253b4eb3dbdef71cb777401684264f6e1a2aef4f71e45bea36fcd', 'root/STATUS.json': 'ba801db43c79c4a4a8ca44ac7133661b8936c04dbad8c6789f3c36a302f057d0', 'case/evaluate.log': 'e7416caee7578899c5125e60e6c5e89828fece54ece467d401c21129ac1c281e', 'case/evaluate.exit': '4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865', 'case/EVAL_DISPATCH.json': 'a86ec6e0ada1aa8f0aa5f3648643e0ef10b48743e0c3cd25cf8803b3a2ecad66', 'case/EVAL_ADMISSION.json': '38f3a6caaacbc08a69a0b758ab2e2d4c8bc93ea5c96cd9643980272d249ada80', 'case/EVAL_EVENTS.jsonl': '92e5c666db32e9cec1962023f0dc77e8838aa85ab2570aaed220b2a0c0a9aa7a', 'case/EVALUATION_STARTED.json': 'f0a733ff7449f0cc415053979bf8ae3c6b78578d69bb93cb884dcd20e64210cd'}, 'failed_events': [{'kind': 'environment_acquired', 'time': 1794458091.7617228, 'case': 'OBJECT_DEBR_41', 'purpose': 'formal', 'pid': 159797}, {'kind': 'evaluate_dispatch', 'time': 1794458091.7648602, 'case': 'OBJECT_DEBR_41', 'pid': 164955}, {'kind': 'environment_released', 'time': 1794458110.9969296, 'case': 'OBJECT_DEBR_41', 'purpose': 'formal', 'pid': 159797, 'success': False}, {'kind': 'evaluate_exit', 'time': 1794458110.997014, 'case': 'OBJECT_DEBR_41', 'pid': 164955, 'returncode': 1}], 'completed_hashes': {'OUTCOMES.json': '8d12778a4abf6758163d9c1842a4d868f6279caa7426c187a4309ac590d51f00', 'NORMALIZED_OUTCOMES.json': '56d5e530e253597b6b7649ffd21879853e5efd5e69bd37afbe43e279145a6e4f', 'EVAL_EVENTS.jsonl': '2db9de2d2c3b1c223ffdee46840bc727005708244b1d73efd313d085e8239e55', 'EVALUATION_COMPLETE.json': 'cb74da5c1a7eec06ab8d74c114596b7a5c301eb60ad2da000658f1472957b5da', 'evaluate.exit': '9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa'}})))
