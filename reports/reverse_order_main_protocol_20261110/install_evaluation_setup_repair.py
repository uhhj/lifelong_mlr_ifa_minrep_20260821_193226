SOURCE='"""Restore omitted simulator assets and run the existing evaluation-only queue."""\nfrom pathlib import Path\nimport fcntl\nimport json\nimport os\nimport re\nimport subprocess\nimport sys\nimport time\nimport traceback\n\nROOT = Path(\'/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05\')\nsys.path.insert(0, str(ROOT))\nfrom controller import evaluate\nfrom reverse_common import PY, capacity, environment, event, processes, read, sha, verify, write\n\nARCHIVE = ROOT / \'FAILED_OBJECT_MLR_41_PRE_ROLLOUT\'\nRECEIPT = ROOT / \'EVALUATION_SETUP_REPAIR.json\'\nFAILED = \'OBJECT_MLR_41\'\nCONTRACT = \'167eb6d15fab4f6d9f7e4db396f826f546160edacb0e2b3049bfa60d5ee80216\'\nASSETS = Path(\'/data/mlr_ifa/src/libero/libero/assets\')\nSCENE = \'scenes/libero_floor_base_style.xml\'\nSCENE_SHA = \'df092a7d8ac7a5c4d1ce48e9a6dacf2f7ebaa362580e20de46d9644f3d5d1039\'\nROOT_FILES = (\'controller.exit\', \'FAILURE.json\', \'STATUS.json\')\nCASE_FILES = (\'evaluate.log\', \'evaluate.exit\', \'EVAL_DISPATCH.json\', \'EVAL_ADMISSION.json\', \'EVAL_EVENTS.jsonl\',\n              \'EVALUATION_STARTED.json\', \'EVALUATION_PROTOCOL_DECLARATION.json\')\n\n\ndef install(source, expected):\n    verify()\n    capacity()\n    assert sha(ROOT / \'CONTRACT.json\') == CONTRACT and not processes()\n    assert not any(p.exists() for p in (ARCHIVE, RECEIPT, ROOT / \'FINAL_CPU_REQUEST.json\'))\n    assert (ROOT / \'controller.exit\').read_text().strip() == \'1\'\n    spec = read(ROOT / \'SPEC.json\')[\'cases\']\n    assert len(spec) == 12 and sha(ASSETS / SCENE) == SCENE_SHA\n    links = {}\n    for name, row in spec.items():\n        case = ROOT / \'cases\' / name\n        inp = read(case / \'INPUT.json\')\n        mlr = row[\'method\'] == \'MLR\'\n        epoch_path = Path(inp[\'output_root\']) / \'epochs.json\' if mlr else case / \'EPOCHS_S.json\'\n        history = read(epoch_path)\n        assert (case / \'train.exit\').read_text().strip() == \'0\' and len(history) == 400\n        assert history[-1][\'steps\'] == inp[\'accepted_budget\']\n        calls = read(case / \'OPTIMIZER_CALLS.json\')\n        assert calls[\'status\'] == \'ACTUAL_CALLS_COMPLETED\' and calls[\'calls\'][\'production\'] == inp[\'physical_budget\']\n        assert not list((case / \'outcomes\').glob(\'*\'))\n        assert not any((case / p).exists() for p in (\'OUTCOMES.json\', \'NORMALIZED_OUTCOMES.json\', \'EVALUATION_COMPLETE.json\'))\n        if name != FAILED:\n            assert not any((case / p).exists() for p in CASE_FILES)\n        path = Path(inp[\'source_root\']) / \'libero/libero/assets\'\n        if mlr:\n            assert path == case / \'source/libero/libero/assets\' and path.parent.is_dir()\n            assert not path.exists() and not path.is_symlink()\n            links[name] = path\n            native = Path(inp[\'output_root\']) / \'evaluation\'\n            assert not any(native.glob(\'trials*.jsonl\')) and not any(native.glob(\'attempts*.jsonl\'))\n        else:\n            assert path.resolve() == ASSETS and sha(path / SCENE) == SCENE_SHA\n    assert len(links) == 6\n\n    failed_case = ROOT / \'cases\' / FAILED\n    failed = {\'root/\' + n: ROOT / n for n in ROOT_FILES}\n    failed.update({\'case/\' + n: failed_case / n for n in CASE_FILES})\n    assert {key: sha(path) for key, path in failed.items()} == expected[\'failed_hashes\']\n    assert (failed_case / \'evaluate.exit\').read_text().strip() == \'1\'\n    log = (failed_case / \'evaluate.log\').read_text(errors=\'replace\')\n    assert set(map(int, re.findall(r\'Process Process-(\\d+):\', log))) == set(range(1, 11))\n    assert log.count(\'FileNotFoundError: [Errno 2]\') == 10\n    assert log.count(str(links[FAILED] / SCENE)) == 10\n    assert \'return [w.seed(s) for w, s in zip(self.workers, seed_list)]\' in log\n    assert \'ob,_,_,_=env.step\' not in log\n    failed_vectors = [json.loads(line) for line in (failed_case / \'EVAL_EVENTS.jsonl\').read_text().splitlines()]\n    assert [(row[\'kind\'], row[\'run\']) for row in failed_vectors] == [(\'start\', \'MLR_s6_t6_g0\')]\n    events = [json.loads(line) for line in (ROOT / \'EVENTS.jsonl\').read_text().splitlines()]\n    assert events[-4:] == expected[\'last_events\']\n\n    entry = ROOT / \'evaluation_setup_repair.py\'\n    linked, moved = [], []\n    started = False\n    try:\n        with entry.open(\'x\') as file:\n            file.write(source)\n        for path in links.values():\n            path.symlink_to(ASSETS, target_is_directory=True)\n            linked.append(path)\n            assert sha(path / SCENE) == SCENE_SHA\n        verify()\n        ARCHIVE.mkdir()\n        for key, path in failed.items():\n            target = ARCHIVE / key\n            target.parent.mkdir(exist_ok=True)\n            path.rename(target)\n            moved.append((target, path))\n            assert sha(target) == expected[\'failed_hashes\'][key]\n        proof = dict(status=\'PRE_ROLLOUT_ASSETS_RESTORED\', observed=time.time(), contract=CONTRACT,\n                     case=FAILED, failed_vectors=failed_vectors, failed_archived_hashes=expected[\'failed_hashes\'],\n                     failed_events=expected[\'last_events\'], worker_constructor_errors=10, resets=0, steps=0,\n                     formal_outcomes=0, new_training_updates=0, acquisition_repeated=0,\n                     archive=str(ARCHIVE), common_assets=str(ASSETS), scene_sha256=SCENE_SHA,\n                     linked_cases=sorted(links), source_sha256=sha(entry))\n        write(RECEIPT, proof)\n        with (ROOT / \'evaluation_setup_repair.log\').open(\'x\') as log:\n            worker = subprocess.Popen([PY, \'-B\', str(entry), \'evaluate\'], cwd=ROOT, env=environment(),\n                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\n        started = True\n        return dict(proof, controller=worker.pid, receipt_sha256=sha(RECEIPT))\n    finally:\n        if not started:\n            for target, path in reversed(moved):\n                target.rename(path)\n            for path in reversed(linked):\n                path.unlink()\n\n\ndef run():\n    with (ROOT / \'evaluation_setup_repair.lock\').open(\'a+\') as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n        code = 1\n        try:\n            proof = read(RECEIPT)\n            assert sha(ROOT / \'CONTRACT.json\') == CONTRACT == proof[\'contract\']\n            assert sha(__file__) == proof[\'source_sha256\']\n            for key, digest in proof[\'failed_archived_hashes\'].items():\n                assert sha(ARCHIVE / key) == digest\n            assert not (ROOT / \'controller.exit\').exists()\n            event(\'parallel_dispatch_started\', pid=os.getpid(), evaluation_setup_repair_sha256=sha(RECEIPT))\n            spec = read(ROOT / \'SPEC.json\')[\'cases\']\n            for suite in (\'OBJECT\', \'GOAL\'):\n                for case in spec:\n                    if spec[case][\'suite\'] == suite:\n                        verify()\n                        capacity()\n                        evaluate(case)\n                write(ROOT / (suite + \'_COMPLETE.json\'), dict(status=\'PASS_EXECUTION_PENDING_FINAL\', formal=1200, training_runs=6, time=time.time()))\n            event(\'controller_complete\', pid=os.getpid())\n            write(ROOT / \'STATUS.json\', dict(status=\'EXECUTION_COMPLETE_PENDING_CPU_ACCEPTANCE\', time=time.time(), controller=os.getpid()))\n            code = 0\n        except BaseException:\n            write(ROOT / \'FAILURE.json\', dict(time=time.time(), traceback=traceback.format_exc()))\n            raise\n        finally:\n            with (ROOT / \'controller.exit\').open(\'x\') as file:\n                file.write(str(code) + \'\\n\')\n\n\nif __name__ == \'__main__\' and sys.argv[1:] == [\'evaluate\']:\n    run()\n'
"""Restore omitted simulator assets and run the existing evaluation-only queue."""
from pathlib import Path
import fcntl
import json
import os
import re
import subprocess
import sys
import time
import traceback

ROOT = Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
sys.path.insert(0, str(ROOT))
from controller import evaluate
from reverse_common import PY, capacity, environment, event, processes, read, sha, verify, write

ARCHIVE = ROOT / 'FAILED_OBJECT_MLR_41_PRE_ROLLOUT'
RECEIPT = ROOT / 'EVALUATION_SETUP_REPAIR.json'
FAILED = 'OBJECT_MLR_41'
CONTRACT = '167eb6d15fab4f6d9f7e4db396f826f546160edacb0e2b3049bfa60d5ee80216'
ASSETS = Path('/data/mlr_ifa/src/libero/libero/assets')
SCENE = 'scenes/libero_floor_base_style.xml'
SCENE_SHA = 'df092a7d8ac7a5c4d1ce48e9a6dacf2f7ebaa362580e20de46d9644f3d5d1039'
ROOT_FILES = ('controller.exit', 'FAILURE.json', 'STATUS.json')
CASE_FILES = ('evaluate.log', 'evaluate.exit', 'EVAL_DISPATCH.json', 'EVAL_ADMISSION.json', 'EVAL_EVENTS.jsonl',
              'EVALUATION_STARTED.json', 'EVALUATION_PROTOCOL_DECLARATION.json')


def install(source, expected):
    verify()
    capacity()
    assert sha(ROOT / 'CONTRACT.json') == CONTRACT and not processes()
    assert not any(p.exists() for p in (ARCHIVE, RECEIPT, ROOT / 'FINAL_CPU_REQUEST.json'))
    assert (ROOT / 'controller.exit').read_text().strip() == '1'
    spec = read(ROOT / 'SPEC.json')['cases']
    assert len(spec) == 12 and sha(ASSETS / SCENE) == SCENE_SHA
    links = {}
    for name, row in spec.items():
        case = ROOT / 'cases' / name
        inp = read(case / 'INPUT.json')
        mlr = row['method'] == 'MLR'
        epoch_path = Path(inp['output_root']) / 'epochs.json' if mlr else case / 'EPOCHS_S.json'
        history = read(epoch_path)
        assert (case / 'train.exit').read_text().strip() == '0' and len(history) == 400
        assert history[-1]['steps'] == inp['accepted_budget']
        calls = read(case / 'OPTIMIZER_CALLS.json')
        assert calls['status'] == 'ACTUAL_CALLS_COMPLETED' and calls['calls']['production'] == inp['physical_budget']
        assert not list((case / 'outcomes').glob('*'))
        assert not any((case / p).exists() for p in ('OUTCOMES.json', 'NORMALIZED_OUTCOMES.json', 'EVALUATION_COMPLETE.json'))
        if name != FAILED:
            assert not any((case / p).exists() for p in CASE_FILES)
        path = Path(inp['source_root']) / 'libero/libero/assets'
        if mlr:
            assert path == case / 'source/libero/libero/assets' and path.parent.is_dir()
            assert not path.exists() and not path.is_symlink()
            links[name] = path
            native = Path(inp['output_root']) / 'evaluation'
            assert not any(native.glob('trials*.jsonl')) and not any(native.glob('attempts*.jsonl'))
        else:
            assert path.resolve() == ASSETS and sha(path / SCENE) == SCENE_SHA
    assert len(links) == 6

    failed_case = ROOT / 'cases' / FAILED
    failed = {'root/' + n: ROOT / n for n in ROOT_FILES}
    failed.update({'case/' + n: failed_case / n for n in CASE_FILES})
    assert {key: sha(path) for key, path in failed.items()} == expected['failed_hashes']
    assert (failed_case / 'evaluate.exit').read_text().strip() == '1'
    log = (failed_case / 'evaluate.log').read_text(errors='replace')
    assert set(map(int, re.findall(r'Process Process-(\d+):', log))) == set(range(1, 11))
    assert log.count('FileNotFoundError: [Errno 2]') == 10
    assert log.count(str(links[FAILED] / SCENE)) == 10
    assert 'return [w.seed(s) for w, s in zip(self.workers, seed_list)]' in log
    assert 'ob,_,_,_=env.step' not in log
    failed_vectors = [json.loads(line) for line in (failed_case / 'EVAL_EVENTS.jsonl').read_text().splitlines()]
    assert [(row['kind'], row['run']) for row in failed_vectors] == [('start', 'MLR_s6_t6_g0')]
    events = [json.loads(line) for line in (ROOT / 'EVENTS.jsonl').read_text().splitlines()]
    assert events[-4:] == expected['last_events']

    entry = ROOT / 'evaluation_setup_repair.py'
    linked, moved = [], []
    started = False
    try:
        with entry.open('x') as file:
            file.write(source)
        for path in links.values():
            path.symlink_to(ASSETS, target_is_directory=True)
            linked.append(path)
            assert sha(path / SCENE) == SCENE_SHA
        verify()
        ARCHIVE.mkdir()
        for key, path in failed.items():
            target = ARCHIVE / key
            target.parent.mkdir(exist_ok=True)
            path.rename(target)
            moved.append((target, path))
            assert sha(target) == expected['failed_hashes'][key]
        proof = dict(status='PRE_ROLLOUT_ASSETS_RESTORED', observed=time.time(), contract=CONTRACT,
                     case=FAILED, failed_vectors=failed_vectors, failed_archived_hashes=expected['failed_hashes'],
                     failed_events=expected['last_events'], worker_constructor_errors=10, resets=0, steps=0,
                     formal_outcomes=0, new_training_updates=0, acquisition_repeated=0,
                     archive=str(ARCHIVE), common_assets=str(ASSETS), scene_sha256=SCENE_SHA,
                     linked_cases=sorted(links), source_sha256=sha(entry))
        write(RECEIPT, proof)
        with (ROOT / 'evaluation_setup_repair.log').open('x') as log:
            worker = subprocess.Popen([PY, '-B', str(entry), 'evaluate'], cwd=ROOT, env=environment(),
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = True
        return dict(proof, controller=worker.pid, receipt_sha256=sha(RECEIPT))
    finally:
        if not started:
            for target, path in reversed(moved):
                target.rename(path)
            for path in reversed(linked):
                path.unlink()


def run():
    with (ROOT / 'evaluation_setup_repair.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = 1
        try:
            proof = read(RECEIPT)
            assert sha(ROOT / 'CONTRACT.json') == CONTRACT == proof['contract']
            assert sha(__file__) == proof['source_sha256']
            for key, digest in proof['failed_archived_hashes'].items():
                assert sha(ARCHIVE / key) == digest
            assert not (ROOT / 'controller.exit').exists()
            event('parallel_dispatch_started', pid=os.getpid(), evaluation_setup_repair_sha256=sha(RECEIPT))
            spec = read(ROOT / 'SPEC.json')['cases']
            for suite in ('OBJECT', 'GOAL'):
                for case in spec:
                    if spec[case]['suite'] == suite:
                        verify()
                        capacity()
                        evaluate(case)
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


if __name__ == '__main__' and sys.argv[1:] == ['evaluate']:
    run()

print(json.dumps(install(SOURCE, {'failed_hashes': {'root/controller.exit': '4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865', 'root/FAILURE.json': '0774ec12b654ca3174e1f1eaf3b4f15de6168c96b68f444489e284298935f8a4', 'root/STATUS.json': 'f59d21df33c67d020b562879d2a6a0fadedca64c1609c578a5b6e279377ec8a3', 'case/evaluate.log': '4ad2800c5d084724f1249a4575eed5908680a3555706a1ff3097965e9bc48645', 'case/evaluate.exit': '4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865', 'case/EVAL_DISPATCH.json': '5deb7fe40b5adf6c47313df9ec2345ee54e1732f5e66dadde68ba48ce17af1ae', 'case/EVAL_ADMISSION.json': 'af87ea484f8f74ca7ed1b27ab28601facaf0280bd365a6ae24373b7cfc4085be', 'case/EVAL_EVENTS.jsonl': '2663a6601f80ea0228410ea8e3fe4cbb6fbc9139acffdbcf68fa05bae6e89ba7', 'case/EVALUATION_STARTED.json': '6bdf2a0e3f93e88877c4378a1f904de6785b42ec51615f69f915a8a4c98a4a0d', 'case/EVALUATION_PROTOCOL_DECLARATION.json': '0ff6295fcb63f99d6e2a2d529590a12447866f2ae3c56c5bc1abae3d207cd7f0'}, 'last_events': [{'kind': 'environment_acquired', 'time': 1794433838.6304035, 'case': 'OBJECT_MLR_41', 'purpose': 'formal', 'pid': 152917}, {'kind': 'evaluate_dispatch', 'time': 1794433838.6340268, 'case': 'OBJECT_MLR_41', 'pid': 158673}, {'kind': 'environment_released', 'time': 1794433858.2567139, 'case': 'OBJECT_MLR_41', 'purpose': 'formal', 'pid': 152917, 'success': False}, {'kind': 'evaluate_exit', 'time': 1794433858.256784, 'case': 'OBJECT_MLR_41', 'pid': 158673, 'returncode': 1}]})))
