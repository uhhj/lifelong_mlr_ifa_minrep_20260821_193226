SOURCE='"""Dispatch the remaining evaluation cases after the pre-dispatch metadata error."""\nfrom pathlib import Path\nfrom types import SimpleNamespace\nimport fcntl\nimport json\nimport os\nimport subprocess\nimport sys\nimport time\nimport traceback\n\nROOT = Path(\'/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05\')\nsys.path.insert(0, str(ROOT))\nimport controller\nimport evaluation_spawn_repair as guard\nfrom reverse_common import PY, capacity, environment, event, processes, read, sha, verify, write\n\nARCHIVE = ROOT / \'FAILED_EVALUATION_DISPATCH_METADATA\'\nRECEIPT = ROOT / \'EVALUATION_DISPATCH_REPAIR.json\'\n\n\ndef plan():\n    contract = verify()\n    assert sha(ROOT / \'CONTRACT.json\') == guard.CONTRACT\n    spec = read(ROOT / \'SPEC.json\')[\'cases\']\n    assert list(spec) == list(contract[\'cases\'])\n    assert all(name == \'%s_%s_%s\' % (row[\'suite\'], row[\'method\'], row[\'seed\']) for name, row in spec.items())\n    proof = read(guard.RECEIPT)\n    assert sha(guard.__file__) == proof[\'source_sha256\']\n    for key, digest in proof[\'failed_archived_hashes\'].items():\n        assert sha(guard.ARCHIVE / key) == digest\n    guard.check_prefix(spec, proof[\'completed_hashes\'], failed_expected=False)\n    pending = {suite: [name for name, row in spec.items() if row[\'suite\'] == suite and name != guard.COMPLETE]\n               for suite in (\'OBJECT\', \'GOAL\')}\n    assert len(pending[\'OBJECT\']) == 5 and len(pending[\'GOAL\']) == 6\n    assert pending[\'OBJECT\'][0] == guard.FAILED\n    return pending\n\n\ndef install(source, expected):\n    pending = plan()\n    capacity()\n    assert not processes() and not ARCHIVE.exists() and not RECEIPT.exists()\n    assert not any((ROOT / p).exists() for p in (\'STATUS.json\', \'FINAL_CPU_REQUEST.json\'))\n    paths = {name: ROOT / name for name in (\'controller.exit\', \'FAILURE.json\')}\n    assert {name: sha(path) for name, path in paths.items()} == expected[\'hashes\']\n    assert (ROOT / \'controller.exit\').read_text().strip() == \'1\'\n    assert "KeyError: \'suite\'" in read(ROOT / \'FAILURE.json\')[\'traceback\']\n    tail = json.loads((ROOT / \'EVENTS.jsonl\').read_text().splitlines()[-1])\n    assert tail == expected[\'last_event\'] and tail[\'evaluation_spawn_repair_sha256\'] == sha(guard.RECEIPT)\n    entry = ROOT / \'evaluation_spawn_dispatch_v2.py\'\n    with entry.open(\'x\') as f:\n        f.write(source)\n    checked = subprocess.run([PY, \'-B\', str(entry), \'plan\'], cwd=ROOT, env=environment(), check=True,\n                             capture_output=True, text=True, timeout=45)\n    assert json.loads(checked.stdout) == pending\n    ARCHIVE.mkdir()\n    moved = []\n    started = False\n    try:\n        for name, path in paths.items():\n            target = ARCHIVE / name\n            path.rename(target)\n            moved.append((target, path))\n            assert sha(target) == expected[\'hashes\'][name]\n        proof = dict(status=\'PRE_DISPATCH_METADATA_REPAIRED\', observed=time.time(), contract=guard.CONTRACT,\n                     pending=pending, failed_archived_hashes=expected[\'hashes\'], failed_dispatcher=tail,\n                     spawn_repair_sha256=sha(guard.RECEIPT), source_sha256=sha(entry),\n                     failed_launches=0, new_training_updates=0, new_environments=0, completed_formal_preserved=200)\n        write(RECEIPT, proof)\n        with (ROOT / \'evaluation_spawn_dispatch_v2.log\').open(\'x\') as log:\n            worker = subprocess.Popen([PY, \'-B\', str(entry), \'queue\'], cwd=ROOT, env=environment(),\n                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)\n        started = True\n        return dict(proof, controller=worker.pid, receipt_sha256=sha(RECEIPT))\n    finally:\n        if not started:\n            for target, path in reversed(moved):\n                target.rename(path)\n\n\ndef run_queue():\n    def guarded_popen(command, **kwargs):\n        assert command[:3] == [PY, \'-B\', str(ROOT / \'run_case.py\')] and command[4:] == [\'evaluate\']\n        if \'_DEBR_\' in command[3]:\n            command[2:] = [str(Path(guard.__file__).resolve()), \'case\', command[3]]\n        return subprocess.Popen(command, **kwargs)\n\n    controller.subprocess = SimpleNamespace(Popen=guarded_popen, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT)\n    with (ROOT / \'evaluation_spawn_dispatch_v2.lock\').open(\'a+\') as lock:\n        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)\n        code = 1\n        try:\n            proof = read(RECEIPT)\n            pending = plan()\n            assert pending == proof[\'pending\'] and sha(__file__) == proof[\'source_sha256\']\n            assert sha(guard.RECEIPT) == proof[\'spawn_repair_sha256\']\n            assert not (ROOT / \'controller.exit\').exists()\n            event(\'parallel_dispatch_started\', pid=os.getpid(), evaluation_dispatch_repair_sha256=sha(RECEIPT))\n            for suite, cases in pending.items():\n                for case in cases:\n                    verify()\n                    capacity()\n                    controller.evaluate(case)\n                write(ROOT / (suite + \'_COMPLETE.json\'), dict(status=\'PASS_EXECUTION_PENDING_FINAL\', formal=1200, training_runs=6, time=time.time()))\n            event(\'controller_complete\', pid=os.getpid())\n            write(ROOT / \'STATUS.json\', dict(status=\'EXECUTION_COMPLETE_PENDING_CPU_ACCEPTANCE\', time=time.time(), controller=os.getpid()))\n            code = 0\n        except BaseException:\n            write(ROOT / \'FAILURE.json\', dict(time=time.time(), traceback=traceback.format_exc()))\n            raise\n        finally:\n            with (ROOT / \'controller.exit\').open(\'x\') as file:\n                file.write(str(code) + \'\\n\')\n\n\nif __name__ == \'__main__\' and sys.argv[1:] == [\'plan\']:\n    print(json.dumps(plan()))\nelif __name__ == \'__main__\' and sys.argv[1:] == [\'queue\']:\n    run_queue()\n'
"""Dispatch the remaining evaluation cases after the pre-dispatch metadata error."""
from pathlib import Path
from types import SimpleNamespace
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback

ROOT = Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
sys.path.insert(0, str(ROOT))
import controller
import evaluation_spawn_repair as guard
from reverse_common import PY, capacity, environment, event, processes, read, sha, verify, write

ARCHIVE = ROOT / 'FAILED_EVALUATION_DISPATCH_METADATA'
RECEIPT = ROOT / 'EVALUATION_DISPATCH_REPAIR.json'


def plan():
    contract = verify()
    assert sha(ROOT / 'CONTRACT.json') == guard.CONTRACT
    spec = read(ROOT / 'SPEC.json')['cases']
    assert list(spec) == list(contract['cases'])
    assert all(name == '%s_%s_%s' % (row['suite'], row['method'], row['seed']) for name, row in spec.items())
    proof = read(guard.RECEIPT)
    assert sha(guard.__file__) == proof['source_sha256']
    for key, digest in proof['failed_archived_hashes'].items():
        assert sha(guard.ARCHIVE / key) == digest
    guard.check_prefix(spec, proof['completed_hashes'], failed_expected=False)
    pending = {suite: [name for name, row in spec.items() if row['suite'] == suite and name != guard.COMPLETE]
               for suite in ('OBJECT', 'GOAL')}
    assert len(pending['OBJECT']) == 5 and len(pending['GOAL']) == 6
    assert pending['OBJECT'][0] == guard.FAILED
    return pending


def install(source, expected):
    pending = plan()
    capacity()
    assert not processes() and not ARCHIVE.exists() and not RECEIPT.exists()
    assert not any((ROOT / p).exists() for p in ('STATUS.json', 'FINAL_CPU_REQUEST.json'))
    paths = {name: ROOT / name for name in ('controller.exit', 'FAILURE.json')}
    assert {name: sha(path) for name, path in paths.items()} == expected['hashes']
    assert (ROOT / 'controller.exit').read_text().strip() == '1'
    assert "KeyError: 'suite'" in read(ROOT / 'FAILURE.json')['traceback']
    tail = json.loads((ROOT / 'EVENTS.jsonl').read_text().splitlines()[-1])
    assert tail == expected['last_event'] and tail['evaluation_spawn_repair_sha256'] == sha(guard.RECEIPT)
    entry = ROOT / 'evaluation_spawn_dispatch_v2.py'
    with entry.open('x') as f:
        f.write(source)
    checked = subprocess.run([PY, '-B', str(entry), 'plan'], cwd=ROOT, env=environment(), check=True,
                             capture_output=True, text=True, timeout=45)
    assert json.loads(checked.stdout) == pending
    ARCHIVE.mkdir()
    moved = []
    started = False
    try:
        for name, path in paths.items():
            target = ARCHIVE / name
            path.rename(target)
            moved.append((target, path))
            assert sha(target) == expected['hashes'][name]
        proof = dict(status='PRE_DISPATCH_METADATA_REPAIRED', observed=time.time(), contract=guard.CONTRACT,
                     pending=pending, failed_archived_hashes=expected['hashes'], failed_dispatcher=tail,
                     spawn_repair_sha256=sha(guard.RECEIPT), source_sha256=sha(entry),
                     failed_launches=0, new_training_updates=0, new_environments=0, completed_formal_preserved=200)
        write(RECEIPT, proof)
        with (ROOT / 'evaluation_spawn_dispatch_v2.log').open('x') as log:
            worker = subprocess.Popen([PY, '-B', str(entry), 'queue'], cwd=ROOT, env=environment(),
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = True
        return dict(proof, controller=worker.pid, receipt_sha256=sha(RECEIPT))
    finally:
        if not started:
            for target, path in reversed(moved):
                target.rename(path)


def run_queue():
    def guarded_popen(command, **kwargs):
        assert command[:3] == [PY, '-B', str(ROOT / 'run_case.py')] and command[4:] == ['evaluate']
        if '_DEBR_' in command[3]:
            command[2:] = [str(Path(guard.__file__).resolve()), 'case', command[3]]
        return subprocess.Popen(command, **kwargs)

    controller.subprocess = SimpleNamespace(Popen=guarded_popen, DEVNULL=subprocess.DEVNULL, STDOUT=subprocess.STDOUT)
    with (ROOT / 'evaluation_spawn_dispatch_v2.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        code = 1
        try:
            proof = read(RECEIPT)
            pending = plan()
            assert pending == proof['pending'] and sha(__file__) == proof['source_sha256']
            assert sha(guard.RECEIPT) == proof['spawn_repair_sha256']
            assert not (ROOT / 'controller.exit').exists()
            event('parallel_dispatch_started', pid=os.getpid(), evaluation_dispatch_repair_sha256=sha(RECEIPT))
            for suite, cases in pending.items():
                for case in cases:
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


if __name__ == '__main__' and sys.argv[1:] == ['plan']:
    print(json.dumps(plan()))
elif __name__ == '__main__' and sys.argv[1:] == ['queue']:
    run_queue()

print(json.dumps(install(SOURCE, {'hashes': {'controller.exit': '4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865', 'FAILURE.json': '8ec275d97ff6287c0b2faf36ec48611c565771ca31a89b7aaf4c1a1ea84d47d8'}, 'last_event': {'kind': 'parallel_dispatch_started', 'time': 1794459140.5200515, 'pid': 165201, 'evaluation_spawn_repair_sha256': 'dff4e921a1667330954a5e3c08c9eb179ba966912b6ad635da93ea94ffc67cbb'}})))
