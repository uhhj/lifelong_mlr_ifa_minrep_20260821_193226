"""Dispatch six acquisition/paired training/evaluation sequences in order."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
CASES = [suite + '_' + str(seed) for suite in ('GOAL', 'OBJECT') for seed in (41, 42, 43)]


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.sha256(stream.read()).hexdigest()


def write(name, **fields):
    fields.update(controller=os.getpid(), observed=time.time())
    path = ROOT / name
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(fields, indent=2) + '\n')
    os.replace(temp, path)


def run(case, path, phase, args, distributed=False):
    launcher = ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=1'] if distributed else []
    with (path / (phase + '.log')).open('x') as log:
        child = subprocess.Popen([sys.executable, '-B', *launcher, *args], cwd=path, stdout=log,
            stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1'))
        write('STATUS.json', case=case, phase=phase, pid=child.pid, directory=str(path))
        rc = child.wait()
    (path / (phase + '.exit')).write_text(str(rc))
    assert rc == 0, (case, phase, rc)


def main():
    with (ROOT / 'STARTED.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), time=time.time()), stream)
    try:
        contract = json.loads((ROOT / 'PAIR_CONTRACT.json').read_text())
        assert all(digest(path) == h for path, h in contract['files'].items())
        for case in CASES:
            if case in contract.get('inherited_completed_cases', []):
                continue
            acquisition = ROOT / case
            pair = ROOT / 'pairs' / case
            if case == 'GOAL_41':
                assert (acquisition / 'acquire.exit').read_text() == '0'
            else:
                run(case, acquisition, 'dispatch_acquisition', ['acquire.py'])
            assert (acquisition / 'acquire.exit').read_text() == '0'
            run(case, pair, 'match', ['match_histories.py'])
            for arm in ('E', 'S'):
                run(case, pair, 'initial_' + arm, ['train_pair.py', arm, 'initial'], distributed=True)
            for arm in ('E', 'S'):
                run(case, pair, 'train_' + arm, ['train_pair.py', arm, 'train'], distributed=True)
            run(case, pair, 'evaluate', ['evaluate.py'])
        write('STATUS.json', phase='complete_pending_final_audit')
        (ROOT / 'controller.exit').write_text('0')
    except BaseException:
        write('FAILURE.json', traceback=traceback.format_exc())
        (ROOT / 'controller.exit').write_text('1')
        raise


if __name__ == '__main__':
    main()
