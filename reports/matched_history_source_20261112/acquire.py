"""Run the shared GOAL seed41 collection exactly once, before either arm trains."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def write(name, **fields):
    path = ROOT / name
    temp = path.with_suffix('.tmp')
    fields['observed'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp.write_text(json.dumps(fields, indent=2) + '\n')
    os.replace(temp, path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    with (ROOT / 'ACQUISITION_DISPATCH.json').open('x') as stream:
        json.dump({'pid': os.getpid(), 'phase': 'common_collection'}, stream)
    try:
        contract = read('CONTRACT.json')
        teachers = read('CASE.json')['teachers']
        assert all(digest(path) == sha for path, sha in contract['files'].items())
        inherited = contract.get('inherited_task6')
        for task in (6, 7, 8):
            teacher = teachers[str(task)]
            assert digest(teacher['path']) == teacher['sha256']
            name = 'collect_%d' % task
            if task == 6 and inherited:
                assert digest(inherited['cache']) == inherited['sha256']
                assert (ROOT / (name + '.exit')).read_text() == '0'
            else:
                with (ROOT / (name + '.log')).open('x') as log:
                    child = subprocess.Popen(
                        [sys.executable, '-B', str(ROOT / 'collect.py'), 'S', str(task)],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1'),
                    )
                    write('ACQUISITION_STATUS.json', phase='common_collection', task=task,
                          pid=child.pid, controller=os.getpid())
                    rc = child.wait()
                (ROOT / (name + '.exit')).write_text(str(rc))
                assert rc == 0, (name, rc)
            outcomes = read('COLLECTION_OUTCOMES_S_%d.json' % task)
            assert [row['initial_index'] for row in outcomes] == list(range(20, 30))
            assert all(row['attempt'] == 1 and row['cp_sha256'] == teacher['sha256'] for row in outcomes)
            memory = read('MEMORY_S_%d.json' % task)
            assert memory['status'] == 'PASS_FROZEN_ACQUISITION_MEMORY'
            assert memory['training_attempts'] == 10 and memory['optimizer_updates'] == 0
            if task == 7 and inherited:
                retry = contract['authorized_task7_repeat']
                prior_path = Path(retry['original_outcomes'])
                assert digest(prior_path) == retry['sha256']
                original = json.loads(prior_path.read_text())
                identity = ('initial_index', 'initial_sha256', 'initial_file_sha256', 'noise_sha256', 'cp_sha256', 'env_seed')
                assert all(all(a[k] == b[k] for k in identity) for a, b in zip(original, outcomes))
                write('TASK7_REPEAT_RECEIPT.json', original_sha256=retry['sha256'],
                      replacement_sha256=digest(ROOT / 'COLLECTION_OUTCOMES_S_7.json'),
                      repeated_episodes=10, actual_attempt_ordinal=2,
                      identical_action_streams=sum(a['action_stream_sha256'] == b['action_stream_sha256']
                                                   for a, b in zip(original, outcomes)),
                      identical_success_and_steps=sum((a['success'], a['steps']) == (b['success'], b['steps'])
                                                      for a, b in zip(original, outcomes)))
        write('ACQUISITION_STATUS.json', phase='common_collection_complete',
              attempts=30, production_optimizer_calls=0, formal_episodes=0, controller=os.getpid())
        (ROOT / 'acquire.exit').write_text('0')
    except BaseException:
        write('ACQUISITION_FAILURE.json', traceback=traceback.format_exc(), controller=os.getpid())
        (ROOT / 'acquire.exit').write_text('1')
        raise


if __name__ == '__main__':
    main()
