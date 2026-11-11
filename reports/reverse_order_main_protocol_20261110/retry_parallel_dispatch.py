"""Retry only pre-authentication failures, then verify the authorized dispatch."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'PARALLEL_DISPATCH_LOCAL_STATUS.json'
ZONE = datetime.timezone(datetime.timedelta(hours=8))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(status, **details):
    stamp = datetime.datetime.now(ZONE).isoformat()
    payload = dict(status=status, observed=stamp, retry_pid=os.getpid(), **details)
    temporary = STATE.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(STATE)


def invoke(script, *args):
    return subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / script), *map(str, args)],
                          cwd=ROOT.parents[1], capture_output=True, text=True, encoding='utf-8')


def main():
    while True:
        next_retry = datetime.datetime.now(ZONE) + datetime.timedelta(seconds=90)
        save('WAITING_FOR_SSH', next_retry_CST=next_retry.isoformat(), remote_mutation_confirmed=False)
        time.sleep(90)
        before = set(ROOT.glob('install_parallel_dispatch_*/ATTEMPT.json'))
        result = invoke('transport.py', 'install_parallel_dispatch.py')
        new = set(ROOT.glob('install_parallel_dispatch_*/ATTEMPT.json')) - before
        if len(new) != 1:
            save('DISPATCH_ATTEMPT_REQUIRES_REVIEW', exit=result.returncode)
            return
        attempt = next(iter(new))
        if result.returncode:
            details = read(attempt)
            if details.get('authenticated') is False and details.get('remote_commands') == 0:
                continue
            save('DISPATCH_ATTEMPT_REQUIRES_REVIEW', attempt=attempt.relative_to(ROOT).as_posix())
            return
        evidence = attempt.with_name('EVIDENCE.json')
        receipt = read(evidence)
        if receipt['status'] == 'ORIGINAL_QUEUE_HAS_LEFT_PARALLEL_TRAINING_WINDOW':
            save(receipt['status'], evidence=evidence.relative_to(ROOT).as_posix(), remote_mutation_confirmed=False)
            return
        if receipt['status'] != 'DISPATCHER_RELEASED' or receipt['failure'] is not None or receipt['first_GOAL_dispatch'] is None:
            save('DISPATCH_ATTEMPT_REQUIRES_REVIEW', evidence=evidence.relative_to(ROOT).as_posix())
            return
        save('GOAL_PARALLEL_TRAINING_DISPATCHED', evidence=evidence.relative_to(ROOT).as_posix(), controller=receipt['pid'],
             GOAL_MLR_41=receipt['first_GOAL_dispatch']['pid'], active=receipt['coordinator_status']['active'])
        return


if __name__ == '__main__':
    import msvcrt

    with (ROOT / 'PARALLEL_DISPATCH_RETRY.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        main()
