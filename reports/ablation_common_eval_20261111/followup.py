"""Run the prepared follow-up once the existing observer accepts its queue."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

from audit_results import audit, finalize
from common import ROOT, read, sha, write

PREDECESSOR = ROOT.parent / '20261110_reverse_order_main_protocol'
ZONE = datetime.timezone(datetime.timedelta(hours=8))
STATE = ROOT / 'RUN_STATUS.json'


def timestamp(epoch=None):
    return datetime.datetime.fromtimestamp(time.time() if epoch is None else epoch, ZONE).isoformat()


def state(status, **fields):
    write(STATE, dict(status=status, observed=timestamp(), observer_pid=os.getpid(), **fields))


def invoke(script, *arguments):
    env = os.environ.copy()
    settings = {'http.sslBackend': 'openssl', 'http.version': 'HTTP/1.1', 'http.sslVerify': 'true',
                'http.lowSpeedLimit': '1', 'http.lowSpeedTime': '20'}
    env['GIT_CONFIG_COUNT'] = str(len(settings))
    for i, (name, value) in enumerate(settings.items()):
        env[f'GIT_CONFIG_KEY_{i}'] = name
        env[f'GIT_CONFIG_VALUE_{i}'] = value
    result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / script), *map(str, arguments)],
                            cwd=ROOT.parents[1], env=env, capture_output=True, text=True, encoding='utf-8')
    if result.returncode:
        raise RuntimeError((script, result.returncode, result.stderr[-4500:], result.stdout[-1800:]))
    return result.stdout


def request(operation, *arguments):
    output = invoke('transport.py', operation, *arguments)
    lines = [Path(line.strip()) for line in output.splitlines() if line.strip() and not line.lstrip().startswith('{')]
    files = [line / 'EVIDENCE.json' for line in lines if line.is_absolute() and line.is_dir() and (line / 'EVIDENCE.json').exists()]
    assert len(files) == 1, output[:1000]
    return files[0]


def publish(message):
    invoke('publish.py', '--message', message)
    receipt = read(ROOT / 'PUBLICATION_RECEIPT.json')
    assert receipt['status'] == 'PUSH_VERIFIED'
    return receipt['commit']


def wait_for_predecessor(contract):
    while True:
        try:
            current = read(PREDECESSOR / 'CURRENT_TASK_WORKER.json')
        except json.JSONDecodeError:
            time.sleep(2)
            continue
        if current['status'] == 'COMPLETE_ACCEPTED_AND_PUBLISHED':
            break
        assert current['status'] != 'STOPPED_FOR_AGENT_ENGINEERING_REVIEW', 'Reverse queue observer needs engineering review'
        time.sleep(45)
    final_status = read(PREDECESSOR / 'RUN_STATUS.json')
    assert final_status['actual_final_cpu_acceptance']
    directory = Path(final_status['evidence_directory'])
    acceptance = directory / 'LOCAL_FINAL_AUDIT.json'
    accepted = read(acceptance)
    assert accepted['status'] == 'PASS_COMPLETE_REVERSE_LOCAL_ACCEPTANCE' and accepted['totals']['formal'] == 2400
    receipt = read(PREDECESSOR / 'PUBLICATION_RECEIPT.json')
    assert receipt['status'] == 'PUSH_VERIFIED' and receipt['commit'] == current['commit']
    names = read(PREDECESSOR / 'PUBLICATION_REVIEW.json')['files_sha256']
    assert names[acceptance.relative_to(PREDECESSOR).as_posix()]['sha256'] == sha(acceptance)
    remote = read(directory / 'EVIDENCE.json')
    payload = dict(status='PASS_REMOTE_REVERSE_CPU_PENDING_LOCAL_ACCEPTANCE', cases=remote['cpu'], new_optimizer=0, new_environments=0)
    digest = hashlib.sha256((json.dumps(payload, indent=2, allow_nan=False) + '\n').encode()).hexdigest()
    return dict(predecessor_final_audit_sha256=sha(acceptance), predecessor_publication=receipt['commit'],
                remote_cpu_results=remote['remote_cpu_directory'] + '/CPU_RESULTS.json', remote_cpu_results_sha256=digest,
                prepared_contract=contract)


def reference_seconds_per_episode(suite, spec):
    values = []
    for seed in (41, 42, 43):
        path, field = spec['comparison_outcomes'][f'{suite}_{seed}']['S_latest'].split('#light/', 1)
        light = read(ROOT.parents[1] / path)['light']
        entry = light.get(field.replace('OUTCOMES.json', 'EVAL_EVENTS.jsonl'))
        if entry:
            assert hashlib.sha256(entry['text'].encode()).hexdigest() == entry['sha256']
            events = [json.loads(line) for line in entry['text'].splitlines() if line]
            times = [event['seconds'] for event in events if event['kind'] == 'complete']
            if len(times) == 20:
                values.append(sum(times) / 200)
    return statistics.mean(values) if values else 8.0


def main():
    queued = read(ROOT / 'QUEUE_REGISTRATION.json')
    contract = queued['contract']
    assert sha(ROOT / queued['preparation_evidence']) == queued['preparation_sha256']
    spec = read(ROOT / 'SPEC.json')
    state('PARTIAL_QUEUED_WAITING_FOR_REVERSE_ACCEPTANCE', formal=0, contract=contract, predecessor=spec['predecessor'])
    authorization = wait_for_predecessor(contract)
    auth = ROOT / 'LAUNCH_AUTHORIZATION.json'
    with auth.open('x', encoding='utf-8') as file:
        json.dump(authorization, file, indent=2)
        file.write('\n')
    launch_file = request('launch', auth)
    launch = read(launch_file)
    assert launch['status'] == 'DISPATCHED_AFTER_REVERSE_ACCEPTANCE'
    state('PARTIAL_EVALUATING', formal=0, contract=contract, controller=launch['pid'], evidence=launch_file.relative_to(ROOT).as_posix())
    publish('Launch queued component ablation re-evaluation after reverse experiment acceptance')

    previous = None
    counts = {case: 0 for case in spec['evaluation_cases']}
    first = next(iter(counts))
    snapshot = dict(status=dict(case=first), observed=launch['observed'])
    while True:
        active = snapshot['status']['case']
        suite = spec['evaluation_cases'][active]['suite']
        remaining = sum(200 - done for case, done in counts.items() if spec['evaluation_cases'][case]['suite'] == suite)
        rate = reference_seconds_per_episode(suite, spec)
        next_time = time.time() + max(5 * 60, remaining * rate + 10 * 60)
        state('PARTIAL_EVALUATING', formal=sum(counts.values()), contract=contract, controller=launch['pid'],
              active=active, next_checkpoint_CST=timestamp(next_time))
        time.sleep(max(0, next_time - time.time()))
        evidence = request('snapshot')
        snapshot = read(evidence)
        counts = audit(snapshot, contract, previous)
        write(evidence.with_name('LOCAL_PROGRESS_AUDIT.json'), dict(status='PASS_PROGRESS_NOT_FINAL', observed=timestamp(snapshot['observed']),
              cases=counts, formal=sum(counts.values()), contract=contract, evidence_sha256=sha(evidence)))
        if snapshot['controller_exit'] == '0' and not snapshot['processes']:
            break
        previous = snapshot
        publish('Record audited progress for queued component ablation re-evaluation')

    evidence = request('final')
    finalize(evidence, contract)
    state('COMPLETE_ACCEPTED_PENDING_PUBLICATION', formal=2400, contract=contract, evidence=evidence.relative_to(ROOT).as_posix())
    commit = publish('Accept unified component ablation evaluation and publish recomputed results')
    state('COMPLETE_ACCEPTED_AND_PUBLISHED', formal=2400, contract=contract, commit=commit)


if __name__ == '__main__':
    import msvcrt

    with (ROOT / 'FOLLOWUP_WORKER.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            main()
        except BaseException:
            state('STOPPED_FOR_ENGINEERING_REVIEW', detail=traceback.format_exc())
            raise
