"""Finite, evaluation-only queue launched after the reverse experiment."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import ENV_LOCK, PYTHON, ROOT, capacity, environment, processes, read, sha, verify_frozen, write


def run_case(case, operation):
    directory = ROOT / 'cases' / case
    directory.mkdir(parents=True, exist_ok=True)
    env = environment()
    env['ABLATION_EVAL_CASE'] = case
    log = directory / (operation + '.log')
    with log.open('x') as output:
        result = subprocess.run([PYTHON, '-B', str(ROOT / 'case.py'), operation], cwd=directory,
                                env=env, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError((case, operation, result.returncode, log.read_text(errors='replace')[-2200:]))
    if operation != 'evaluate':
        return json.loads(log.read_text().splitlines()[-1])
    return read(directory / 'EVALUATION_COMPLETE.json')


def audit_cases(operation):
    spec = verify_frozen()
    results = {case: run_case(case, operation) for case in spec['evaluation_cases']}
    expected = 'PASS_CPU_PREPARATION' if operation == 'prepare' else 'PASS_FINAL_CASE'
    assert len(results) == 12 and all(result['status'] == expected for result in results.values())
    return results


def predecessor_complete(spec, authorization):
    predecessor = Path(spec['predecessor'])
    assert (predecessor / 'controller.exit').read_text().strip() == '0' and not processes(predecessor)
    assert read(predecessor / 'STATUS.json')['status'] == 'EXECUTION_COMPLETE_PENDING_CPU_ACCEPTANCE'
    result_file = Path(authorization['remote_cpu_results'])
    assert result_file.parent.parent == predecessor and result_file.name == 'CPU_RESULTS.json'
    assert sha(result_file) == authorization['remote_cpu_results_sha256']
    final = read(result_file)
    assert final['status'] == 'PASS_REMOTE_REVERSE_CPU_PENDING_LOCAL_ACCEPTANCE' and len(final['cases']) == 12
    assert all(row['status'] == 'PASS_REVERSE_CPU_CASE' and row['formal'] == 200 for row in final['cases'].values())
    assert authorization['prepared_contract'] == sha(ROOT / 'CONTRACT.json')


def launch(authorization):
    spec = verify_frozen()
    predecessor_complete(spec, authorization)
    assert read(ROOT / 'PREPARATION.json')['status'] == 'READY_AWAITING_PREDECESSOR'
    assert not processes(ROOT) and not (ROOT / 'controller.exit').exists()
    capacity()
    with (ROOT / 'LAUNCH_AUTHORIZATION.json').open('x') as file:
        json.dump(authorization, file, indent=2)
    with (ROOT / 'controller.log').open('x') as output:
        process = subprocess.Popen([PYTHON, '-B', str(ROOT / 'job.py'), 'run'], cwd=ROOT, env=environment(),
                                   stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    receipt = dict(status='DISPATCHED_AFTER_REVERSE_ACCEPTANCE', pid=process.pid, observed=time.time(), authorization=authorization)
    write(ROOT / 'LAUNCH.json', receipt)
    return receipt


def run():
    spec = verify_frozen()
    predecessor_complete(spec, read(ROOT / 'LAUNCH_AUTHORIZATION.json'))
    with ENV_LOCK.open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for case in spec['evaluation_cases']:
            directory = ROOT / 'cases' / case
            assert not (directory / 'EVALUATION_STARTED.json').exists()
            write(ROOT / 'STATUS.json', dict(status='PARTIAL_EVALUATING', case=case, observed=time.time()))
            result = run_case(case, 'evaluate')
            assert result['episodes'] == 200
    write(ROOT / 'STATUS.json', dict(status='EXECUTION_COMPLETE_PENDING_CPU_ACCEPTANCE', observed=time.time()))


def snapshot():
    spec = verify_frozen()
    cases = {}
    for name in spec['evaluation_cases']:
        directory = ROOT / 'cases' / name
        row = {}
        for field in ('OUTCOMES', 'EVALUATION_COMPLETE', 'EVAL_ADMISSION'):
            path = directory / (field + '.json')
            if path.exists():
                row[field] = read(path)
        cases[name] = row
    state = ROOT / 'STATUS.json'
    code = ROOT / 'controller.exit'
    failure = ROOT / 'FAILURE.json'
    return dict(observed=time.time(), contract=sha(ROOT / 'CONTRACT.json'), status=read(state) if state.exists() else None,
                controller_exit=code.read_text().strip() if code.exists() else None, processes=processes(ROOT), cases=cases,
                failure=read(failure) if failure.exists() else None, capacity=capacity())


def final():
    assert (ROOT / 'controller.exit').read_text().strip() == '0' and not processes(ROOT)
    accepted = ROOT / 'CPU_RESULTS.json'
    if not accepted.exists():
        results = audit_cases('final')
        assert sum(row['episodes'] for row in results.values()) == 2400
        write(accepted, results)
    result = snapshot()
    result.update(cpu=read(accepted), outcomes_sha256={case: sha(ROOT / 'cases' / case / 'OUTCOMES.json') for case in result['cases']})
    return result


if __name__ == '__main__':
    operation = sys.argv[1]
    if operation == 'prepare':
        proof = dict(status='READY_AWAITING_PREDECESSOR', observed=time.time(), contract=sha(ROOT / 'CONTRACT.json'),
                     cpu=audit_cases('prepare'), capacity=capacity(), formal_episodes=0, new_optimizer=0)
        write(ROOT / 'PREPARATION.json', proof)
        print(json.dumps(proof))
    else:
        assert operation == 'run'
        code = 1
        try:
            run()
            code = 0
        except BaseException:
            write(ROOT / 'FAILURE.json', dict(observed=time.time(), traceback=traceback.format_exc()))
            raise
        finally:
            with (ROOT / 'controller.exit').open('x') as file:
                file.write(str(code) + '\n')
