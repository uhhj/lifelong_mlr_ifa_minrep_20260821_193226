"""Check original models on CPU or invoke their single formal evaluation."""
import collections
import sys

from runtime import CASE, NAME, R, SEED, EasyDict, js, np, sha, torch, verify
from coreset_exp.model import data_setup
from libero.lifelong.models.brief import BriefPolicy


def array_hash(array):
    import hashlib
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def initial_identities(benchmark, config):
    from pathlib import Path

    rows = {}
    for task_id in range(6, 10):
        task = benchmark.get_task(task_id)
        path = Path(config.init_states_folder) / task.problem_folder / task.init_states_file
        initial = torch.load(path, map_location='cpu')
        digest = sha(path)
        for index in range(20):
            rows[f'{task_id}_{index}'] = [digest, array_hash(np.asarray(initial[index]))]
    assert rows == CASE['initials']
    return rows


def cpu_audit(final):
    import json

    verify()
    assert not torch.cuda.is_initialized()
    config = EasyDict(CASE['eval_config'])
    config.device = 'cpu'
    config.training = False
    benchmark, _, shape = data_setup(config, True)
    config.shape_meta = shape
    initials = initial_identities(benchmark, config)
    checkpoints = {}
    for stage, meta in CASE['models'].items():
        policy = BriefPolicy(config, shape).eval().requires_grad_(False)
        raw = torch.load(meta['path'], map_location='cpu')['state_dict']
        state = {k[7:] if k.startswith('module.') else k: v for k, v in raw.items()}
        assert len(state) == len(policy.state_dict()) == 100
        policy.load_state_dict(state, strict=True)
        assert all(torch.equal(value, state[name]) for name, value in policy.state_dict().items())
        checkpoints[stage] = dict(sha256=meta['sha256'], keys=100, parameters=sum(p.numel() for p in policy.parameters()))
        del policy, raw, state
    evidence = dict(status='PASS_FINAL_CASE' if final else 'PASS_CPU_PREPARATION', case=NAME,
                    checkpoints=checkpoints, initial_bytes_verified=len(initials), new_optimizer=0, new_environment=0)
    if final:
        records = js(R / 'OUTCOMES.json')
        expected = {(stage, task, i) for stage in range(6, 10) for task in range(6, stage + 1) for i in range(20)}
        groups = collections.defaultdict(list)
        assert len(records) == len(expected) == 200
        for row in records:
            stage, task, i = row['stage'], row['task'], row['initial_index']
            expected.remove((stage, task, i))
            assert row['arm'] == 'S' and row['attempt'] == 1 and row['env_seed'] == SEED + i // 10
            assert row['cp_sha256'] == checkpoints[str(stage)]['sha256']
            assert initials[f'{task}_{i}'] == [row['initial_file_sha256'], row['initial_sha256']]
            noise = np.random.RandomState(20261024 + SEED * 1000 + task * 100 + i).standard_normal((600, 7)).astype(np.float32)
            assert row['noise_sha256'] == array_hash(noise)
            assert isinstance(row['success'], bool) and 1 <= row['steps'] <= 600
            assert (row['success'] and row['termination'] == 'success') or (not row['success'] and row['termination'] == 'timeout600' and row['steps'] == 600)
            groups[f'S_s{stage}_t{task}_g{i // 10}'].append(row)
        assert not expected and len(groups) == 20
        events = [json.loads(line) for line in (R / 'EVAL_EVENTS.jsonl').read_text().splitlines()]
        assert len(events) == 40 and len({event['run'] for event in events}) == 20
        for index in range(0, len(events), 2):
            start, stop = events[index:index + 2]
            name = start['run']
            assert start['kind'] == 'start' and stop['kind'] == 'complete' and stop['run'] == name
            assert stop['episodes'] == (index // 2 + 1) * 10 and start['time'] <= stop['time']
            assert js(R / 'outcomes' / (name + '.json')) == groups[name]
            assert all(row['cp_sha256'] == start['cp_sha256'] and row['initial_file_sha256'] == start['initial_file_sha256'] for row in groups[name])
        admission = js(R / 'EVAL_ADMISSION.json')
        assert admission['status'] == 'PASS' and admission['official_cp_sha256'] == CASE['official_sha256']
        assert [row['length'] for row in admission['cache_parity']] == list(range(1, 9))
        assert js(R / 'EVALUATION_COMPLETE.json')['episodes'] == 200
        evidence.update(episodes=200, vectors=20, outcomes_sha256=sha(R / 'OUTCOMES.json'))
    assert not torch.cuda.is_initialized()
    print(json.dumps(evidence))


if __name__ == '__main__':
    if sys.argv[1] == 'evaluate':
        import evaluate
        evaluate.main()
    else:
        assert sys.argv[1] in ('prepare', 'final')
        cpu_audit(sys.argv[1] == 'final')
