"""Local metadata/attempt accounting only; no partial performance or model execution."""
import json
import sys
from pathlib import Path

P = Path(sys.argv[1])
arm = sys.argv[2]
def js(p):
    return json.loads(p.read_text(encoding='utf8'))
def lines(p):
    return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines()]
d = js(P/'EVIDENCE.json')
audit = js(P/'LOCAL_AUDIT.json')
assert audit['status'] == 'PASS_PROGRESS_NOT_FINAL' and not d['hash_bad']
root = P/'metadata'/arm
training = js(root/'TRAINING_COMPLETE.json')
assert training['epochs'] == 400 and training['steps'] == 105500
assert audit['arms'][arm] == dict(complete_epochs=400, main_updates=105500)
diagnostics = {}
for task in range(6, 10):
    complete = js(root/'TASK_COMPLETE'/('task%d.json'%task))
    diag = js(root/'diagnostics'/('task%d.json'%task))
    rows = js(root/'dataset'/('task%d.json'%task))['rows']
    assert complete['checkpoint_sha256'] == diag['checkpoint_sha256'] == training['checkpoints'][str(task)]
    assert diag['codec_sha256'] == training['codec_sha256']
    assert len(rows) == len(diag['rows']) == diag['windows']
    assert diag['environment_steps'] == diag['optimizer_steps'] == 0
    for a, b in zip(rows, diag['rows']):
        assert all(a[k] == b[k] for k in ('task','sample','demo','u'))
        assert a['sha256'] == b['sample_sha256']
        assert b['metrics']['predicted_action']['count'] == sum(a['action_mask'])
    diagnostics[str(task)] = dict(rows=len(rows), checkpoint_sha256=diag['checkpoint_sha256'])
loads = js(root/'CHECKPOINT_LOAD_AUDIT.json')
count = starts_count = full_cells = 0
durations = []
ongoing = []
cells = []
for path in sorted((root/'evaluation').glob('attempts_*.jsonl')):
    events = lines(path)
    trials = lines(path.with_name(path.name.replace('attempts_', 'trials_')))
    assert events
    stage, task = events[0]['stage'], events[0]['task']
    assert 6 <= task <= stage <= 9
    load = [x for x in loads if x['stage'] == stage and x['task'] == task]
    assert len(load) == 1 and load[0]['single_history'] and load[0]['executed_future_steps'] == 1
    assert load[0]['policy_sha256'] == training['checkpoints'][str(stage)]
    assert not any(x['event'] == 'technical_exception' for x in events)
    starts = {x['group']: x for x in events if x['event'] == 'vector_attempt_start'}
    ends = {x['group']: x for x in events if x['event'] == 'vector_attempt_complete'}
    assert len(starts) == sum(x['event'] == 'vector_attempt_start' for x in events)
    assert len(ends) == sum(x['event'] == 'vector_attempt_complete' for x in events)
    assert set(ends) <= set(starts) <= {0,1}
    assert len(trials) == 10 * len(ends)
    assert len({x['initial_state_index'] for x in trials}) == len(trials)
    for x in events + trials:
        assert x['arm'] == arm and x['stage'] == stage and x['task'] == task
        assert x['contract_sha256'] == d['contract_sha256'] and x['epoch'] == 99
        assert x['checkpoint_sha256'] == training['checkpoints'][str(stage)]
        assert x['codec_sha256'] == training['codec_sha256']
        assert x['loaded_downstream_keys'] == load[0]['strict_keys']
    for group, start in starts.items():
        assert start['env_seed'] == 41+group
        assert start['initial_indices'] == list(range(10*group,10*group+10))
        if group not in ends:
            ongoing.append(dict(stage=stage, task=task, group=group, elapsed_seconds_at_snapshot=d['unix']-start['unix']))
            continue
        end = ends[group]
        assert end['env_seed'] == 41+group and end['lanes'] == 10 and 1 <= end['steps'] <= 600
        batch = [x for x in trials if 10*group <= x['initial_state_index'] < 10*group+10]
        assert [x['initial_state_index'] for x in batch] == start['initial_indices']
        assert [x['success'] for x in batch] == end['successes']
        for x in batch:
            assert x['seed'] == 41 and x['vector_group_steps'] == end['steps']
            assert x['initial_state_sha256'] == start['initial_hashes'][x['initial_state_index']]
            assert x['initial_file_sha256'] == start['initial_file_sha256']
        durations.append(end['unix']-start['unix'])
    count += len(trials)
    starts_count += len(starts)
    full_cells += len(ends) == 2 and sum(x['event'] == 'environment_closed' for x in events) == 1
    cells.append(dict(stage=stage, task=task, complete_episodes=len(trials)))
out = dict(status='PASS_PROGRESS_NOT_FINAL', arm=arm, observation_unix=d['unix'],
           complete_epochs=400, main_updates=105500, diagnostics=diagnostics,
           complete_episodes=count, started_vectors=starts_count, complete_vectors=len(durations),
           closed_complete_cells=full_cells, ongoing_vectors=ongoing, vector_seconds=durations,
           cells=cells, technical_exceptions=0, no_partial_performance_metrics=True,
           initial_hash_scope='Internal episode/attempt binding; independent initial-file bytes pending final audit',
           checkpoint_scope='Recorded evaluator strict loading; independent CPU full-checkpoint audit pending')
(P/'EVALUATION_PROGRESS_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,indent=2))
