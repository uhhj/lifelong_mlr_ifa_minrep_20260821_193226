import re
from pathlib import Path
def audit_formal(formal, seed=42):
    trials=formal['trials'];events=formal['attempts']
    keys=[(r['stage'],r['task'],r['trial']) for r in trials]
    assert len(keys)==len(set(keys))<=200
    for r in trials:
        assert 6<=r['task']<=r['stage']<=9 and r['epoch']==99 and r['seed']==seed
        assert 0<=r['trial']<20 and r['initial_state_index']==r['trial']
        assert r['initial_state_sha256']==r['initial_hashes'][r['trial']]
        assert isinstance(r['success'],bool) and 0<r['vector_group_steps']<=600
    assert not any(e['event']=='technical_exception' for e in events)
    starts={(e['stage'],e['task'],e['group']):e for e in events if e['event']=='vector_attempt_start'}
    done={(e['stage'],e['task'],e['group']):e for e in events if e['event']=='vector_attempt_complete'}
    assert len(starts)==sum(e['event']=='vector_attempt_start' for e in events)
    assert len(done)==sum(e['event']=='vector_attempt_complete' for e in events)
    assert set(done)<=set(starts) and len(starts)<=20 and len(starts)-len(done)<=1
    for k,e in starts.items():
        assert e['group'] in (0,1) and e['env_seed']==seed+e['group']
        assert e['initial_indices']==list(range(10*e['group'],10*(e['group']+1)))
    for k,e in done.items():
        rows=sorted((r for r in trials if (r['stage'],r['task'],r['trial']//10)==k),key=lambda r:r['trial'])
        assert len(rows)==10 and [r['success'] for r in rows]==e['successes']
        assert all(r['vector_group_steps']==e['steps'] for r in rows)
        assert 0<e['steps']<=600 and e['lanes']==10
    for name,cell in formal['cells'].items():
        m=re.fullmatch(r'cell_stage(\d+)_task(\d+)\.json',name);assert m
        stage,task=map(int,m.groups())
        assert cell['stage']==stage and cell['epoch']==99
        assert Path(cell['trial_log']).name==f'trials_stage{stage}_task{task}.jsonl'
        assert Path(cell['attempt_log']).name==f'attempts_stage{stage}_task{task}.jsonl'
        rows=[r for r in trials if (r['stage'],r['task'])==(stage,task)]
        assert len(rows)==20 and sum(r['success'] for r in rows)/20==cell['success_rate']
        assert all(r['checkpoint_sha256']==cell['checkpoint_sha256'] and r['initial_hashes']==cell['initial_hashes'] for r in rows)
    durations=[e['unix']-starts[k]['unix'] for k,e in done.items()]
    avg=sum(durations)/len(durations) if durations else None
    return dict(records=len(trials),completed_vectors=len(done),started_vectors=len(starts),
        completed_cells=len(formal['cells']),technical_exceptions=0,
        mean_vector_seconds=avg,remaining_evaluation_hours=(20-len(done))*avg/3600 if avg else None)
