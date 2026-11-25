"""Recompute the demonstration counts and full-data training budget from public metadata."""
import hashlib
import json
from math import ceil, isclose, isfinite
from pathlib import Path


def load(path):
    return json.loads(path.read_bytes())


def audit(path):
    manifest=load(path/'manifest.json')
    history=load(path/'history.json')
    cfg=load(path/'config.json')
    ev=load(path/'evidence.json')
    entries=manifest['episodes']
    digest=hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest()
    assert digest==cfg['manifest_sha256']==ev['manifest_sha256']
    assert hashlib.sha256((path/'refresh.py').read_bytes()).hexdigest()==cfg['adapter_sha256']
    assert hashlib.sha256((path.parents[1]/'scripts/piper_base/piper_base.py').read_bytes()).hexdigest()==cfg['script_sha256']
    assert cfg['checkpoint_sha256']=='241d3370acfc04d694df04256956a97aa13cfdc2e651cf258d338f8e72988be7'
    assert len(entries)==len({r['id'] for r in entries})==len({r['alignment_sha256'] for r in entries})==150
    assert all(r['split']=='train' and r['frames']>0 for r in entries)
    assert cfg['validation_episodes']==ev['validation_episodes']==0
    assert manifest['remaining_unselected_success_count']==34 and len(manifest['excluded_failures'])==7
    assert not {r['id'] for r in entries}.intersection(manifest['excluded_failures'])
    for task in range(3):
        rows=[r for r in entries if r['task']==task]
        label=f'B{task+1}'
        assert ev['training'][label]==dict(episodes=len(rows),action_rows=sum(r['frames'] for r in rows))
        assert len(rows)==50 and ev['verification']['training_metrics'][label]['rows']==ev['training'][label]['action_rows']
    count=sum(r['frames'] for r in entries)
    assert len(history)==cfg['epochs']==ev['finished']['epochs']==12
    for i,r in enumerate(history,1):
        assert r['epoch']==i and r['rows']==count and r['updates']==ceil(count/cfg['batch_size'])
        assert isfinite(r['train_nll'])
    assert ev['finished']['updates']==sum(r['updates'] for r in history)
    assert ev['finished']['checkpoint_sha256']==ev['verification']['final_sha256']
    assert {r['task'] for r in ev['verification']['raw_training_windows']}=={'B1','B2','B3'}
    for r in ev['verification']['raw_training_windows']:
        assert r['episode'] in {e['id'] for e in entries if r['task']==f"B{e['task']+1}"}
    metrics=ev['verification']['training_metrics']
    assert isclose(sum(r['normalized_mae'] for r in metrics.values())/3,ev['verification']['training_macro_normalized_mae'])
    print('Verified',len(entries),'training episodes,',count,'action rows,',ev['finished']['updates'],'updates;',ev['verification']['final_sha256'])


if __name__=='__main__':
    audit(Path(__file__).resolve().parent)
