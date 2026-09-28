"""Summarize only a complete validated 200-episode formal record."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from libero.lifelong.runbook_core import metrics, atomic_json

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.dry_run:
        print('Requires 4 registered final checkpoints and 200 unique completed formal rollouts.')
        return
    run = args.run_dir
    checkpoints = json.loads((run / 'checkpoints/manifest.json').read_text())
    rows = [json.loads(s) for s in (run / 'evaluation/rollouts.jsonl').read_text().splitlines() if s.strip()]
    if len(rows) != 200:
        raise ValueError('Exactly 200 formal episode records required')
    r = np.full((4, 4), np.nan)
    for stage in range(6, 10):
        cp = checkpoints[str(stage)]
        assert cp['epoch_index'] == 99
        for task in range(6, stage + 1):
            cell = [v for v in rows if v['stage'] == stage and v['task_id'] == task]
            assert len(cell) == 20 and {v['init_state_id'] for v in cell} == set(range(20))
            assert len({v['init_state_hash'] for v in cell}) == 20
            for v in cell:
                assert v['kind'] == 'FORMAL' and v['training_seed'] == 42
                assert not v['exception'] and v['checkpoint_sha'] == cp['sha256']
                assert type(v['success']) is bool
            r[stage - 6, task - 6] = sum(v['success'] for v in cell) / 20
    atomic_json(run / 'metrics.json', metrics(r))
    np.savetxt(run / 'evaluation/success_matrix.csv', r, delimiter=',',
               header='task6,task7,task8,task9', comments='')

if __name__ == '__main__':
    main()
