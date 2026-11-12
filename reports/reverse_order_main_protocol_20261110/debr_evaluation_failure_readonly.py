"""Capture the stopped DEBR evaluator and the completed queue prefix."""
from pathlib import Path
import json
import sys
import time

root = Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
sys.path.insert(0, str(root))
from reverse_common import processes, read, sha, verify

spec = verify()['cases']
name = 'OBJECT_DEBR_41'
case = root / 'cases' / name
paths = {'root/' + n: root / n for n in ('controller.exit', 'FAILURE.json', 'STATUS.json')}
paths.update({'case/' + n: case / n for n in ('evaluate.log', 'evaluate.exit', 'EVAL_DISPATCH.json', 'EVAL_ADMISSION.json', 'EVAL_EVENTS.jsonl', 'EVALUATION_STARTED.json')})
log = (case / 'evaluate.log').read_text(errors='replace')
print(json.dumps(dict(observed=time.time(), contract=sha(root / 'CONTRACT.json'), processes=processes(),
    failed_hashes={k: sha(v) for k, v in paths.items()}, log=log[-22000:], events=(case / 'EVAL_EVENTS.jsonl').read_text(),
    coordinator=[json.loads(line) for line in (root / 'EVENTS.jsonl').read_text().splitlines()][-8:],
    shards={n: len(list((root / 'cases' / n / 'outcomes').glob('*.json'))) for n in spec})))
