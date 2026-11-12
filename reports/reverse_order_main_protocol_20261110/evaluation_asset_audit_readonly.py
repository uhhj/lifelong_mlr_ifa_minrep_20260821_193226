"""Identify original simulator assets and distinguish constructor failure from rollout."""
from pathlib import Path
import hashlib
import json
import re
import sys
import time

root = Path('/data/mlr_ifa/experiment_workspaces/REVERSE_MAIN_MLR_DEBR_OBJECT_GOAL_20261110_v1_engineering05')
sys.path.insert(0, str(root))
from reverse_common import processes, read, sha, verify

verify()
case = root / 'cases/OBJECT_MLR_41'
log = (case / 'evaluate.log').read_text(errors='replace')
assets = {}
for name in read(root / 'SPEC.json')['cases']:
    directory = root / 'cases' / name
    info = read(directory / 'INPUT.json')
    path = Path(info['source_root']) / 'libero/libero/assets'
    xml = path / 'scenes/libero_floor_base_style.xml'
    assets[name] = dict(path=str(path), real=str(path.resolve()), exists=path.is_dir(), link=path.is_symlink(),
                        scene=sha(xml) if xml.is_file() else None)
official = Path('/data/mlr_ifa/src/libero/libero/assets')
xml = official / 'scenes/libero_floor_base_style.xml'
lines = [line for line in log.splitlines() if any(x in line for x in ('Process Process-', 'env=SubprocVectorEnv', 'FileNotFoundError:', 'ConnectionResetError:', 'env.reset(', 'env.step('))]
events = [json.loads(line) for line in (root / 'EVENTS.jsonl').read_text().splitlines()]
paths = {'root/' + n: root / n for n in ('controller.exit', 'FAILURE.json', 'STATUS.json')}
paths.update({'case/' + n: case / n for n in ('evaluate.log', 'evaluate.exit', 'EVAL_DISPATCH.json', 'EVAL_ADMISSION.json', 'EVAL_EVENTS.jsonl', 'EVALUATION_STARTED.json', 'EVALUATION_PROTOCOL_DECLARATION.json')})
print(json.dumps(dict(status='READONLY', observed=time.time(), contract=sha(root / 'CONTRACT.json'), processes=processes(),
    assets=assets, official=dict(real=str(official.resolve()), scene=sha(xml) if xml.is_file() else None),
    trace=dict(workers=sorted(set(map(int, re.findall(r'Process Process-(\d+):', log)))), missing_scene=log.count('FileNotFoundError: [Errno 2]'),
               vector_constructor=log.count('env=SubprocVectorEnv'), reset='env.reset(' in log, step='env.step(' in log, lines=lines),
    failed_hashes={k: sha(v) for k, v in paths.items()}, last_events=events[-4:])))
