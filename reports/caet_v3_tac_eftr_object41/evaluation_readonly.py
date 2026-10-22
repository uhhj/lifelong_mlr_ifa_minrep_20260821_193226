"""Read only lightweight outcome/attempt records for the authorized queue."""
import json,time,os
from pathlib import Path
Q=Path('/data/mlr_ifa/experiment_workspaces/CAETv3_TAC_EFTR_ALIGN_OBJECT41_20261022_collate01')
out=dict(observed_unix=time.time(),runs={},processes=[])
for entry in json.loads((Q/'QUEUE.json').read_text()):
 p=Path(entry['output'])/'ra_coreset/evaluation'
 def records(pattern):
  return [json.loads(line) for f in sorted(p.glob(pattern)) for line in f.read_text().splitlines() if line.strip()]
 out['runs'][entry['key']]=dict(trials=records('trials*.jsonl'),attempts=records('attempts*.jsonl'),cells={f.name:json.loads(f.read_text()) for f in sorted(p.glob('cell*.json'))})
for p in Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();cwd=str((p/'cwd').resolve())
  if int(p.name)!=os.getpid() and 'python -c' not in cmd and ('CAETv3_' in cmd or 'CAETv3_' in cwd):out['processes'].append(dict(pid=int(p.name),command=cmd,cwd=cwd))
 except OSError:pass
print(json.dumps(out))
