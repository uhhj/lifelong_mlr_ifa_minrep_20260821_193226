"""Read-only progress verification; incomplete results are not final metrics."""
from pathlib import Path
from collections import defaultdict
import json,hashlib,datetime
P=Path(__file__).resolve().parent;E=P/'evidence'
m=json.loads((E/'PACKAGE_MANIFEST.json').read_text())
for n,r in m['files'].items():
 p=E/n
 assert p.stat().st_size==r['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],n
s=json.loads((E/'progress_snapshot_1646.json').read_text())
epochs=json.loads((E/'epochs.json').read_text())
assert len(epochs)==40 and sum(x['batches'] for x in epochs)==10550 and sum(x['samples'] for x in epochs)==105240
structures=[json.loads(x) for x in (E/'STRUCTURE.jsonl').read_text().splitlines()]
assert len(structures)==16 and all(x['passed'] for x in structures)
records=[];attempts=[];cells=defaultdict(list)
for p in (E/'evaluation').glob('trials*.jsonl'):
 for line in p.read_text().splitlines():
  r=json.loads(line);assert type(r['success']) is bool
  records.append(r);cells[r['stage'],r['task']].append(r)
for p in (E/'evaluation').glob('attempts*.jsonl'):attempts.extend(json.loads(x) for x in p.read_text().splitlines())
assert len(records)==190 and len({(r['stage'],r['task'],r['trial']) for r in records})==190
assert sum(r['event']=='vector_attempt_complete' for r in attempts)==19
assert not any(r['event']=='technical_exception' for r in attempts)
starts={(r['stage'],r['task'],r['group']):r['unix'] for r in attempts if r['event']=='vector_attempt_start'}
durations=[r['unix']-starts[r['stage'],r['task'],r['group']] for r in attempts if r['event']=='vector_attempt_complete']
out=dict(status='PARTIAL',observed_CST=datetime.datetime.fromtimestamp(s['time'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),epochs=40,optimizer_steps=10550,exposures=105240,structure_pass=16,formal_records=190,completed_vectors=19,technical_exceptions=0,mean_vector_seconds=sum(durations)/len(durations),remaining_minutes_estimate=[5,10],cells=[dict(stage=k[0],task=k[1],n=len(v),successes=sum(r['success'] for r in v),complete=len(v)==20) for k,v in sorted(cells.items())],final_metrics_not_computed=True,checked_files=len(m['files']))
(P/'VERIFICATION.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
