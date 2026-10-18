"""Verify actual final-buffer Task6 files and historical stage manifests; no mutation."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
inputs={};rows=[]
d=json.loads((P/'DIAGNOSTICS.json').read_text())
for c in d['cells']:
 root=Path(c['root'])
 for arm in ('random','coreset'):
  cfg=root/('resolved_%s.json'%arm);inputs[str(cfg)]=sha(cfg);buffer=Path(json.loads(cfg.read_text())['buffer_dir'])
  p=root/arm/'storage/task6.json';inputs[str(p)]=sha(p);st=json.loads(p.read_text())
  expected={w['destination']:w['sha256'] for w in st['selected']}
  assert set(p.name for p in buffer.glob('t6__*.pt'))==set(expected)
  for name,h in expected.items():
   p=buffer/name;assert sha(p)==h;inputs[str(p)]=h
  for stage in range(6,10):
   p=root/arm/('buffer_after_task%d.json'%stage);inputs[str(p)]=sha(p);b=json.loads(p.read_text())
   assert {k:v for k,v in b['hashes'].items() if k.startswith('t6__')}==expected
  rows.append(dict(suite=c['suite'],seed=c['seed'],arm=arm,count=len(expected),actual_buffer=str(buffer),unchanged_across_stages=True))
assert all(sha(p)==h for p,h in inputs.items())
assert not (P/'BUFFER_VERIFICATION.json').exists()
(P/'BUFFER_VERIFICATION.json').write_text(json.dumps(dict(status='PASS_READONLY',rows=rows,inputs_sha256=inputs,hashes_before_after_equal=True),indent=2)+'\n')
print(json.dumps(dict(status='PASS_READONLY',actual_payload_files=sum(r['count'] for r in rows),arms=len(rows))))
