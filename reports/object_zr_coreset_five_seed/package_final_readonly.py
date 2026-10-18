"""Remote read-only hashes and allowlisted lightweight evidence package."""
import hashlib,json,tarfile,time
from pathlib import Path
Q=Path('/root/mlr_ifa/experiment_workspaces/ZR_coreset_seeds40_41_20261017')
D=Q/'monitor/final_20261018_1230'
D.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
checks={'seeds':{},'input_mismatches':[]}
inputs={}
files={}
for seed in (40,41):
 o=Path(f'/root/mlr_ifa/experiments/ZR_coreset_OBJECT_seed{seed}_20261017')
 c=read(o/'CONTRACT.json')
 bad=[p for p,v in c['source_sha256'].items() if sha(Q/f'seed{seed}/source'/p)!=v]
 cfg=[p for p,v in c['configs'].items() if sha(Path(p))!=v]
 checks['seeds'][seed]={'source_mismatches':bad,'config_mismatches':cfg}
 assert not bad and not cfg
 for p,v in read(o/'INPUT_SHA256.json').items():
  assert p not in inputs or inputs[p]==v
  inputs[p]=v
 for p in o.rglob('*'):
  if p.is_file() and p.suffix in ('.json','.jsonl','.log','.md','.yaml','.exit','.csv') and 'integration' not in p.relative_to(o).parts:
   assert p.stat().st_size<25*1024*1024
   files[f'seed{seed}/'+str(p.relative_to(o))]=p
for p,v in inputs.items():
 if sha(Path(p))!=v:checks['input_mismatches'].append(p)
checks['unique_inputs']=len(inputs)
checks['orchestration_unchanged']=all(sha(Q/p)==v for p,v in read(Q/'ORCHESTRATION_SHA256.json').items())
assert not checks['input_mismatches'] and checks['orchestration_unchanged']
(D/'hash_checks.json').write_text(json.dumps(checks,indent=2))
files['hash_checks.json']=D/'hash_checks.json'
for name in ('FIVE_SEED_SUMMARY.json','FINAL_QUEUE_ACCEPTANCE.json','queue.exit','commands.log','ORCHESTRATION_SHA256.json','aggregate_five_seeds.py','final_queue_audit.py'):
 files['queue/'+name]=Q/name
manifest={k:{'remote':str(p),'size':p.stat().st_size,'sha256':sha(p)} for k,p in files.items()}
(D/'manifest.json').write_text(json.dumps(manifest,indent=2))
with tarfile.open(D/'light.tar.gz','w:gz') as t:
 for k,p in files.items():t.add(p,arcname=k,recursive=False)
print(json.dumps({'files':len(files),'bytes':(D/'light.tar.gz').stat().st_size,'archive_sha256':sha(D/'light.tar.gz'),'hash_checks':checks}))
