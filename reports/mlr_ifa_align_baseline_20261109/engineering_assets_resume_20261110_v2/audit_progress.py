from pathlib import Path
import json,sys,subprocess,hashlib
R=Path(__file__).resolve().parent;BASE=R.parent;D=Path(sys.argv[1]);J=lambda p:json.loads(p.read_text(encoding='utf8'))
e=J(D/'EVIDENCE.json');assert e['engineering_contract_sha256']=='404ffb7e4a04f4d52091401e77df82e20086cfd2c7001f88c7c449642fd75e12'
subprocess.run([sys.executable,str(BASE/'audit_progress.py'),str(D)],check=True)
L={n:v['text'] for n,v in e['light'].items()}
assert L['coordinator/controller.exit'].strip()=='1' and L['OBJECT_41/evaluate.exit'].strip()=='1'
assert 'engineering/FAILURE.json' not in L,'Recovery failed: preserve evidence and inspect before any retry'
assert all(v.strip()=='0' for n,v in L.items() if n.startswith('engineering/') and n.endswith('.exit'))
events=[json.loads(x) for x in L.get('engineering/EVENTS.jsonl','').splitlines()];held=None;done=0
for x in events:
 if x['kind']=='environment_acquired':assert held is None;held=(x['case'],x['pid'])
 if x['kind']=='environment_released':assert held==(x['case'],x['pid']) and x['code']==0;held=None;done+=1
prefix='NOT_REQUESTED'
if len(sys.argv)>2:
 old=J(Path(sys.argv[2])/'EVIDENCE.json')['light']
 preserved=json.loads(L['engineering/PRESERVED.json'])['files']
 for n,v in old.items():
  if n=='OBJECT_41/random/evaluation/attempts_stage6_task6.jsonl' and v['sha256'] in preserved.values():continue
  if n.endswith('/random/epochs.json'):
   x=json.loads(v['text']);assert json.loads(L[n])[:len(x)]==x,n
  elif n.endswith('.jsonl'):assert L[n].startswith(v['text']),n
  elif '/ADMISSION_' in n or n.endswith('/INITIAL_STATE.json'):assert L[n]==v['text'],n
 prefix='PASS_ALL_EPOCHS_ADMISSIONS_AND_LEDGER_PREFIXES_EXCEPT_EXACT_HASH_PRESERVED_FAILED_ATTEMPT'
guard=json.loads(L.get('engineering/RNG_BOUNDARY_AUDIT.json','{}'))
if any(n.startswith('OBJECT_41/random/evaluation/trials_') for n in L):assert guard['status']=='PASS_EXACT_PRE_CELL_RNG'
out=dict(status='PASS_ENGINEERING_PROGRESS_NOT_FINAL',prefix=prefix,rng_boundary=guard,environment_mutex=dict(completed=done,active=held),old_failures_preserved=True,evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest())
(D/'ENGINEERING_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))
