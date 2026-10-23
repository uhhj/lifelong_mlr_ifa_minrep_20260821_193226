import sys,json,hashlib
from pathlib import Path
P=Path(sys.argv[1]);d=json.loads((P/'EVIDENCE.json').read_text());assert not d['hash_bad'];download=json.loads((P/'DOWNLOAD.json').read_text());advanced=[]
for name,m in download.items():
 p=P/'metadata'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']
 if m['sha256']!=m['at_snapshot']['sha256']:advanced.append(name)
counts={};batches={6:157,7:227,8:309,9:362}
for arm in ('B0','B1','B2'):
 epochs=steps=0
 for p in (P/'metadata'/arm/'epochs').glob('task*.json'):
  rows=json.loads(p.read_text());task=int(p.stem[4:]);assert [r['epoch'] for r in rows]==list(range(len(rows))) and len(rows)<=100
  assert all(r['batches']==batches[task] and r['steps']==100*sum(batches[t] for t in range(6,task))+(r['epoch']+1)*batches[task] for r in rows)
  ds=json.loads((P/'metadata'/arm/'dataset'/('task%d.json'%task)).read_text())['rows'];valid=sum(sum(r['action_mask']) for r in ds);effect=sum(sum(r['effect_mask']) for r in ds)
  assert all(r['population']==len(ds) and r['action_valid']==valid and r['action_invalid']==4*len(ds)-valid and (arm=='B0' or r['effect_valid']==effect) for r in rows)
  epochs+=len(rows);steps+=sum(r['batches'] for r in rows)
 counts[arm]=dict(complete_epochs=epochs,main_updates=steps)
codec=P/'metadata/shared/CODEC_TRAINING_AND_VALIDATION.json';cc=json.loads(codec.read_text()) if codec.exists() else None
report=dict(status='PASS_PROGRESS_NOT_FINAL',observation_unix=d['unix'],contract_sha256=d['contract_sha256'],frozen_files=d['frozen_files'],arms=counts,codec_epochs=len(cc['epochs']) if cc else 0,codec_updates=cc['optimizer_steps'] if cc else 0,metadata_files=len(download),mutable_advances=advanced,controller_exit=d['controller_exit'],engineering_failure=d['controller_exit']=='1',no_partial_scientific_metrics=True)
(P/'LOCAL_AUDIT.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
