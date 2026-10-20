"""Post-exit lightweight evidence package; no policy/optimizer/environment imports."""
from pathlib import Path
import json,hashlib,zipfile,time
N=Path('/data/mlr_ifa/experiment_workspaces/RA_ZRv2_autonomous_20261019')
Q=Path('/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019')
def js(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
light=Path((N/'package_light.log').read_text().strip())
m=js(light/'PACKAGE_MANIFEST.json');items={}
for name,v in m['files'].items():
 p=light/name;assert p.is_file() and sha(p)==v['sha256'] and p.stat().st_size==v['bytes'],name
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 assert p.suffix in ['.json','.jsonl','.log','.exit','.md']
 items['evidence/'+name]=p
for name in ['controller.exit','wrapper.exit','supervisor.exit','final_audit.exit','package_light.exit']:
 assert (N/name).read_text().strip()=='0';items['current/'+name]=N/name
for name in ['DEPLOYMENT_PROGRESS.json','FINAL_SERVER_ACCEPTANCE.json','EXPERIMENT_RESULTS.json','STATUS.json']:
 items['current/'+name]=N/name
parents=js(Q/'PARENTS.json')
for seed in [41,42,43]:
 p=Path(parents[str(seed)]['root'])
 for arm in ['random','coreset']:
  fs=list((p/arm/'evaluation').glob('trials*.jsonl'));assert len(fs)==10
  for f in fs:items['baselines/seed%d/%s/evaluation/%s'%(seed,arm,f.name)]=f
 for name in ['CONTRACT.json','MODAL_CONTRACT.json','GPU_PREFLIGHT.json','resolved_eval.json','resolved_coreset.json']:
  items['baselines/seed%d/%s'%(seed,name)]=p/name
for s in (N/'zr_runs').glob('*/RUN_SPEC.json'):items['specs/'+s.parent.name+'/RUN_SPEC.json']=s
for s in (N/'zr_ledger').glob('*'):
 for f in s.glob('*'):
  if f.is_file() and f.suffix in ['.json','.jsonl','.exit']:
   items['ledger/'+s.name+'/'+f.name]=f
items['REMOTE_PACKAGE_MANIFEST.json']=light/'PACKAGE_MANIFEST.json'
manifest=dict(time=time.time(),status='POST_EXIT_READONLY',files={n:dict(sha256=sha(p),bytes=p.stat().st_size,source=str(p)) for n,p in items.items()})
assert not (N/'completion_reviewed_20261020.zip').exists()
with zipfile.ZipFile(N/'completion_reviewed_20261020.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n,p in items.items():z.write(p,n)
 z.writestr('DOWNLOAD_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
receipt=dict(status='PASS',archive=str(N/'completion_reviewed_20261020.zip'),sha256=sha(N/'completion_reviewed_20261020.zip'),bytes=(N/'completion_reviewed_20261020.zip').stat().st_size,files=len(items),optimizer_calls=0,environment_steps=0)
(N/'completion_download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
