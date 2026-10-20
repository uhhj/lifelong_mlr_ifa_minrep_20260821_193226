"""Preserve original package; supplement stage1 metadata with unique names."""
from pathlib import Path
import json,hashlib,zipfile,time
N=Path('/data/mlr_ifa/experiment_workspaces/RA_ZRv2_autonomous_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
files={}
for s in [41,42,43]:
 o=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed%d_20261019'%s)
 for p in o.glob('*.json'):
  assert p.stat().st_size<8*1024**2
  raw=p.read_bytes();assert b'"last_action"' not in raw and b'"password"' not in raw
  files['RA_STAGE1_seed%d/%s'%(s,p.name)]=p
receipt=dict(status='PASS_READONLY_SUPPLEMENT',reason='Original package collector used identical OBJECT_RA_seed{s} prefixes for RA and ZRv2; top-level metadata collided. Original package preserved. Scientific artifacts and phase-specific outcomes unchanged.',time=time.time(),files={n:dict(bytes=p.stat().st_size,sha256=sha(p),source=str(p)) for n,p in files.items()},optimizer_calls=0,environment_steps=0)
out=N/'stage1_metadata_supplement_20261020.zip'
assert not out.exists()
with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
 for n,p in files.items():z.write(p,n)
 z.writestr('SUPPLEMENT_MANIFEST.json',json.dumps(receipt,indent=2)+'\n')
(N/'stage1_supplement_receipt.json').write_text(json.dumps(dict(status='PASS',sha256=sha(out),bytes=out.stat().st_size,files=len(files)),indent=2)+'\n')
print((N/'stage1_supplement_receipt.json').read_text())
