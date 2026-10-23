import json
from pathlib import Path
P=Path('/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed41_20261019')
out={}
for t in (6,7):
 b=json.loads((P/'bindings'/('task%d.json'%t)).read_text());rows=[]
 for name,r in b['windows'].items():
  if not r['identity_unique']:rows.append({k:v for k,v in r.items() if k!='last_action'})
 out[str(t)]=rows
print(json.dumps(out))
