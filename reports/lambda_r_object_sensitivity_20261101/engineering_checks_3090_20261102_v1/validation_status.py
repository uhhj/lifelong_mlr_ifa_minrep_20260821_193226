import json,time,hashlib
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/lambda_3090_checks_training_20261102_v1')
v=json.loads((D/'DISPATCH.json').read_text());p=Path('/proc')/str(v['pid']);out=dict(dispatch=v,observed=time.time(),process_alive=p.exists(),source_sha256=hashlib.sha256((D/'validate.py').read_bytes()).hexdigest())
if p.exists():out['process_state']=(p/'stat').read_text().split(') ',1)[1].split()[0]
for n in ['VALIDATION_PROGRESS.json','VALIDATION.json']:
 if (D/n).exists():out[n]=json.loads((D/n).read_text())
if (D/'validation.log').exists():out['log_tail']='\n'.join((D/'validation.log').read_text(errors='replace').splitlines()[-12:])
print(json.dumps(out))
