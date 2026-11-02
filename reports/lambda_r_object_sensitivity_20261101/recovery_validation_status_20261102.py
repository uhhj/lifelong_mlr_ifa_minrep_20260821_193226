import os,json,time
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/lambda_recovery_3090_20261102_v1')
out=dict(observed=time.time(),root=str(D),files={n:json.loads((D/n).read_text()) for n in ['DISPATCH.json','LEDGER.json','VALIDATION.json','FAILURE.json'] if (D/n).exists()},log_tail=(D/'validation.log').read_text(errors='replace')[-5000:])
pid=out['files']['DISPATCH.json']['pid'];p=Path('/proc')/str(pid)
out['launcher_state']=(p/'stat').read_text().split(') ',1)[1].split()[0] if p.exists() else None
print(json.dumps(out))
