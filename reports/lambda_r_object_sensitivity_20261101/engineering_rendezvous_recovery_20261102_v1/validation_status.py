import json,time,hashlib,os
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/rendezvous_recovery_20261102_v1');names=['BINDING.json','DISPATCH.json','LEDGER.json','LAUNCHER_VALIDATION.json','VALIDATION.json','FAILURE.json','REPLAY_PROGRESS.json','STARTUP_ATTEMPT1_FAILURE.json'];data={n:json.loads((D/n).read_text()) for n in names if (D/n).exists()};pid=data['DISPATCH.json']['pid'];p=Path('/proc')/str(pid);proc=None
if p.exists():
 fields=(p/'stat').read_text().split(') ',1)[1].split();proc=dict(pid=pid,state=fields[0],exit_code=int(fields[49]))
print(json.dumps(dict(status='READ_ONLY_EXISTING_RECOVERY_VALIDATION',time=time.time(),data=data,process=proc,validation_log=(D/'validation.log').read_text()[-6000:],source_hashes={n:hashlib.sha256((D/n).read_bytes()).hexdigest() for n in ['validate.py','probe_worker.py']})))
