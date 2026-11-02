"""Read existing diagnostic evidence only; never launch workers."""
import json,time
from pathlib import Path
D=Path('/data/mlr_ifa/diagnostics/lambda_parallel_stage9_20261102_v1')
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return None
 s=(p/'stat').read_text().split(') ',1)[1].split()
 return dict(pid=pid,state=s[0],cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode())
files={str(p.relative_to(D)):json.loads(p.read_text()) for p in D.rglob('*.json') if not p.name.endswith('_TRACE.json')}
dispatch=files['DISPATCH.json'];events=files.get('EVENTS.json',[])
print(json.dumps(dict(status='READ_EXISTING_PARALLEL_DIAGNOSTIC',observed=time.time(),files=files,driver=proc(dispatch['pid']),workers=[proc(x['pid']) for x in events if x['event']=='launch'],logs={str(p.relative_to(D)):p.read_text(errors='replace')[-5000:] for p in D.rglob('*.log')},optimizer_calls=0,environments=0,production_writes=False)))
