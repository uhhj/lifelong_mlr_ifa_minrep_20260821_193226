"""Independent local episode/attempt audit; no partial performance metrics."""
import json,sys,hashlib,datetime
from pathlib import Path
from formal_audit import audit_formal
p=Path(sys.argv[1]);raw=(p/'EVIDENCE.json').read_bytes();d=json.loads(raw)
assert json.loads((p/'ATTEMPT.json').read_text())['exit']==0
runs={}
for key,r in d['runs'].items():
 if r['trials'] or r['attempts']:
  result=audit_formal(r,41)
  assert result['records']==200 or any('coreset_exp.evaluate' in x['command'] and '/'+key.split('_')[0]+'/source' in x['cwd'] for x in d['processes'])
  runs[key]=result
out=dict(status='PARTIAL_EVALUATION_ACCOUNTING_PASS',observed_at=datetime.datetime.fromtimestamp(d['observed_unix'],datetime.timezone(datetime.timedelta(hours=8))).isoformat(),sha256=hashlib.sha256(raw).hexdigest(),runs=runs,final_acceptance=False)
(p/'LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
