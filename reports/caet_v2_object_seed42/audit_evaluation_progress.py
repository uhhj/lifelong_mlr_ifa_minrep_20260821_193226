"""Read-only evaluation progress accounting; not final checkpoint/process acceptance."""
import json,sys,hashlib,datetime
from pathlib import Path
from formal_audit import audit_formal
p=Path(sys.argv[1]);raw=(p/'SNAPSHOT.json').read_bytes();assert hashlib.sha256(raw).hexdigest()==json.loads((p/'RECEIPT.json').read_text())['sha256'];d=json.loads(raw)
f=audit_formal(dict(trials=d['formal'],attempts=d['attempts'],cells={}))
# This progress snapshot contains original per-episode identities and attempt
# outcomes, but not cell summary files. Final acceptance independently loads cells.
f.pop('completed_cells');f['cell_summary_check']='DEFERRED_TO_FINAL_PACKAGE'
assert any('coreset_exp.evaluate' in p['command'] for p in d['processes']) or len(d['formal'])==200
out=dict(status='PARTIAL_EVALUATION_ACCOUNTING_PASS',observed_at=datetime.datetime.fromtimestamp(d['observed_unix']).astimezone().isoformat(),formal=f,not_final_acceptance=True)
(p/'EVALUATION_LOCAL_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out,indent=2))
