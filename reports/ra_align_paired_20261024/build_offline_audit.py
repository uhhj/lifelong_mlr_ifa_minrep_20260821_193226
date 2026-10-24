from pathlib import Path
R=Path(__file__).resolve().parent;P=R.parent/'20261024_align_behavior_drift'
t=(P/'audit_local.py').read_text()
a=t.index("spec=read(");b=t.index('complete=read(',a)
t=t[:a]+'''spec=read(R/'INPUT_SPEC.json');frozen=read(R/'OFFLINE_FROZEN.json')
for name,h in frozen.items():assert hashlib.sha256((R/name).read_bytes()).hexdigest()==h
attempt=read(OUT/'ATTEMPT.json');assert attempt['exit']==0 and attempt['script_sha256']==hashlib.sha256((R/'ra_drift_readonly.py').read_bytes()).hexdigest()
'''+t[b:]
t=t.replace("old['contract_sha256']==d['contract_sha256']==e['contract_sha256']", "old['contract_sha256']==d['common_align_contract_sha256']==e['contract_sha256'] and d['contract_sha256']==e['ra_contract_sha256']")
t=t.replace("e['checkpoints'].items()", "e['ra_checkpoints'].items()")
t=t.replace("Fraction(x['current'])", "Fraction(x['parent'])")
# Completed package matrix uses parent naming, checked before execution below.
(R/'audit_ra_local.py').write_text(t,encoding='utf8')
