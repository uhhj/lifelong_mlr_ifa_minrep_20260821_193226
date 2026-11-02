import sys,json,hashlib
from pathlib import Path
E=Path(__file__).resolve().parent;Q=E.parent;case=sys.argv[1]
assert case in ['L005_43','L020_41','L020_42','L020_43']
c=json.loads((E/'CONTRACT.json').read_text());assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in c['files'].items())
assert (E/'LIVE_COMMITTED.json').exists() and (E/'HOOK_ENABLED.json').exists()
P=Q/'engineering_checks_3090_20261102_v1/train_wrapper.py'
exec(compile(P.read_text(),str(P),'exec'),dict(__file__=str(P),__name__='__main__'))
