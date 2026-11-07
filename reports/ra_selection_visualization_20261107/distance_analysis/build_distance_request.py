"""Bundle an isolated read-only CPU analysis against exact archived manifests."""
from pathlib import Path
import json,hashlib
R=Path(__file__).resolve().parent;W=R.parents[1]
cases=json.loads((R/'INPUT_CASES.json').read_text(encoding='utf8'))
for c in cases:
    p=Path(c.pop('local'))
    c['expected']={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['FEATURES.json']+[f'ra_coreset/storage/task{t}.json' for t in range(6,10)]}
src=W/'execution/20261019_ra_object_conditional/source/libero/lifelong/selectors/greedy_kcenter.py'
raw=src.read_bytes();code=raw.decode('utf8')
prefix='CASES='+repr(cases)+'\nFROZEN_SOURCE='+repr(code)+'\nFROZEN_FILE_SHA256='+repr(hashlib.sha256(raw).hexdigest())+'\n'
(R/'distance_readonly.py').write_text(prefix+(R/'distance_body.py').read_text(encoding='utf8'),encoding='utf8',newline='\n')
print('Prepared CPU-only distance request; not dispatched.')
