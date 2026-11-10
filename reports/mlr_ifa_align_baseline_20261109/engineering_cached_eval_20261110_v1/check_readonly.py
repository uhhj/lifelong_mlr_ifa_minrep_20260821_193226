import sys
sys.path.insert(0,'/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2/engineering_cached_eval_20261110_v1')
from runtime_cached import *
verify()
from observe import snapshot
s=snapshot();s['engineering_contract_sha256']=sha(OLD/'CONTRACT.json');s['cache_contract_sha256']=sha(E/'CONTRACT.json')
for root,prefix in [(OLD,'engineering/'),(E,'cached_engineering/')]:
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix not in ['.json','.jsonl','.exit']:continue
        raw=p.read_bytes()
        if p.suffix=='.jsonl':raw=raw[:raw.rfind(b'\n')+1]
        s['light'][prefix+str(p.relative_to(root))]=dict(text=raw.decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
print(json.dumps(s))
