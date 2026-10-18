"""Publish this read-only audit's reviewed code, report and metadata; no payloads."""
from pathlib import Path
import json, hashlib, re, subprocess
R=Path(__file__).resolve().parent
REPO=R.parents[1]/'publish_mlr_branch'
assert subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip()=='MLR'
assert subprocess.check_output(['git','-C',str(REPO),'remote','get-url','origin'],text=True).strip()=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
dest=REPO/'reports/goal_scv2_mechanism_audit'
allow={'.json','.jsonl','.log','.py','.md','.csv'}
exclude={'PUBLICATION_RECEIPT.json','scv2_batch_order_20261018.json','scv2_mechanism_tensor_rng_20261018.json'}
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
def review(obj):
    if isinstance(obj,dict):
        assert not {'last_action','password','private_key','latent_tensor'}&set(obj)
        for v in obj.values():review(v)
    elif isinstance(obj,list):
        for v in obj:review(v)
manifest={}
for p in sorted(R.rglob('*')):
    if not p.is_file() or '__pycache__' in p.parts or p.name in exclude:continue
    rel=p.relative_to(R)
    assert p.suffix in allow, str(p)
    assert not {'buffer','features','bindings','recovery'}&set(rel.parts),str(p)
    raw=p.read_bytes();assert len(raw)<5*1024**2
    content=raw.decode('utf-8-sig');assert not secret.search(content),str(p)
    assert not re.search(r'password\s*=\s*[\x22\x27][^\x22\x27]+[\x22\x27]',content,re.I),str(p)
    if p.suffix=='.json':review(json.loads(content))
    if p.suffix=='.jsonl':
        for line in content.splitlines():
            if line.strip():review(json.loads(line))
    target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
    manifest[rel.as_posix()]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
(dest/'.gitattributes').write_bytes(b'* -text -whitespace\n')
(dest/'PUBLICATION_MANIFEST.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode())
print(json.dumps({'files':len(manifest),'bytes':sum(v['bytes'] for v in manifest.values()),'destination':str(dest)}))
