"""Review/copy only current task lightweight artifacts into publication checkout."""
from pathlib import Path
import hashlib,json,re,shutil,subprocess
R=Path(__file__).resolve().parent;WORK=R.parents[1];REPO=WORK/'publish_mlr_branch'
assert subprocess.check_output(['git','-C',str(REPO),'branch','--show-current'],text=True).strip()=='MLR'
url=subprocess.check_output(['git','-C',str(REPO),'remote','get-url','origin'],text=True).strip()
assert url=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
allowed={'.py','.json','.jsonl','.md','.sh','.yaml','.csv','.patch','.exit','.log','.pid','.started','.txt','.sbatch'}
blocked_parts={'__pycache__','bindings','buffer','features','models','recovery','smoke'}
# Model source code lives in source/libero/lifelong/models: explicitly permit source .py.
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
manifest={}
for src,label in [(R.parent/'20261018_scv2_offline','offline'),(R,'launch')]:
    dest=REPO/'reports/goal_scv2_seed42'/label;dest.mkdir(parents=True,exist_ok=True)
    for p in sorted(src.rglob('*')):
        if not p.is_file() or (p.suffix not in allowed and p.name!='LICENSE'):continue
        rel=p.relative_to(src)
        if '__pycache__' in rel.parts:continue
        if any(x in rel.parts for x in blocked_parts) and not ('source' in rel.parts and p.suffix=='.py'):continue
        if p.name in {'PUBLICATION_RECEIPT.json','build_source.py'}:continue
        raw=p.read_bytes();assert len(raw)<5*1024**2,(str(p),len(raw))
        text=raw.decode('utf-8');assert not secret.search(text),str(p)
        # Reject literal credential assignments while allowing code that accepts user secrets.
        assert not re.search(r'password\s*=\s*[\x22\x27][^\x22\x27]+[\x22\x27]',text,re.I),str(p)
        if p.suffix=='.json':
            obj=json.loads(text)
            def check(v):
                if isinstance(v,dict):
                    assert 'last_action' not in v and 'private_key' not in v and 'password' not in v,str(p)
                    for x in v.values():check(x)
                elif isinstance(v,list):
                    for x in v:check(x)
            check(obj)
        target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
        manifest[str(target.relative_to(REPO)).replace('\\','/')]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (dest/'.gitattributes').write_text('* -text -whitespace\n')
out=REPO/'reports/goal_scv2_seed42/PUBLICATION_MANIFEST.json'
out.write_bytes((json.dumps(manifest,indent=2)+'\n').encode('utf-8'))
print(json.dumps(dict(reviewed_files=len(manifest),bytes=sum(v['bytes'] for v in manifest.values()),destination=str(out.parent))))
