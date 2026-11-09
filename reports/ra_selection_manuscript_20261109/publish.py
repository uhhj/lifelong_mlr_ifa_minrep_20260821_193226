"""Publish the exact reviewed lightweight comparison, never scientific payloads."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REPO=ROOT/'publish_mlr_branch'
DEST=REPO/'reports/ra_selection_manuscript_20261109'
FILES=['edit_manuscript.py','publish.py','EDIT_RECEIPT.json','QA.json','INSERTED_SECTION.md']

def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],text=True,encoding='utf8').strip()

assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only'), 'Unrelated staged changes'
audit=json.loads((HERE/'EDIT_RECEIPT.json').read_text(encoding='utf8'))
assert audit['status']=='PASS_EDITED_AND_VISUALLY_VERIFIED'
qa=json.loads((HERE/'QA.json').read_text(encoding='utf8'))
assert qa['status']=='PASS_TARGETED_EDIT_AND_VISUAL_QA'
for key in ['source','output','data_source']:
    hashkey='data_sha256' if key=='data_source' else key+'_sha256'
    assert hashlib.sha256(Path(audit[key]).read_bytes()).hexdigest()==audit[hashkey]
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
manifest={}
for name in FILES:
    b=(HERE/name).read_bytes()
    assert len(b)<2*1024**2
    assert not secret.search(b.decode('utf8'))
    if name.endswith('.json'):json.loads(b)
    manifest[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
DEST.mkdir(parents=True,exist_ok=True)
prior=json.loads((DEST/'PUBLICATION_MANIFEST.json').read_text(encoding='utf8')) if (DEST/'PUBLICATION_MANIFEST.json').exists() else {}
for name in FILES:
    target=DEST/name
    if target.exists():
        assert target.read_bytes()==(HERE/name).read_bytes() or hashlib.sha256(target.read_bytes()).hexdigest()==prior.get(name,{}).get('sha256'), 'Preserve unrelated existing edits'
    target.write_bytes((HERE/name).read_bytes())
(DEST/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
paths=[(DEST/n).relative_to(REPO).as_posix() for n in [*FILES,'PUBLICATION_MANIFEST.json']]
subprocess.run(['git','-C',str(REPO),'add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b'\0'.join(p.encode('utf8') for p in paths)+b'\0',check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run(['git','-C',str(REPO),'-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol','diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):
    subprocess.run(['git','-C',str(REPO),'commit','-q','-m','Add RA replay selection ablation to manuscript with verified layout'],check=True)
    subprocess.run(['git','-C',str(REPO),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD')
remote=git('ls-remote','origin','refs/heads/MLR').split()[0]
assert head==remote
receipt=dict(status='PUSH_VERIFIED',commit=head,remote=remote,files=manifest,
    observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
(HERE/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=receipt['status'],commit=head,files=len(paths)),indent=2))
