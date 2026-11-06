"""Publish only the reviewed figure package to the standing authorized destination."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[2]
REPO=WORKSPACE/'publish_mlr_branch'
DEST=REPO/'reports/lambda_r_goal_sensitivity_20261103/publication_figures_20261106'

def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],text=True,encoding='utf8').strip()

assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only'),'Unrelated staged changes'
review=json.loads((ROOT/'VISUAL_REVIEW.json').read_text(encoding='utf8'))
assert review['status']=='PASS_RECOMPUTATION_EXPORT_AND_VISUAL_REVIEW'
export=json.loads((ROOT/'EXPORT_AUDIT.json').read_text(encoding='utf8'))
for item in export['figures']:
    assert hashlib.sha256((ROOT/'figures'/item['file']).read_bytes()).hexdigest()==item['sha256']
assert hashlib.sha256((ROOT/'figures/all_figures.pdf').read_bytes()).hexdigest()==export['book_sha256']
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
allowed={'.py','.json','.md','.csv','.pdf','.png','.svg'}
manifest={}
for p in sorted(ROOT.rglob('*')):
    if not p.is_file() or p.suffix not in allowed or 'review' in p.relative_to(ROOT).parts or p.name.startswith('PUBLICATION_'):
        continue
    raw=p.read_bytes();assert len(raw)<16*1024**2,p
    if p.suffix in {'.py','.json','.md','.csv','.svg'}:
        txt=raw.decode('utf-8-sig');assert not secret.search(txt),p
        if p.suffix=='.json':json.loads(txt)
    else:
        assert p.parent==ROOT/'figures' and (p.name.startswith(('fig01_','fig02_','fig03_','fig04_','figS01_','figS02_')) or p.name=='all_figures.pdf'),p
        assert raw.startswith(b'%PDF') if p.suffix=='.pdf' else raw.startswith(b'\x89PNG'),p
    rel=p.relative_to(ROOT).as_posix();dest=DEST/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    manifest[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(DEST/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
paths=[(DEST/x).relative_to(REPO).as_posix() for x in [*manifest,'PUBLICATION_MANIFEST.json']]
subprocess.run(['git','-C',str(REPO),'add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b'\0'.join(x.encode() for x in paths)+b'\0',check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):
    subprocess.run(['git','-C',str(REPO),'commit','-q','-m','Add verified OBJECT and GOAL publication figures'],check=True)
    subprocess.run(['git','-C',str(REPO),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0];assert head==remote
receipt=dict(status='PUSH_VERIFIED',commit=head,remote=remote,files=len(paths),bytes=sum(x['bytes'] for x in manifest.values()),observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
(ROOT/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt,indent=2))
