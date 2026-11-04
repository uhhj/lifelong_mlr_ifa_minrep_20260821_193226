"""Publish only reviewed authored text; original template and binaries stay local."""
from pathlib import Path
import subprocess,json,hashlib,datetime,re
R=Path(__file__).resolve().parent
REPO=R.parents[1]/'publish_mlr_branch'
DEST=REPO/'papers/ras_ra_coreset_shbr'
def git(*args):
    return subprocess.check_output(['git','-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','-C',str(REPO),*args],text=True,encoding='utf8').strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
staged=git('diff','--cached','--name-only').splitlines()
assert all(p.startswith('papers/ras_ra_coreset_shbr/') for p in staged), 'Unrelated staged files'
DEST.mkdir(parents=True,exist_ok=True)
manifest={}
for p in sorted(R.iterdir()):
    if not p.is_file() or p.suffix not in {'.tex','.bib','.md','.json','.txt','.py','.ps1'} or p.name.startswith('PUBLICATION_'):
        continue
    raw=p.read_bytes(); assert len(raw)<2*1024**2
    text=raw.decode('utf-8-sig')
    assert not re.search(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|github_pat_[A-Za-z0-9_]{30,}|ghp_[A-Za-z0-9]{20,}',text)
    (DEST/p.name).write_bytes(raw)
    manifest[p.name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
(DEST/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
names=[*manifest,'PUBLICATION_MANIFEST.json']
paths=[(DEST/n).relative_to(REPO).as_posix() for n in names]
for p in paths:
    git('add','--sparse','--',p)
assert set(git('-c','core.quotepath=false','diff','--cached','--name-only').splitlines())<=set(paths)
git('-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--cached','--check')
if git('diff','--cached','--name-only'):
    git('commit','-m','Add official Elsevier-based RA-Coreset SHBR manuscript framework')
git('push','origin','MLR')
head=git('rev-parse','HEAD'); remote=git('ls-remote','origin','refs/heads/MLR').split()[0]
assert head==remote
receipt=dict(status='PUSH_VERIFIED',commit=head,remote=remote,time=datetime.datetime.now().astimezone().isoformat(),files=len(paths))
(R/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
