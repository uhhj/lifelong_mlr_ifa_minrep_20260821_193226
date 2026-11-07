"""Publish only the reviewed report, verification and their source scripts."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]/'publish_mlr_branch'
DEST=REPO/'reports/debr_manuscript_review_20261107'
def git(*args):return subprocess.check_output(['git','-C',str(REPO),*args],text=True,encoding='utf8').strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
allowed={str((DEST/n).relative_to(REPO).as_posix()) for n in ['REPORT.md','VERIFICATION.json','verify_review.py','publish_review.py','PUBLICATION_MANIFEST.json']}
assert set(git('diff','--cached','--name-only').splitlines())<=allowed,'Unrelated staged changes'
assert json.loads((ROOT/'VERIFICATION.json').read_text(encoding='utf8'))['status']=='PASS_SELECTED_TABLE_NUMERIC_RECOMPUTATION'
names=['REPORT.md','VERIFICATION.json','verify_review.py','publish_review.py']
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
manifest={}
DEST.mkdir(parents=True,exist_ok=True)
for name in names:
    raw=(ROOT/name).read_bytes();assert len(raw)<100000
    assert not secret.search(raw.decode('utf8'))
    manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (DEST/name).write_bytes(raw)
(DEST/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8',newline='\n')
paths=[(DEST/n).relative_to(REPO).as_posix() for n in names+['PUBLICATION_MANIFEST.json']]
subprocess.run(['git','-C',str(REPO),'add','-f','--',*paths],check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run(['git','-C',str(REPO),'diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):
    subprocess.run(['git','-C',str(REPO),'commit','-q','-m','Review DEBR manuscript evidence and comparison framing'],check=True)
    subprocess.run(['git','-C',str(REPO),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0]
assert head==remote
receipt=dict(status='PUSH_VERIFIED',commit=head,remote=remote,files=len(paths),observed=datetime.datetime.now().astimezone().isoformat(),private_docx_published=False,full_manuscript_text_published=False)
(ROOT/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt,indent=2))
