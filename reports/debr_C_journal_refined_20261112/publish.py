"""Publish code and hashes only. All RGB-bearing artifacts remain local."""
from pathlib import Path
import hashlib,json,re,subprocess,datetime
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]/'publish_mlr_branch'
DEST=REPO/'reports/debr_C_journal_refined_20261112'
def git(*a):return subprocess.check_output(['git','-C',str(REPO),*a],text=True,encoding='utf8').strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only'),'Unrelated staged changes'
assert json.loads((HERE/'REVIEW.json').read_text(encoding='utf8'))['status']=='PASS_JOURNAL_FIGURE_REVIEW'
files=['README.md','ASSET_PROVENANCE.json','REVIEW.json','RENDER_RECEIPT.json','LOCAL_ARTIFACT_MANIFEST.json','svgkit.py','figkit.py','build.py','render.cjs','audit.py','publish.py','FIGURE_CAPTION.md','NOTATION_NOTE.md']
manifest={}
for name in files:
    p=HERE/name;raw=p.read_bytes();s=raw.decode('utf8')
    assert p.suffix in ['.md','.json','.py','.cjs'] and len(raw)<200000
    assert not re.search(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|[A-Za-z0-9+/]{1000,}={0,2}',s)
    d=DEST/name;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(raw)
    manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(DEST/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
paths=[(DEST/n).relative_to(REPO).as_posix() for n in files+['PUBLICATION_MANIFEST.json']]
subprocess.run(['git','-C',str(REPO),'add','-f','--',*paths],check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run(['git','-C',str(REPO),'-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol','diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):subprocess.run(['git','-C',str(REPO),'commit','-m','Refine selected DEBR C figure for journal layout and requested notation'],check=True)
subprocess.run(['git','-C',str(REPO),'-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/MLR').split()[0]
assert head==remote
r=dict(status='PUSH_VERIFIED_CODE_AND_HASHES_ONLY',commit=head,remote=remote,files=len(paths),rgb_artifacts_published=False,observed=datetime.datetime.now().astimezone().isoformat())
(HERE/'PUBLICATION_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
