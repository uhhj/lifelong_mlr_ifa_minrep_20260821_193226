from pathlib import Path
import datetime,hashlib,json,re,subprocess
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]/'publish_mlr_branch'
DEST=REPO/'reports/debr_B_refined_v2_20261112'
def git(*args):
    return subprocess.check_output(['git','-C',str(REPO),*args],text=True,encoding='utf8').strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only'),'Unrelated staged changes'
r=json.loads((HERE/'REVIEW.json').read_text())
assert r['status']=='PASS_VECTOR_RENDER_AND_VISUAL_REVIEW'
assert hashlib.sha256((HERE/'DEBR_B_refined.svg').read_bytes()).hexdigest()==r['svg_sha256']
assert hashlib.sha256((HERE/'DEBR_B_refined.png').read_bytes()).hexdigest()==r['png_sha256']
files=['svgkit.py','build.py','render.cjs','README.md','audit.py','REVIEW.json','RENDER_RECEIPT.json','publish.py','DEBR_B_refined.svg','DEBR_B_refined.png','DEBR_B_preview.png','DEBR_B_SVG_PNG_comparison.png']
manifest={}
for name in files:
    p=HERE/name;raw=p.read_bytes()
    assert len(raw)<16*1024**2
    if p.suffix in ['.py','.cjs','.md','.json','.svg']:
        assert not re.search(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}',raw.decode('utf8'))
    dest=DEST/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
    manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(DEST/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
paths=[(DEST/n).relative_to(REPO).as_posix() for n in files+['PUBLICATION_MANIFEST.json']]
subprocess.run(['git','-C',str(REPO),'add','-f','--',*paths],check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run(['git','-C',str(REPO),'-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol','diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):
    subprocess.run(['git','-C',str(REPO),'commit','-m','Refine selected DEBR B figure around dual-experience coordination'],check=True)
subprocess.run(['git','-C',str(REPO),'-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','push','origin','MLR'],check=True)
head=git('rev-parse','HEAD')
remote=git('-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/MLR').split()[0]
assert head==remote
receipt=dict(status='PUSH_VERIFIED',commit=head,remote=remote,files=len(paths),observed=datetime.datetime.now().astimezone().isoformat())
(HERE/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
