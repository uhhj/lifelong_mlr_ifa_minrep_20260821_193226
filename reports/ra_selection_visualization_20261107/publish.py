"""Publish the exact reviewed lightweight figure package, never scientific payloads."""
from pathlib import Path
import json,hashlib,subprocess,re,datetime
R=Path(__file__).resolve().parent;W=R.parents[1];G=W/'publish_mlr_branch';D=G/'reports/ra_selection_visualization_20261107'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(G),*args],text=True,encoding='utf8').strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only')
assert json.loads((R/'VISUAL_REVIEW.json').read_text())['status']=='PASS_RENDERED_PDF_VISUAL_REVIEW'
assert json.loads((R/'RECOMPUTATION_AUDIT.json').read_text())['status']=='PASS_SOURCE_CSV_AND_PDF_CHECKS'
for p,h in json.loads((R/'SOURCE_MANIFEST.json').read_text()).items():assert sha(W/p)==h
for p,h in json.loads((R/'RECOMPUTATION_AUDIT.json').read_text())['implementation_sha256'].items():assert sha(W/p)==h
exports=json.loads((R/'EXPORT_AUDIT.json').read_text())['figures']
for x in exports:assert sha(R/x['file'])==x['sha256']
files=['make_figures.py','check_exports.py','publish.py','PROTOCOL.md','REPORT.md','CAPTIONS.md','SOURCE_MANIFEST.json','AUDIT.json','EXPORT_AUDIT.json','RECOMPUTATION_AUDIT.json','VISUAL_REVIEW.json','TRANSPORT_ATTEMPT.json']
files += ['data/'+x for x in ['all_cases_original_distance.csv','all_cases_demonstration_counts.csv','illustrated_window_membership.csv','illustrated_trajectory_coverage.csv']]
files += [x['file'] for x in exports]
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
manifest={}
for rel in files:
    p=R/rel;b=p.read_bytes();assert len(b)<8*1024**2
    assert p.suffix in {'.py','.md','.json','.csv','.pdf','.svg','.png'}
    if p.suffix in {'.py','.md','.json','.csv','.svg'}:assert not secret.search(b.decode('utf8'))
    if p.suffix=='.pdf':assert b.startswith(b'%PDF')
    if p.suffix=='.png':assert b.startswith(b'\x89PNG')
    if p.suffix=='.json':json.loads(b)
    q=D/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
    manifest[rel]=dict(bytes=len(b),sha256=sha(p))
(D/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
paths=[(D/p).relative_to(G).as_posix() for p in files+['PUBLICATION_MANIFEST.json']]
subprocess.run(['git','-C',str(G),'add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b'\0'.join(p.encode() for p in paths)+b'\0',check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run(['git','-C',str(G),'-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol','diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):
    subprocess.run(['git','-C',str(G),'commit','-q','-m','Visualize archived RA selection membership and original-distance tradeoffs'],check=True)
    subprocess.run(['git','-C',str(G),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0];assert head==remote
receipt=dict(status='PUSH_VERIFIED',head=head,remote=remote,files=len(paths),bytes=sum(x['bytes'] for x in manifest.values()),observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
(R/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt,indent=2))
