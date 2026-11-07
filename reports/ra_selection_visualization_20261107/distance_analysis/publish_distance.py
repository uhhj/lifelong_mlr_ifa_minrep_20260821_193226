"""Publish reviewed code, rendered figures and scalar ECDFs; no latent/coordinate payloads."""
from pathlib import Path
import json,hashlib,subprocess,re,datetime
R=Path(__file__).resolve().parent;W=R.parents[1];G=W/'publish_mlr_branch';D=G/'reports/ra_selection_visualization_20261107/distance_analysis'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def git(*args):return subprocess.check_output(['git','-C',str(G),*args],text=True,encoding='utf8').strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only')
assert json.loads((R/'DISTANCE_VISUAL_REVIEW.json').read_text())['status']=='PASS_RENDERED_PDF_VISUAL_REVIEW'
assert json.loads((R/'DISTANCE_RECOMPUTATION_AUDIT.json').read_text())['status']=='PASS_REMOTE_HASHES_SCALARS_ECDF_AND_PDF_EXPORT'
bind=json.loads((R/'DISTANCE_LOCAL_BINDING.json').read_text());assert sha(R/bind['directory']/'EVIDENCE.json')==bind['evidence_sha256']
assert sha(R/'distance_readonly.py')==bind['request_sha256'] and sha(R/'DISTANCE_PROTOCOL.md')==bind['source_protocol_sha256']
exports=json.loads((R/'DISTANCE_EXPORT_AUDIT.json').read_text())['figures']
for x in exports:assert sha(R/x['file'])==x['sha256']
files=['build_distance_request.py','distance_body.py','distance_readonly.py','plot_distances.py','check_distance_exports.py','publish_distance.py','DISTANCE_PROTOCOL.md','DISTANCE_REPORT.md','DISTANCE_CAPTIONS.md','DISTANCE_SUMMARY.json','DISTANCE_LOCAL_BINDING.json','DISTANCE_EXPORT_AUDIT.json','DISTANCE_RECOMPUTATION_AUDIT.json','DISTANCE_VISUAL_REVIEW.json']
files += ['data_distance/'+x for x in ['nearest_distance_summary.csv','nearest_distance_ecdf.csv','all_cases_trajectory_allocation.csv','REMOTE_AUDIT_SUMMARY.json']]
files += [x['file'] for x in exports]
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
manifest={}
for rel in files:
    assert not any(part in ['diagnostics','review_distance'] for part in Path(rel).parts)
    p=R/rel;b=p.read_bytes();assert len(b)<12*1024**2
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
    subprocess.run(['git','-C',str(G),'commit','-q','-m','Add CPU-verified original-distance ECDFs and shared RA selection projections'],check=True)
subprocess.run(['git','-C',str(G),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0];assert head==remote
receipt=dict(status='PUSH_VERIFIED',head=head,remote=remote,files=len(paths),bytes=sum(x['bytes'] for x in manifest.values()),observed=datetime.datetime.now().astimezone().isoformat())
(R/'DISTANCE_PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt,indent=2))
