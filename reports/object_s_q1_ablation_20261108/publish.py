"""Publish reviewed code, reports, hashes and lightweight evidence only."""
from pathlib import Path
import datetime,hashlib,json,re,subprocess
R=Path(__file__).resolve().parent;W=R.parents[1];G=W/'publish_mlr_branch';D=G/'reports/object_s_q1_ablation_20261108'
def git(*args):return subprocess.check_output(['git','-C',str(G),*args],text=True,encoding='utf8').strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
staged=git('diff','--cached','--name-only').splitlines()
assert all(n.startswith('reports/object_s_q1_ablation_20261108/') for n in staged),'Unrelated staged changes'
review=json.loads((R/'PREDEPLOY_REVIEW.json').read_text(encoding='utf8'))
assert review['status']=='PASS_STATIC_REVIEW_NOT_SCIENTIFIC_ACCEPTANCE'
assert all(hashlib.sha256((R/n).read_bytes()).hexdigest()==h for n,h in review['files'].items())
binding=json.loads((R/'S_ACCEPTANCE.json').read_text(encoding='utf8'))['sources']['OBJECT']
for name,key in [('EVIDENCE.json','evidence_sha256'),('LOCAL_FINAL_AUDIT.json','audit_sha256')]:assert hashlib.sha256((Path(binding['final_path'])/name).read_bytes()).hexdigest()==binding[key]
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
manifest={}
size_review_path=R/'PUBLICATION_SIZE_REVIEW.json'
size_exceptions={}
if size_review_path.exists():
 size_review=json.loads(size_review_path.read_text(encoding='utf8'))
 assert size_review['status']=='REVIEWED_SCALAR_EVIDENCE_SIZE_EXCEPTION'
 size_exceptions=size_review['files']
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.suffix not in ['.py','.json','.md','.csv','.diff']:continue
 if p.name in ['DEPLOY_TEXT_BUNDLE.json','PUBLICATION_MANIFEST.json'] or p.name.startswith('PUBLICATION_RECEIPT'):continue
 if 'light' in p.relative_to(R).parts:continue
 raw=p.read_bytes()
 if len(raw)>=16*1024**2:
  exact=size_exceptions.get(p.relative_to(R).as_posix())
  assert exact and len(raw)==exact['bytes'] and hashlib.sha256(raw).hexdigest()==exact['sha256']
 txt=raw.decode('utf-8-sig');assert not secret.search(txt),p
 if p.suffix=='.json':json.loads(txt)
 rel=p.relative_to(R).as_posix();dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 manifest[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(D/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
paths=[(D/n).relative_to(G).as_posix() for n in [*manifest,'PUBLICATION_MANIFEST.json']]
subprocess.run(['git','-C',str(G),'add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b'\0'.join(n.encode() for n in paths)+b'\0',check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
# Preserve frozen bytes; unified-diff context blank lines and inherited blank EOF
# are intentional evidence, not code edits. Check remaining files for real whitespace errors.
subprocess.run(['git','-C',str(G),'-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check','--','.',':(exclude)**/*.diff'],check=True)
if git('diff','--cached','--name-only'):
 subprocess.run(['git','-C',str(G),'commit','-q','-m','OBJECT full S fixed-support ablation: reviewed source and actual partial evidence'],check=True)
 subprocess.run(['git','-C',str(G),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0];assert head==remote
receipt=dict(status='PUSH_VERIFIED',commit=head,remote=remote,files=len(paths),bytes=sum(v['bytes'] for v in manifest.values()),observed=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat())
(R/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt))
