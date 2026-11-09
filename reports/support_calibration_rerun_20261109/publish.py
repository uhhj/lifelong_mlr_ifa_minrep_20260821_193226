"""Publish exact reviewed lightweight files only, preserving unrelated changes."""
from pathlib import Path
import ast,datetime,hashlib,json,re,subprocess
R=Path(__file__).resolve().parent;G=R.parents[1]/'publish_mlr_branch';PREFIX='reports/support_calibration_rerun_20261109';D=G/PREFIX
GIT='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
def git(*a):return subprocess.check_output([GIT,'-C',str(G),*a],text=True,encoding='utf8').strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert all(n.startswith(PREFIX+'/') for n in git('diff','--cached','--name-only').splitlines())
review=json.loads((R/'UPLOAD_REVIEW.json').read_text(encoding='utf8'))
bundle=json.loads((R/'DEPLOY_TEXT_BUNDLE.json').read_text(encoding='utf8'))
assert all(hashlib.sha256(bundle[n].encode()).hexdigest()==v['sha256'] for n,v in review['files'].items())
assert all((R/n).read_text(encoding='utf-8-sig')==txt for n,txt in bundle.items() if n not in ['PREDEPLOY.json','INPUT_BINDING.json'])
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
manifest={};exclude={'DEPLOY_TEXT_BUNDLE.json','PUBLICATION_MANIFEST.json','PUBLICATION_RECEIPT.json','archive_cleanup.py'}
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.suffix not in ['.py','.json','.md','.diff']:continue
 if p.name in exclude or 'light' in p.relative_to(R).parts:continue
 raw=p.read_bytes();assert len(raw)<16*1024**2,p
 txt=raw.decode('utf-8-sig');assert not secret.search(txt),p
 if p.suffix=='.json':json.loads(txt)
 if p.suffix=='.py':ast.parse(txt)
 rel=p.relative_to(R).as_posix();manifest[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 dest=D/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
(D/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
paths=[PREFIX+'/'+n for n in [*manifest,'PUBLICATION_MANIFEST.json']]
subprocess.run([GIT,'-C',str(G),'add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b'\0'.join(n.encode() for n in paths)+b'\0',check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run([GIT,'-C',str(G),'-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check','--','.',':(exclude)**/*.diff'],check=True)
if git('diff','--cached','--name-only'):
 subprocess.run([GIT,'-C',str(G),'commit','-q','-m','Q1 rerun: reviewed six-run startup and obsolete diagnostic cleanup evidence'],check=True)
 subprocess.run([GIT,'-C',str(G),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0];assert head==remote
receipt=dict(status='PUSH_VERIFIED',commit=head,remote=remote,files=len(paths),bytes=sum(v['bytes'] for v in manifest.values()),observed=datetime.datetime.now().astimezone().isoformat())
(R/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt))
