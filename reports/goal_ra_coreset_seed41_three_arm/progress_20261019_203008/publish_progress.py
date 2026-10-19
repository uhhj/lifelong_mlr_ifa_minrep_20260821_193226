"""Review and publish only one progress package; retain original launch evidence."""
from pathlib import Path
import json,hashlib,re,subprocess,sys,datetime
R=Path(__file__).resolve().parent;D=R/sys.argv[1];repo=R.parents[1]/'publish_mlr_branch';dest=repo/'reports/goal_ra_coreset_seed41_three_arm'/D.name
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only'),'Preserve unrelated staged changes'
files={p.relative_to(D).as_posix():p for p in D.rglob('*') if p.is_file()}
for n in ['collect_readonly_progress.py','collect_ra_progress.py','audit_progress.py','audit_baseline_complete.py','publish_progress.py','RUN_STATUS.json','AUTOMATION.json']:files[n]=R/n
allowed={'.py','.json','.jsonl','.md','.log','.pid','.started','.exit'}
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
def inspect(v):
 if isinstance(v,dict):
  assert not {'last_action','password','private_key','latent_tensor'}&set(v)
  for x in v.values():inspect(x)
 elif isinstance(v,list):
  for x in v:inspect(x)
m={}
for n,p in files.items():
 assert p.suffix in allowed and not {'features','buffer','bindings','recovery','smoke','models'}&set(Path(n).parts)
 raw=p.read_bytes();assert len(raw)<8*1024**2;txt=raw.decode('utf-8-sig');assert not secret.search(txt)
 assert not re.search(r'password\s*=\s*[\x22\x27][^\x22\x27]+[\x22\x27]',txt,re.I)
 if p.suffix=='.json':inspect(json.loads(txt))
 if p.suffix=='.jsonl':
  for l in txt.splitlines():inspect(json.loads(l))
 out=dest/n;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw)
 m[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(dest/'.gitattributes').write_bytes(b'* -text -whitespace\n')
(dest/'PUBLICATION_MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
paths=[(dest/n).relative_to(repo).as_posix() for n in list(m)+['.gitattributes','PUBLICATION_MANIFEST.json']]
subprocess.run(['git','-C',str(repo),'add','-f','--',*paths],check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run(['git','-C',str(repo),'diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):
 subprocess.run(['git','-C',str(repo),'commit','-q','-m','Record GOAL seed41 baseline training complete and partial evaluation audit'],check=True)
 subprocess.run(['git','-C',str(repo),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0];assert head==remote
receipt=dict(status='PUSH_VERIFIED',time=datetime.datetime.now().astimezone().isoformat(),commit=head,remote=remote,files=len(paths),bytes=sum(v['bytes'] for v in m.values()),experiment_status='PARTIAL',destination=str(dest))
(R/('PUBLICATION_RECEIPT_'+D.name+'.json')).write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt,indent=2))
