"""Publish reviewed source and metadata only to the standing MLR destination."""
from pathlib import Path
import json,hashlib,re,subprocess,datetime
R=Path(__file__).resolve().parent;repo=R.parents[1]/'publish_mlr_branch';dest=repo/'reports/gripper_repair_stageA_object_seed41'
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()
assert git('branch','--show-current')=='MLR'
assert git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only'),'Preserve unrelated staged changes'
allowed={'.py','.sh','.json','.jsonl','.md','.yaml','.yml','.txt','.patch','.log','.exit','.pid','.started','.sbatch','.csv'}
files={}
for p in R.rglob('*'):
 if not p.is_file() or '__pycache__' in p.parts or p.name.startswith('PUBLICATION_RECEIPT') or p.name=='WINDOW_OUTCOMES.json':continue
 if p.suffix not in allowed and p.name!='LICENSE':continue
 rel=p.relative_to(R).as_posix()
 if 'source' not in Path(rel).parts:assert not {'features','buffer','bindings','models','recovery','smoke'}&set(Path(rel).parts),rel
 else:assert p.suffix in {'.py','.sh','.md','.yaml','.yml','.txt','.json','.sbatch','.csv'} or p.name=='LICENSE',rel
 files[rel]=p
def inspect(v):
 if isinstance(v,dict):
  assert not {'password','private_key','last_action','latent_tensor'}&set(v)
  for x in v.values():inspect(x)
 elif isinstance(v,list):
  for x in v:inspect(x)
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
manifest={}
for n,p in files.items():
 raw=p.read_bytes();assert len(raw)<8*1024**2,n;txt=raw.decode('utf-8-sig');assert not secret.search(txt),n
 assert not re.search(r'password\s*=\s*[\x22\x27][^\x22\x27]+[\x22\x27]',txt,re.I),n
 if p.suffix=='.json':inspect(json.loads(txt))
 if p.suffix=='.jsonl':
  for l in txt.splitlines():inspect(json.loads(l))
 out=dest/n;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(raw);manifest[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(dest/'.gitattributes').write_bytes(b'* -text -whitespace\n');(dest/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
paths=[(dest/n).relative_to(repo).as_posix() for n in list(manifest)+['.gitattributes','PUBLICATION_MANIFEST.json']]
subprocess.run(['git','-C',str(repo),'add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b'\0'.join(p.encode('utf-8') for p in paths)+b'\0',check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run(['git','-C',str(repo),'diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):
 subprocess.run(['git','-C',str(repo),'commit','-q','-m','Complete gripper repair Stage A with held-out harms and native-input audit'],check=True)
 subprocess.run(['git','-C',str(repo),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0];assert head==remote
receipt=dict(status='PUSH_VERIFIED',commit=head,remote=remote,files=len(paths),bytes=sum(v['bytes'] for v in manifest.values()),experiment_status=json.loads((R/'RUN_STATUS.json').read_text())['status'],time=datetime.datetime.now().astimezone().isoformat())
(R/('PUBLICATION_RECEIPT_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S')+'.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
