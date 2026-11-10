"""Publish only the reviewed prepared code, metadata and task handoff."""
from pathlib import Path
import ast,datetime,hashlib,json,re,subprocess
R=Path(__file__).resolve().parent;W=R.parents[1];G=W/'publish_mlr_branch'
PREFIX='reports/mlr_ifa_align_baseline_20261109';OLDPREFIX='reports/support_calibration_rerun_20261109'
GIT='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
def git(*a):return subprocess.check_output([GIT,'-C',str(G),*a],text=True,encoding='utf8').strip()
def sha(x):return hashlib.sha256(x).hexdigest()
assert git('branch','--show-current')=='MLR' and git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert all(n.startswith(PREFIX+'/') or n in {OLDPREFIX+'/'+x for x in ['AUTHORIZED_MLR_IFA_ALIGN_FOLLOWUP.json','PREPARED_ALIGN_FOLLOWUP.json','MONITOR_HANDOFF.md']} for n in git('diff','--cached','--name-only').splitlines()),'Preserve unrelated staged edits'
review=json.loads((R/'UPLOAD_REVIEW.json').read_text(encoding='utf8'));bundle=json.loads((R/'DEPLOY_TEXT_BUNDLE.json').read_text(encoding='utf8'))
for n,v in review['files_sha256'].items():
 assert sha(bundle[n].encode())==v['sha256']
 if n=='CONTRACT.json':p=R/'PREPARED_CONTRACT.json'
 elif n.startswith('private/'):p=R/'cases'/n[len('private/'):]
 else:p=R/n
 assert p.read_text(encoding='utf-8-sig')==bundle[n],n
assert json.loads((R/'PREPARATION_VALIDATION.json').read_text(encoding='utf8'))['contract']==review['contract_sha256']
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}')
exclude={'DEPLOY_TEXT_BUNDLE.json','ARMED_DEPLOY_TEXT_BUNDLE.json','PUBLICATION_RECEIPT.json','PUBLICATION_MANIFEST.json'}
files=[]
for p in R.iterdir():
 if p.is_file() and p.name not in exclude and p.suffix in ['.py','.md','.json']:files.append(p)
for folder in ['cases','templates','review','capacity_readonly_20261109_165910','engineering_completion_gate_20261110_v2','engineering_environment_20261110_v3','engineering_startup_20261110_v4','engineering_assets_resume_20261110_v1','engineering_assets_resume_20261110_v2']:
 files += [p for p in (R/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.yaml','.json','.diff'] and '__pycache__' not in p.parts and p.name not in exclude]
manifest={};paths=[]
def add(p,dest):
 raw=p.read_bytes();assert len(raw)<8*1024**2,p;text=raw.decode('utf-8-sig');assert not secret.search(text),p
 if p.suffix=='.py':ast.parse(text)
 if p.suffix=='.json':json.loads(text)
 target=G/dest;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 manifest[dest]=dict(bytes=len(raw),sha256=sha(raw));paths.append(dest)
for p in sorted(set(files)):add(p,PREFIX+'/'+p.relative_to(R).as_posix())
for name in ['AUTHORIZED_MLR_IFA_ALIGN_FOLLOWUP.json','PREPARED_ALIGN_FOLLOWUP.json','MONITOR_HANDOFF.md']:add(R.parent/'20261109_support_calibration_rerun'/name,OLDPREFIX+'/'+name)
mname=PREFIX+'/PUBLICATION_MANIFEST.json';(G/mname).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8');paths.append(mname)
subprocess.run([GIT,'-C',str(G),'add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b'\0'.join(n.encode() for n in paths)+b'\0',check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run([GIT,'-C',str(G),'-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check','--',PREFIX,OLDPREFIX,':(exclude)**/*.diff',':(exclude)'+PREFIX+'/cases/*/source/libero/**',':(exclude)'+PREFIX+'/cases/*/source/coreset_exp/eval_metric.py',':(exclude)'+PREFIX+'/cases/*/source/coreset_exp/reference_train.py'],check=True)
if git('diff','--cached','--name-only'):
 subprocess.run([GIT,'-C',str(G),'commit','-q','-m','Repair ALIGN assets and spawn entry with exact RNG guard and actual resumed rollouts'],check=True)
 subprocess.run([GIT,'-C',str(G),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0];assert head==remote
out=dict(status='PUSH_VERIFIED',observed=datetime.datetime.now().astimezone().isoformat(),commit=head,remote=remote,files=len(paths),bytes=sum(x['bytes'] for x in manifest.values()),scope='Reviewed ALIGN code and PARTIAL stopped evaluation evidence; failure attempts preserved; no tensor/weight/demo/cache/RNG payload')
(R/'PUBLICATION_RECEIPT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))
