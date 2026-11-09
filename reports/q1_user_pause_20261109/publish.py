from pathlib import Path
import ast,json,re,hashlib,subprocess,datetime
R=Path(__file__).resolve().parent;G=R.parents[1]/'publish_mlr_branch';PREFIX='reports/q1_user_pause_20261109'
GIT='C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
def git(*args):return subprocess.check_output([GIT,'-C',str(G),*args],text=True,encoding='utf8').strip()
assert git('branch','--show-current')=='MLR' and git('remote','get-url','origin')=='https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only')
secret=re.compile(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}')
manifest={}
def add(p,n):
 raw=p.read_bytes();assert len(raw)<4*1024**2;s=raw.decode('utf-8-sig');assert not secret.search(s)
 if p.suffix=='.json':json.loads(s)
 if p.suffix=='.py':ast.parse(s)
 d=G/n;d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(raw);manifest[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for p in sorted(R.rglob('*')):
 if p.is_file() and p.suffix in ['.py','.json','.md'] and p.name not in ['PUBLICATION_RECEIPT.json','PUBLICATION_MANIFEST.json']:add(p,PREFIX+'/'+p.relative_to(R).as_posix())
for local,prefix in [('20261109_support_calibration_rerun','support_calibration_rerun_20261109'),('20261109_mlr_ifa_align_baseline','mlr_ifa_align_baseline_20261109')]:
 for name in ['MONITOR_HANDOFF.md','USER_PAUSE_STATUS.json']:add(R.parent/local/name,'reports/'+prefix+'/'+name)
n=PREFIX+'/PUBLICATION_MANIFEST.json';(G/n).write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8');paths=list(manifest)+[n]
subprocess.run([GIT,'-C',str(G),'add','-f','--pathspec-from-file=-','--pathspec-file-nul'],input=b'\0'.join(x.encode() for x in paths)+b'\0',check=True)
assert set(git('diff','--cached','--name-only').splitlines())<=set(paths)
subprocess.run([GIT,'-C',str(G),'-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof','diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):
 subprocess.run([GIT,'-C',str(G),'commit','-q','-m','Pause Q1 experiment by user request for storage expansion; preserve live state'],check=True)
 subprocess.run([GIT,'-C',str(G),'push','origin','MLR'],check=True)
head=git('rev-parse','HEAD');remote=git('ls-remote','origin','refs/heads/MLR').split()[0];assert head==remote
out=dict(status='PUSH_VERIFIED',commit=head,remote=remote,observed=datetime.datetime.now().astimezone().isoformat(),files=len(paths),bytes=sum(x['bytes'] for x in manifest.values()))
(R/'PUBLICATION_RECEIPT.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf8');print(json.dumps(out))
