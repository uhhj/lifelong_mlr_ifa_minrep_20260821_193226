"""Publish only the reviewed figure package to the authorized MLR checkout."""
from pathlib import Path
import datetime
import hashlib
import json
import re
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / 'publish_mlr_branch'
REL = 'reports/stage_success_delta_curves_20261117'
DEST = REPO / REL
GIT = str(Path('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-611-0_26.1112.611.0_x64__3k8sg7r9htsxt/dependencies/native/git/cmd/git.exe'))
assert Path(GIT).is_file()
assert GIT

def git(*args):
    return subprocess.check_output([GIT,'-C',str(REPO),*args],text=True,encoding='utf8').strip()

assert git('branch','--show-current') == 'MLR'
assert git('remote','get-url','origin') == 'https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert not git('diff','--cached','--name-only'), 'Unrelated staged edits must be preserved.'
audit = json.loads((HERE/'DATA_LAYOUT_AUDIT.json').read_text(encoding='utf8'))
assert audit['status'] == 'PASS_DATA_AND_LAYOUT'
for rel,h in json.loads((HERE/'SOURCE_MANIFEST.json').read_text(encoding='utf8')).items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == h
files = ['make_figure.py','publish.py','README.md','SOURCE_MANIFEST.json','DATA_LAYOUT_AUDIT.json',
    'figures/preview.png',*[f'figures/DEBR_stage_success_delta.{e}' for e in ['png','pdf','svg']],
    *[f'data/{n}.csv' for n in ['delta_per_seed','delta_summary']]]
manifest = {}
for rel in files:
    p = HERE / rel
    raw = p.read_bytes()
    assert len(raw) < 8*1024**2
    if p.suffix in ['.py','.md','.json','.svg','.csv']:
        assert not re.search(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}',raw.decode('utf8'))
    dest = DEST / rel
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_bytes(raw)
    manifest[rel] = dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(DEST/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
paths = [f'{REL}/{p}' for p in files+['PUBLICATION_MANIFEST.json']]
subprocess.run([GIT,'-C',str(REPO),'add','-f','--',*paths],check=True)
assert set(git('diff','--cached','--name-only').splitlines()) <= set(paths)
subprocess.run([GIT,'-C',str(REPO),'-c','core.whitespace=blank-at-eol,space-before-tab,cr-at-eol','diff','--cached','--check'],check=True)
if git('diff','--cached','--name-only'):
    subprocess.run([GIT,'-C',str(REPO),'commit','-m','Add stage-wise success-rate gains relative to MLR plus IFA'],check=True)
subprocess.run([GIT,'-C',str(REPO),'-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','push','origin','MLR'],check=True)
head = git('rev-parse','HEAD')
remote = git('-c','http.sslBackend=openssl','-c','http.version=HTTP/1.1','ls-remote','origin','refs/heads/MLR').split()[0]
assert head == remote
receipt = dict(status='PUSH_VERIFIED',commit=head,remote=remote,files=len(paths),
    destination=REL,observed=datetime.datetime.now().astimezone().isoformat())
(HERE/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt))
