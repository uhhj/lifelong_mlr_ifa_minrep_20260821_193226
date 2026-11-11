from pathlib import Path
import datetime
import hashlib
import json
import re
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / 'publish_mlr_branch'
DEST = REPO / 'reports/manuscript_figure_review_20261111'
def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True, encoding='utf8').strip()
assert git('branch', '--show-current') == 'MLR'
assert git('remote', 'get-url', 'origin') == 'https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
assert all(p.startswith('reports/manuscript_figure_review_20261111/') for p in git('diff', '--cached', '--name-only').splitlines())
files = ['inspect_doc.py', 'prepare_publication.py', 'publish.py', 'REPORT.md', 'REVIEW_AUDIT.json']
manifest = {}
for rel in files:
    p = HERE / rel
    raw = p.read_bytes()
    assert len(raw) < 16*1024**2
    if p.suffix in ['.py', '.md', '.json', '.svg']:
        assert not re.search(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}', raw.decode('utf8'))
    dest = DEST / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    manifest[rel] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
(DEST/'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
paths = [(DEST/p).relative_to(REPO).as_posix() for p in files+['PUBLICATION_MANIFEST.json']]
subprocess.run(['git', '-C', str(REPO), 'add', '-f', '--', *paths], check=True)
assert set(git('diff', '--cached', '--name-only').splitlines()) <= set(paths)
subprocess.run(['git', '-C', str(REPO), '-c', 'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol', 'diff', '--cached', '--check'], check=True)
if git('diff', '--cached', '--name-only'):
    subprocess.run(['git', '-C', str(REPO), 'commit', '-m', 'Review DEBR manuscript figure needs and prioritize framework and learning curves'], check=True)
subprocess.run(['git', '-C', str(REPO), '-c', 'http.sslBackend=openssl', '-c', 'http.version=HTTP/1.1', 'push', 'origin', 'MLR'], check=True)
head = git('rev-parse', 'HEAD')
remote = git('-c', 'http.sslBackend=openssl', '-c', 'http.version=HTTP/1.1', 'ls-remote', 'origin', 'refs/heads/MLR').split()[0]
assert head == remote
receipt = dict(status='PUSH_VERIFIED', commit=head, remote=remote, files=len(paths), observed=datetime.datetime.now().astimezone().isoformat())
(HERE/'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n')
print(json.dumps(receipt))
