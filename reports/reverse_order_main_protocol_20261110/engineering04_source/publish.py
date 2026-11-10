"""Review and publish this experiment's lightweight evidence to the MLR branch.

No training, evaluation, model loading, archive extraction, or scientific mutation
is performed. --review-only validates the source set without changing Git files.
"""
from pathlib import Path
import argparse
import ast
import datetime
import hashlib
import json
import re
import subprocess

R = Path(__file__).resolve().parent
G = R.parents[1] / 'publish_mlr_branch'
PREFIX = 'reports/reverse_order_main_protocol_20261110'
DESTINATION = 'https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
GIT = 'C:/Users/Lenovo/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
ALLOWED = {'.py', '.md', '.json', '.csv', '.yaml', '.yml', '.diff', '.sh'}
EXCLUDED_NAMES = {'DEPLOY_TEXT_BUNDLE.json', 'ARMED_DEPLOY_TEXT_BUNDLE.json',
                  'PUBLICATION_RECEIPT.json', 'PUBLICATION_MANIFEST.json',
                  'PUBLICATION_REVIEW.json'}
EXCLUDED_PARTS = {'__pycache__', '.git', '.session_control'}
MAX_FILE_BYTES = 16 * 1024**2
SECRET = re.compile(r'BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY|'
                    r'ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|'
                    r'hf_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|'
                    r'data:[^\s;]+;base64,[A-Za-z0-9+/=]{512,}')


def git(*args):
    return subprocess.check_output([GIT, '-C', str(G), *args],
                                   text=True, encoding='utf-8').strip()


def now():
    return datetime.datetime.now().astimezone().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-only', action='store_true')
    parser.add_argument('--offline-commit', action='store_true', help='Commit reviewed files locally after a recorded network failure; do not claim remote publication')
    parser.add_argument('--message', default='Publish main-table-protocol reverse-order MLR+IFA and DEBR evidence')
    args = parser.parse_args()
    assert git('branch', '--show-current') == 'MLR'
    assert git('remote', 'get-url', 'origin') == DESTINATION
    staged_before = git('diff', '--cached', '--name-only').splitlines()
    assert all(n.startswith(PREFIX + '/') for n in staged_before), 'Preserve unrelated staged edits'

    reviewed = {}
    payloads = {}
    for path in sorted(R.rglob('*')):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(R)
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if len(relative.parts)>1 and relative.parts[0]=='prepare_remote_20261110_231804' and relative.parts[1]=='diagnostics':
            # Exact transferred text is materialized once at its normal path;
            # keep duplicate transport envelopes local with their original SHA.
            evidence=json.loads((R/'prepare_remote_20261110_231804/EVIDENCE.json').read_text())
            diagnostic_files={k.replace('\\','/'):v for k,v in evidence['diagnostic_files'].items()}
            row=diagnostic_files[Path(*relative.parts[1:]).as_posix()]
            assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
            item=json.loads(path.read_text())
            assert hashlib.sha256((R/item['relative']).read_bytes()).hexdigest()==item['sha256']
            continue
        if relative.as_posix() == 'preflight_readonly_20261110_224104/EVIDENCE.json':
            # Full nested historical selection inventories are 28 MB; publish the
            # exact audit/hash and concise counts, not this repeated raw inventory.
            inventory_audit = json.loads((R/'PREFLIGHT_LOCAL_AUDIT.json').read_text(encoding='utf-8'))
            assert hashlib.sha256(path.read_bytes()).hexdigest() == inventory_audit['sha256']
            continue
        if path.name in EXCLUDED_NAMES or 'receipt' in path.name.lower():
            continue
        if path.suffix.lower() not in ALLOWED:
            continue
        raw = path.read_bytes()
        assert len(raw) <= MAX_FILE_BYTES, ('Oversized text requires separate review', str(relative), len(raw))
        text = raw.decode('utf-8-sig')
        assert '\x00' not in text and not SECRET.search(text), ('Secret/binary check failed', str(relative))
        if path.suffix == '.py':
            ast.parse(text, filename=str(relative), feature_version=(3, 8))
        if path.suffix == '.json':
            json.loads(text)
        name = relative.as_posix()
        reviewed[name] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        payloads[name] = raw
    assert reviewed, 'No reviewed files to publish'
    reviewed_paths = {PREFIX + '/' + name for name in reviewed}
    assert set(staged_before) <= reviewed_paths | {PREFIX + '/PUBLICATION_MANIFEST.json'}, 'Unreviewed staged path'
    review = dict(status='REVIEWED_LIGHTWEIGHT_TEXT_ONLY', observed=now(),
                  destination=DESTINATION, branch='MLR', prefix=PREFIX,
                  files=len(reviewed), bytes=sum(row['bytes'] for row in reviewed.values()),
                  files_sha256=reviewed,
                  exclusions=sorted(EXCLUDED_NAMES),
                  payload_policy='No checkpoints, weights, demonstrations, latent/buffer/RNG payloads, credentials or archives; metadata/hashes/outcomes allowed',
                  scientific_runs_started=0)
    (R / 'PUBLICATION_REVIEW.json').write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.review_only:
        print(json.dumps({k: v for k, v in review.items() if k != 'files_sha256'}, ensure_ascii=False))
        return

    head_before = git('rev-parse', 'HEAD')
    if args.offline_commit:
        assert (R / 'PUBLICATION_FAILURE.json').exists(), 'Record the real network failure first'
    remote_before = head_before if args.offline_commit else git('ls-remote', 'origin', 'refs/heads/MLR').split()[0]
    if head_before != remote_before:
        # An earlier interrupted publication may have committed without pushing.
        # Do not carry unrelated local commits to the remote as a side effect.
        subprocess.run([GIT, '-C', str(G), 'merge-base', '--is-ancestor',
                        remote_before, head_before], check=True)
        pending = git('log', '--format=', '--name-only', remote_before + '..' + head_before).splitlines()
        assert all(not n or n.startswith(PREFIX + '/') for n in pending), 'Unrelated unpublished commit requires separate review'
    destination = G / PREFIX
    for name, raw in payloads.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    manifest_name = PREFIX + '/PUBLICATION_MANIFEST.json'
    (G / manifest_name).write_text(json.dumps(review, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    paths = sorted(reviewed_paths | {manifest_name})
    subprocess.run([GIT, '-C', str(G), 'add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul'],
                   input=b'\0'.join(name.encode('utf-8') for name in paths) + b'\0', check=True)
    assert set(git('diff', '--cached', '--name-only').splitlines()) <= set(paths)
    subprocess.run([GIT, '-C', str(G), '-c',
                    'core.whitespace=blank-at-eol,space-before-tab,cr-at-eol,-blank-at-eof',
                    # Preserve inherited main-table source bytes and recorded SHA;
                    # its original trailing spaces are not newly authored prose.
                    'diff', '--cached', '--check', '--', PREFIX, ':(exclude)**/*.diff',
                    ':(exclude)**/source/**'], check=True)
    committed = bool(git('diff', '--cached', '--name-only'))
    if committed:
        subprocess.run([GIT, '-C', str(G), 'commit', '-q', '-m', args.message], check=True)
    head = git('rev-parse', 'HEAD')
    if args.offline_commit:
        receipt = dict(status='LOCAL_COMMIT_PENDING_REMOTE_PUBLICATION', observed=now(), commit=head,
                       destination=DESTINATION, branch='MLR', prefix=PREFIX, committed=committed,
                       files=len(paths), bytes=review['bytes'], remote_verified=False,
                       reason='Recorded TLS/network failure; no push or successful-publication claim')
        (R / 'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(receipt, ensure_ascii=False))
        return
    remote = git('ls-remote', 'origin', 'refs/heads/MLR').split()[0]
    if head != remote:
        subprocess.run([GIT, '-C', str(G), 'push', 'origin', 'MLR'], check=True)
        remote = git('ls-remote', 'origin', 'refs/heads/MLR').split()[0]
    assert head == remote, 'Remote commit verification failed'
    receipt = dict(status='PUSH_VERIFIED', observed=now(), commit=head, remote=remote,
                   destination=DESTINATION, branch='MLR', prefix=PREFIX,
                   committed=committed, files=len(paths), bytes=review['bytes'],
                   acceptance='Publication is not scientific acceptance; current RUN_STATUS applies')
    (R / 'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
