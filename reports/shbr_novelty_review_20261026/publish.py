"""Publish exact reviewed lightweight novelty-review files via publish_mlr_branch."""
from pathlib import Path
import datetime, hashlib, json, subprocess
from audit_review import inspect_text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1] / 'publish_mlr_branch'
DEST = REPO / 'reports/shbr_novelty_review_20261026'
URL = 'https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226.git'
NAMES = ['REPORT.md', 'robot_retention.md', 'success_replay.md', 'mixture_distillation.md',
         'LATEST_GOAL_EVIDENCE.md', 'LATEST_OBJECT_EVIDENCE.md', 'MECHANISM_EXPERIMENT_EVIDENCE.md',
         'LATEST_RESULTS_RECOMPUTED.json', 'LATEST_METRICS.csv', 'audit_latest_results.py',
         'SUBMISSION_STRATEGY_20261027.md', 'submission_sources_20261027.json',
         'root_sources.json', 'robot_retention_sources.json', 'success_replay_sources.json',
         'mixture_distillation_sources.json', 'REFERENCES.csv', 'QUERY_LOG.json',
         'LOCAL_METHOD_AUDIT.json', 'audit_review.py', 'publish.py']

def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), '-c', 'http.sslBackend=openssl',
                                    '-c', 'http.version=HTTP/1.1', *args], text=True, encoding='utf-8').strip()

def main():
    assert git('branch', '--show-current') == 'MLR'
    assert git('remote', 'get-url', 'origin') == URL
    assert not git('diff', '--cached', '--name-only'), 'Unrelated staged edits; do not touch them.'
    audit = json.loads((HERE / 'LOCAL_METHOD_AUDIT.json').read_text(encoding='utf-8'))
    for name, record in audit['review_artifacts'].items():
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest() == record['sha256']
    manifest = {}
    for name in NAMES:
        source = HERE / name
        inspect_text(source)
        raw = source.read_bytes()
        target = DEST / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        assert hashlib.sha256(target.read_bytes()).hexdigest() == digest
        manifest[name] = {'bytes': len(raw), 'sha256': digest}
    (DEST / 'PUBLICATION_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    (DEST / '.gitattributes').write_bytes(b'* -text -whitespace\n')
    paths = [(DEST / n).relative_to(REPO).as_posix() for n in NAMES + ['PUBLICATION_MANIFEST.json', '.gitattributes']]
    subprocess.run(['git', '-C', str(REPO), 'add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul'],
                   input=b'\0'.join(p.encode() for p in paths) + b'\0', check=True)
    assert set(git('diff', '--cached', '--name-only').splitlines()) <= set(paths)
    git('diff', '--cached', '--check')
    changed = bool(git('diff', '--cached', '--name-only'))
    if changed:
        git('commit', '-m', 'Document joint replay paper positioning and prioritized controlled experiments')
    head = git('rev-parse', 'HEAD')
    remote = git('ls-remote', 'origin', 'refs/heads/MLR').split()[0]
    if head != remote:
        git('push', 'origin', 'MLR')
    verified = git('ls-remote', 'origin', 'refs/heads/MLR').split()[0]
    assert head == verified
    receipt = {'status': 'PUSH_VERIFIED', 'commit': head, 'remote_sha': verified,
               'remote': URL, 'branch': 'MLR', 'files': len(paths), 'new_commit': changed,
               'time': datetime.datetime.now().astimezone().isoformat(),
               'scope': 'Literature/code read-only review, not experimental acceptance',
               'private_payload_published': False}
    (HERE / 'PUBLICATION_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == '__main__':
    main()
