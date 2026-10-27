"""Read-only method-source and report checks; no model/environment imports."""
from pathlib import Path
import csv, datetime, hashlib, json, re

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
METHOD = WORKSPACE / 'execution/20261025_ccbp_v4_full'
METHOD_NAMES = ['PROTOCOL.md', 'IMPLEMENTATION_NOTES.md', 'losses.py',
                'legacy_losses.py', 'memory.py', 'collect.py', 'updates.py']
DOCUMENTS = ['REPORT.md', 'robot_retention.md', 'success_replay.md',
             'mixture_distillation.md', 'LATEST_GOAL_EVIDENCE.md',
             'LATEST_OBJECT_EVIDENCE.md', 'MECHANISM_EXPERIMENT_EVIDENCE.md',
             'SUBMISSION_STRATEGY_20261027.md']
SOURCES = ['root_sources.json', 'robot_retention_sources.json',
           'success_replay_sources.json', 'mixture_distillation_sources.json',
           'submission_sources_20261027.json']
RESULTS = ['LATEST_RESULTS_RECOMPUTED.json', 'LATEST_METRICS.csv']

def fingerprint(path):
    data = path.read_bytes()
    return {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

def inspect_text(path):
    assert path.suffix in {'.md', '.json', '.csv', '.py'}
    raw = path.read_bytes()
    assert 0 < len(raw) < 2000000
    txt = raw.decode('utf-8-sig')
    assert not re.search(r'BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}', txt)
    assert not re.search(r'password\s*=\s*[\'\"][^\'\"]+[\'\"]', txt, re.I)
    return txt

def main():
    rows = []
    queries = []
    for name in SOURCES:
        doc = json.loads(inspect_text(HERE / name))
        assert isinstance(doc['sources'], list)
        for row in doc['sources']:
            url = row.get('url') or row.get('primary_url')
            if not url:
                alternatives = row.get('urls', [])
                url = alternatives[0] if isinstance(alternatives, list) and alternatives else ''
            rows.append({'source_file': name, 'id': row.get('id', ''),
                         'title': row.get('title', ''), 'url': url,
                         'evidence': row.get('evidence_level', row.get('confidence', 'see source record'))})
        queries.extend({'source_file': name, 'query': q}
                       for q in doc.get('queries', doc.get('search_queries', [])))
    for name in DOCUMENTS + RESULTS:
        inspect_text(HERE / name)
    latest = json.loads((HERE / RESULTS[0]).read_text(encoding='utf-8'))
    assert latest['status'].startswith('PASS'), latest['status']
    with (HERE / 'REFERENCES.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['source_file', 'id', 'title', 'url', 'evidence'])
        writer.writeheader()
        writer.writerows(rows)
    (HERE / 'QUERY_LOG.json').write_text(json.dumps({
        'status': 'ACTUAL_DOCUMENTED_QUERIES_NOT_DATABASE_EXHAUSTIVENESS',
        'queries': queries,
        'additional_search_families': 'See success_replay_sources.json; not counted as individual queries.'
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    out = {
        'status': 'PASS_LOCAL_REVIEW_ARTIFACT_INTEGRITY_NOT_EXPERIMENT_ACCEPTANCE',
        'observed_at': datetime.datetime.now().astimezone().isoformat(),
        'scientific_source_files_read_only': {str((METHOD / n).relative_to(WORKSPACE)): fingerprint(METHOD / n) for n in METHOD_NAMES},
        'additional_method_summary_read_only': {
            'execution/20261023_caet_em_v1/RA_ALIGN_METHOD_SUMMARY.md': fingerprint(WORKSPACE / 'execution/20261023_caet_em_v1/RA_ALIGN_METHOD_SUMMARY.md')
        },
        'review_artifacts': {n: fingerprint(HERE / n) for n in DOCUMENTS + SOURCES + RESULTS + ['REFERENCES.csv', 'QUERY_LOG.json', 'audit_latest_results.py']},
        'source_record_count_including_cross_report_duplicates': len(rows),
        'documented_query_count': len(queries),
        'checks': ['UTF-8 readable', 'bounded lightweight sizes', 'JSON syntax', 'credential-pattern scan', 'SHA256 bindings', 'Accepted GOAL/OBJECT metric recomputation PASS'],
        'scope_limits': ['Does not certify no undiscovered prior art.', 'Does not revalidate external papers experimentally.',
                         'No model loads, gradients, rollouts, server commands, or active experiment changes.',
                         'Ongoing experiments retain their existing separate status.']
    }
    (HERE / 'LOCAL_METHOD_AUDIT.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': out['status'], 'source_records': len(rows), 'queries': len(queries)}, ensure_ascii=False))

if __name__ == '__main__':
    main()
