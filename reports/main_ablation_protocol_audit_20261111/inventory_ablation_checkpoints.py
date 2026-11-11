"""Bind the component table's exact checkpoint paths and stat them over SSH."""
import ast
import json
import re
import sys
from pathlib import Path, PurePosixPath

R = Path(__file__).resolve().parent
E = R.parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stage_rows():
    audit = read(R / 'AUDIT.json')
    preflight = E / '20261110_reverse_order_main_protocol/preflight_readonly.py'
    parent_map = ast.literal_eval(ast.parse(preflight.read_text(encoding='utf-8')).body[0].value)
    rows = []
    methods = ('MLR_IFA', 'RA', 'C_normal', 'Q', 'S_latest')
    for method in methods:
        for suite in ('OBJECT', 'GOAL'):
            for seed in (41, 42, 43):
                case = '%s/%s/%d' % (method, suite, seed)
                results = audit['cases'][case]
                source = results['outcome_sources'][0]
                if method == 'MLR_IFA':
                    evidence = Path(source).parents[1] / 'TRAINING_COMPLETE.json'
                    directory = PurePosixPath(parent_map['%s_%d' % (suite, seed)]['main_mlr_parent']) / 'random/models'
                    models = {str(directory / k): v for k, v in read(evidence)['checkpoints'].items()}
                elif method == 'RA':
                    evidence = Path(source).parents[3] / 'current/CONTRACT.json'
                    models = {k: v for k, v in read(evidence)['files'].items()
                              if re.search(r'/ra_coreset/models/mlrifa_[6-9]_model_ep99\.pth$', k)}
                else:
                    name, light_key = source.split('#light/', 1)
                    evidence = Path(name)
                    final = read(evidence)
                    prefix = light_key[:-len('OUTCOMES.json')]
                    arm = {'C_normal': 'C', 'Q': 'Q', 'S_latest': 'S'}[method]
                    models = {}
                    for stage in range(6, 10):
                        stage_key = '%sSTAGE_%s_%d.json' % (prefix, arm, stage)
                        record = json.loads(final['light'][stage_key]['text'])
                        assert record['stage'] == stage and record['epochs'] == 100
                        models[record['model']] = record['sha256']
                assert len(models) == 4, case
                assert all(re.fullmatch(r'[a-f0-9]{64}', s) for s in models.values())
                for stage, (path, expected) in zip(range(6, 10), sorted(models.items(), key=lambda p: PurePosixPath(p[0]).name)):
                    assert PurePosixPath(path).name in ('stage%d.pth' % stage, 'mlrifa_%d_model_ep99.pth' % stage), (case, stage, path)
                    rows.append(dict(case=case, stage=stage, path=path, recorded_sha256=expected,
                                     source=evidence.resolve().relative_to(E.parent).as_posix(), kind='stage_model'))
                if method in ('C_normal', 'Q', 'S_latest'):
                    last = rows[-1]
                    rows.append(dict(last, path=str(PurePosixPath(last['path']).with_name('recovery.pth')),
                                     kind='final_recovery', stage_model_path=last['path']))
    assert len(rows) == 138 and len({r['path'] for r in rows}) == 138
    return rows


REMOTE = '''
import json
import stat
import time
from pathlib import Path

rows = []
for source in EXPECTED:
    row = dict(source)
    try:
        info = Path(row['path']).stat()
        row.update(present=stat.S_ISREG(info.st_mode) and info.st_size > 0,
                   bytes=info.st_size, modified=info.st_mtime, device=info.st_dev, inode=info.st_ino)
        if row['kind'] == 'final_recovery':
            final = Path(row['stage_model_path']).stat()
            row['same_inode_as_stage9'] = (info.st_dev, info.st_ino) == (final.st_dev, final.st_ino)
    except OSError as error:
        row.update(present=False, stat_error=type(error).__name__)
    rows.append(row)
counts = {}
for row in rows:
    method = row['case'].split('/')[0]
    key = method + '/' + row['kind']
    counts.setdefault(key, {'checked': 0, 'present': 0})
    counts[key]['checked'] += 1
    counts[key]['present'] += int(row['present'])
print(json.dumps(dict(status='READONLY_CHECKPOINT_FILE_INVENTORY', observed=time.time(),
                      counts=counts, files=rows, checkpoint_payload_bytes_read=0,
                      new_model_load=0, new_optimizer=0, new_environments=0, server_writes=0)))
'''


def main():
    rows = stage_rows()
    script = R / 'checkpoint_stat_readonly.py'
    script.write_text('EXPECTED = ' + repr(rows) + '\n' + REMOTE, encoding='utf-8')
    sys.path.insert(0, str(E / '20261027_support_gate_forks'))
    import remote
    from credential_store import load
    remote.R = R
    secret = load()
    assert secret
    remote.getpass.getpass = lambda prompt: secret
    try:
        remote.run(script.name)
    finally:
        secret = None
        remote.getpass.getpass = lambda prompt: None


if __name__ == '__main__':
    main()
