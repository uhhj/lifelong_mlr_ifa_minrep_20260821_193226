"""Build a source-bound historical timeline from existing lightweight records only."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import subprocess

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
REPO = WORK/'publish_mlr_branch'
PARTS = ['part_early.json', 'part_align.json', 'part_shbr.json', 'part_late.json']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), '-c', 'http.sslBackend=openssl',
                                    '-c', 'http.version=HTTP/1.1', *args])


def clean(text):
    return str(text).replace('|', '\\|').replace('\n', '<br>')


def main():
    rows = []
    for name in PARTS:
        part = json.loads((HERE/name).read_text(encoding='utf-8-sig'))
        assert isinstance(part, list), name
        rows.extend(part)
    history = json.loads((HERE/'HISTORICAL_COMMIT.json').read_text(encoding='utf-8'))
    assert git('show', '-s', '--format=%cI', history['commit']).decode().strip() == history['commit_time']
    for path, item in history['files'].items():
        raw = git('show', history['commit']+':'+path)
        assert sha(raw) == item['sha256'] and len(raw) == item['bytes']
    rows.append(dict(id='202608_first_reproduction', date_label='2026-08（8月22日17:46:03已提交结果）',
                     start_time_cst=None, start_basis='实际训练开始日志未归档；8月22日为Git结果提交时间，不是启动时间',
                     method='最早MLR / MLR+IFA最小复现', suite_seeds='LIBERO-OBJECT，seed42',
                     change='从官方预训练模型复现四阶段持续模仿学习；比较有无IFA；按当前任务测试成功率选择70–95轮候选checkpoint',
                     result='两组各4×100轮已完成；MLR AUC70.729%，MLR+IFA AUC71.875%。该早期选模/状态/IFA协议与后续固定ep99协议不同。',
                     status='HISTORICAL_COMPLETED_RESULT_ARCHIVED', role='mainline',
                     sources=['execution/20261031_experiment_timeline/HISTORICAL_COMMIT.json']))
    required = {'id', 'date_label', 'start_time_cst', 'start_basis', 'method', 'suite_seeds', 'change', 'result', 'status', 'role', 'sources'}
    seen, bindings = set(), {}
    for row in rows:
        assert required <= row.keys(), row
        assert row['id'] not in seen, row['id']
        seen.add(row['id'])
        assert row['role'] in ['mainline', 'branch', 'audit']
        if row['start_time_cst']:
            parsed = dt.datetime.fromisoformat(row['start_time_cst'])
            assert parsed.tzinfo and parsed.utcoffset() == dt.timedelta(hours=8), row['id']
        assert row['sources'], row['id']
        normalized = []
        for rel in row['sources']:
            p = Path(rel)
            p = p if p.is_absolute() else WORK/p
            assert p.is_file() and not p.is_symlink(), (row['id'], rel)
            rel = p.relative_to(WORK).as_posix()
            assert '.session_control/' not in rel
            assert p.suffix.lower() not in ['.pt', '.pth', '.npz', '.npy', '.zip', '.gz', '.hdf5', '.h5', '.pkl']
            raw = p.read_bytes()
            bindings[rel] = dict(bytes=len(raw), sha256=sha(raw))
            normalized.append(rel)
        row['sources'] = normalized
    expected = {p.name for p in (WORK/'execution').iterdir() if p.is_dir()
                and p.name[:8].isdigit() and p.name[:8] <= '20261031' and p != HERE}
    assert expected <= seen, sorted(expected-seen)
    assert seen-expected == {'202608_first_reproduction'}, sorted(seen-expected)
    def chronological_key(row):
        if row['start_time_cst']:
            return row['start_time_cst']
        date = row['id'][:8]
        if not date.isdigit():
            return '2026-08-00'
        return date[:4]+'-'+date[4:6]+'-'+date[6:8]
    rows.sort(key=chronological_key)
    output = dict(title='MLR+IFA复现至RA、ALIGN与SHBR演进时间线',
                  date='2026-10-31', timezone='Asia/Shanghai', records=rows,
                  scope='Existing local experiment evidence; no new remote check, training, model loading or rollout',
                  current_Q_observation='2026-10-31T10:27:49.659740+08:00',
                  date_rule='Known actual starts are distinguished from pipeline dispatch, first observation, archive dates and document dates; unknown starts remain null',
                  no_cross_protocol_score_ranking=True)
    (HERE/'TIMELINE.json').write_text(json.dumps(output, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    columns = ['id', 'date_label', 'start_time_cst', 'start_basis', 'method', 'suite_seeds', 'change', 'result', 'status', 'role', 'sources']
    with (HERE/'TIMELINE.csv').open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            record = {k: row[k] for k in columns}
            record['sources'] = '; '.join(row['sources'])
            w.writerow(record)
    lines = ['# 完整逐轮时间线', '',
             '北京时间。此表覆盖工作区现有实验、扩展、诊断、方案审查与论文材料目录；这些条目不全部是独立训练实验。工程失败与接续不另充当成功的科学轮次。',
             '“启动依据”明确区分正式训练、流程派发、首次观察、文档日期与结果归档；未找到原始启动日志的条目不补造时分秒。', '',
             f'共{len(rows)}条；本地已存Q观察截止2026-10-31 10:27:49，未重新查询服务器。', '',
             '|时间及性质|轮次/范围|主要改动|结果与状态|证据|', '|---|---|---|---|---|']
    for row in rows:
        evidence = '、'.join(f'[{Path(p).name}]({(WORK/p).as_posix()})' for p in row['sources'])
        actual = row['start_time_cst']
        time = ((actual.replace('T', ' ').replace('+08:00', '')+' CST<br>') if actual else '')
        time += row['date_label']+'<br>'+row['start_basis']
        lines.append('|'+ '|'.join([clean(time), clean(row['method']+' / '+row['suite_seeds']), clean(row['change']),
                                   clean(row['result']+' ['+row['status']+']'), evidence])+'|')
    (HERE/'FULL_TIMELINE.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    audit = dict(status='PASS_TIMELINE_SOURCE_BINDING_AND_DIRECTORY_COVERAGE',
                 observed_at=dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
                 records=len(rows), execution_directories=len(expected), source_files=len(bindings),
                 known_timestamp_rows=sum(bool(r['start_time_cst']) for r in rows),
                 roles={k: sum(r['role'] == k for r in rows) for k in ['mainline', 'branch', 'audit']},
                 source_bindings=bindings, historical_commit=history,
                 scope='Metadata/source binding and coverage review; existing scientific acceptance reused, not re-executed',
                 new_training=0, new_environment_episodes=0, remote_experiment_commands=0)
    (HERE/'SOURCE_AUDIT.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: audit[k] for k in ['status', 'records', 'execution_directories', 'source_files', 'known_timestamp_rows', 'roles']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
