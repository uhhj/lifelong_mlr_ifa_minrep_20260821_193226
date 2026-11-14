import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


ORIGINAL = Path('/data/mlr_ifa/experiment_workspaces/MATCHED_HISTORY_SOURCE_GOAL_OBJECT_20261112_v1')
RECOVERY = ACTION in ('recover', 'recovery_status')
ROOT = ORIGINAL.with_name(ORIGINAL.name + '_engineering02') if RECOVERY else ORIGINAL
DATA = Path('/data/mlr_ifa/experiments') / ROOT.name
PYTHON = '/data/mlr_ifa/env/bin/python'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def gpu(query):
    return subprocess.check_output(['nvidia-smi', query, '--format=csv,noheader'], text=True).strip()


def verify_population(case, path, parent):
    fields = ('POP', 'VALID', 'OLD')
    assignments = {n.targets[0].id: ast.literal_eval(n.value)
                   for n in ast.parse(BUNDLE[path]).body if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
                   and n.targets[0].id in fields}
    for stage in range(6, 10):
        manifest = Path(parent['output']) / 'ra_coreset' / 'dataset_manifests' / ('task%d.json' % stage)
        assert digest(manifest) == parent['files'][str(manifest)]
        rows = json.loads(manifest.read_text())['rows']
        counts = (len(rows), sum(r['bc_valid'] for r in rows),
                  sum(r['bc_valid'] and r['task'] < stage for r in rows))
        assert tuple(assignments[field][stage] for field in fields) == counts, (case, stage, counts)


def inherit():
    prior = ORIGINAL / 'PAIR_CONTRACT.json'
    assert digest(prior) == '1b29710b7336252e3ba754274c1a12c1cbd62fff778e94902002a114c14d224c'
    assert (ORIGINAL / 'controller.exit').read_text() == '1'
    old = json.loads(prior.read_text())['files']
    assert all(digest(path) == value for path, value in old.items())
    for process in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = process.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError):
            continue
        assert not any(arg.startswith(str(ORIGINAL).encode()+b'/') for arg in argv), process
    complete = ORIGINAL / 'pairs' / 'GOAL_41'
    for phase in ('match', 'initial_E', 'initial_S', 'train_E', 'train_S', 'evaluate'):
        assert (complete / (phase + '.exit')).read_text() == '0'
    evaluation = json.loads((complete / 'OUTCOMES.json').read_text())
    assert len(evaluation) == len({(r['arm'],r['task'],r['initial_index']) for r in evaluation}) == 160
    assert json.loads((complete / 'EVALUATION_COMPLETE.json').read_text())['episodes'] == 160
    keep = [prior, ORIGINAL/'controller.exit', ORIGINAL/'FAILURE.json', complete/'OUTCOMES.json',
            complete/'EVALUATION_COMPLETE.json', complete/'EPOCHS_E.json', complete/'EPOCHS_S.json',
            *sorted((complete/'outcomes').glob('*.json')), *sorted(complete.glob('*.exit'))]
    for arm in ('E','S'):
        report = complete / ('TRAIN_%s.json' % arm)
        tr = json.loads(report.read_text())
        model = Path(tr['checkpoint'])
        assert tr['epochs'] == 100 and tr['accepted_updates'] == 33700 and digest(model) == tr['sha256']
        keep.extend([report, model])
    partial = ORIGINAL / 'GOAL_42'
    assert (partial/'collect_6.exit').read_text() == '0' and (partial/'collect_7.exit').read_text() == '1'
    assert not (partial/'COLLECTION_STARTED_S_8.json').exists() and not (partial/'MEMORY_S_7.json').exists()
    first = ('collect_6.exit', 'COLLECTION_STARTED_S_6.json', 'COLLECTION_ADMISSION_S_6.json',
             'COLLECTION_OUTCOMES_S_6.json', 'MEMORY_S_6.json')
    keep.extend(partial / name for name in first)
    keep.extend(partial / name for name in ('COLLECTION_STARTED_S_7.json', 'COLLECTION_OUTCOMES_S_7.json',
               'collect_7.exit', 'collect_7.log', 'ACQUISITION_FAILURE.json'))
    failed = partial/'COLLECTION_OUTCOMES_S_7.json'
    assert digest(failed) == '025e4771403ade9e15ed4d95035f9c64375bcd2eea979222803ca6b5f9365130'
    records = json.loads(failed.read_text())
    assert [r['initial_index'] for r in records] == list(range(20,30))
    old_cache = DATA.with_name(ORIGINAL.name)/'GOAL_42'/'acquisition'/'memory'/'task6.pth'
    memory = json.loads((partial/'MEMORY_S_6.json').read_text())
    assert not old_cache.with_name('task7.pth').exists() and not old_cache.with_name('task8.pth').exists()
    cache_hash = digest(old_cache)
    assert cache_hash == memory['cache_sha256'] and memory['training_attempts'] == 10
    assert digest(partial/'COLLECTION_OUTCOMES_S_6.json') == memory['collection_outcomes_sha256']
    keep.append(old_cache)
    return first, old_cache, {str(p):digest(p) for p in keep}, dict(original_outcomes=str(failed),sha256=digest(failed))


def launch():
    source = ORIGINAL / 'GOAL_41'
    assert (source / 'acquire.exit').read_text() == '0'
    assert not (ROOT / 'pairs').exists() and not (ROOT / 'PAIR_CONTRACT.json').exists()
    assert not gpu('--query-compute-apps=pid'), 'Another process uses the GPU'
    assert shutil.disk_usage('/root').free >= 1024**3 and shutil.disk_usage('/data').free >= 10 * 1024**3
    if RECOVERY:
        assert not ROOT.exists() and not DATA.exists()
        first, old_cache, inherited, retry = inherit()
    acquisition = json.loads((source / 'CONTRACT.json').read_text())
    assert all(digest(path) == h for path, h in acquisition['files'].items())
    cases = json.loads(BUNDLE['INPUT_PLAN.json'])['cases']
    for case, binding in cases.items():
        path = 'pairs/' + case + '/'
        assert json.loads(BUNDLE[path + 'CASE.json']) == binding
        assert all(digest(teacher['path']) == teacher['sha256'] for teacher in binding['teachers'].values())
        parent = json.loads(BUNDLE[path + 'PARENT_BINDING.json'])
        assert all(digest(name) == h for name, h in parent['files'].items())
        assert all(digest(Path(parent['source']) / name) == h for name, h in parent['source_sha256'].items())
        assert binding['common_initial'] == binding['teachers']['8']
        verify_population(case, path + 'layout.py', parent)
        if case != 'GOAL_41':
            verify_population(case, case + '/layout.py', parent)
    for name, value in BUNDLE.items():
        p = Path(name)
        assert not p.is_absolute() and '..' not in p.parts
        assert not (ROOT / p).exists()
        if p.suffix == '.py':
            compile(value, name, 'exec')
    for name, value in BUNDLE.items():
        p = ROOT / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(value, encoding='utf-8')
    frozen = {str(ROOT / name): digest(ROOT / name) for name in BUNDLE}
    frozen[str(source / 'CONTRACT.json')] = digest(source / 'CONTRACT.json')
    if RECOVERY:
        frozen.update(inherited)
        for name in first:
            target = ROOT/'GOAL_42'/name
            shutil.copyfile(ORIGINAL/'GOAL_42'/name, target)
            frozen[str(target)] = digest(target)
        target_cache = DATA/'GOAL_42'/'acquisition'/'memory'/'task6.pth'
        target_cache.parent.mkdir(parents=True)
        os.link(old_cache, target_cache)
        frozen[str(target_cache)] = digest(target_cache)
    for case in cases:
        for directory in [ROOT / 'pairs' / case] + ([ROOT / case] if case != 'GOAL_41' else []):
            contract = {'files': {p: h for p, h in frozen.items() if p.startswith(str(directory) + '/')},
                        'case': case, 'kind': 'acquisition' if directory.parent == ROOT else 'pair'}
            if RECOVERY and case == 'GOAL_42' and directory.parent == ROOT:
                contract.update(inherited_task6={'cache':str(target_cache),'sha256':frozen[str(target_cache)]},
                                authorized_task7_repeat=retry)
            p = directory / 'CONTRACT.json'
            p.write_text(json.dumps(contract, indent=2) + '\n')
            frozen[str(p)] = digest(p)
    p = ROOT / 'PAIR_CONTRACT.json'
    contract = {'files': frozen, 'cases': list(cases), 'budgets': {'train_acquisition': 180,
        'accepted_steps': 419400, 'production_optimizer_calls': 838800, 'formal_episodes': 960}}
    if RECOVERY:
        contract.update(inherited_completed_cases=['GOAL_41'], original_root=str(ORIGINAL),
                        additional_failed_attempt_episodes=10, remaining_collection_episodes=140,
                        remaining_accepted_steps=352000, remaining_formal_episodes=800)
    p.write_text(json.dumps(contract, indent=2) + '\n')
    first = ROOT / 'pairs' / ('GOAL_42' if RECOVERY else 'GOAL_41')
    imported = subprocess.run([PYTHON, '-B', '-c',
        'import match_histories, train_pair, evaluate, runtime; runtime.verify(); runtime.capacity(); '
        + ('[runtime.Dataset(t) for t in range(6,10)]' if RECOVERY else '')],
        cwd=first, capture_output=True, text=True)
    (ROOT / 'PAIR_IMPORT_REVIEW.json').write_text(json.dumps(dict(exit=imported.returncode,
        stdout=imported.stdout, stderr=imported.stderr), indent=2) + '\n')
    assert imported.returncode == 0, imported.stderr
    assert not gpu('--query-compute-apps=pid'), 'Another process started during preflight'
    with (ROOT / 'controller.log').open('x') as log:
        proc = subprocess.Popen([PYTHON, '-B', str(ROOT / 'controller.py')], cwd=ROOT,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    return dict(state='DISPATCHED_PAIR_QUEUE_PENDING_ADMISSION', controller=proc.pid,
        contract_sha256=digest(p), files=len(frozen), root=str(ROOT), acquisition_exit=0,
        inherited_completed_cases=contract.get('inherited_completed_cases', []),
        inherited_task6_cache_sha256=digest(target_cache) if RECOVERY else None)


def status():
    output = dict(root=str(ROOT), gpu=gpu('--query-gpu=name,memory.used,utilization.gpu'),
                  gpu_processes=gpu('--query-compute-apps=pid,used_gpu_memory'))
    for name in ('STARTED.json', 'STATUS.json', 'FAILURE.json', 'PAIR_IMPORT_REVIEW.json', 'controller.exit'):
        path = ROOT / name
        if path.exists():
            output[name] = json.loads(path.read_text())
    output['cases'] = {}
    for directory in sorted((ROOT / 'pairs').glob('*')):
        case = directory.name
        if RECOVERY and case == 'GOAL_41':
            directory = ORIGINAL / 'pairs' / case
        record = {'exits': {p.name: p.read_text() for p in directory.glob('*.exit')}, 'epochs': {}, 'formal_episodes': 0}
        for arm in ('E', 'S'):
            p = directory / ('EPOCHS_%s.json' % arm)
            if p.exists():
                rows = json.loads(p.read_text())
                record['epochs'][arm] = dict(count=len(rows), latest=rows[-1],
                    mean_last_ten_epoch_s=sum(row['seconds'] for row in rows[-10:])/len(rows[-10:]))
            p = directory / ('ADMISSION_%s.json' % arm)
            if p.exists():
                record['admission_' + arm] = json.loads(p.read_text())['status']
            p = directory / ('GATES_%s.jsonl' % arm)
            if p.exists():
                with p.open() as stream:
                    lines = [json.loads(line) for line in stream if line.endswith('\n')]
                if lines:
                    record['production_' + arm] = dict(completed_gate_blocks=len(lines),
                        accepted_updates=sum(row['steps'] for row in lines),
                        physical_optimizer_calls=2*sum(row['steps'] for row in lines),
                        latest_epoch_index=lines[-1]['epoch'])
        matched = directory / 'MATCHED_MEMORY.json'
        if matched.exists():
            rows = json.loads(matched.read_text())['tasks']
            record['matched'] = {task: {key: row[key] for key in ('paired_trajectories', 'train_per_source',
                'probe_per_source', 'support')} for task, row in rows.items()}
        outcomes = directory / 'OUTCOMES.json'
        if outcomes.exists():
            record['formal_episodes'] = len(json.loads(outcomes.read_text()))
        for path in directory.glob('*.exit'):
            if path.read_text() != '0':
                record['failure_tail'] = path.with_suffix('.log').read_text(errors='replace').splitlines()[-35:]
        output['cases'][case] = record
    if RECOVERY:
        acquisition = ROOT/'GOAL_42'
        output['recovery'] = {}
        for name in ('ACQUISITION_STATUS.json','ACQUISITION_FAILURE.json','TASK7_REPEAT_RECEIPT.json',
                     'COLLECTION_STARTED_S_7.json','COLLECTION_STARTED_S_8.json'):
            path = acquisition/name
            if path.exists():
                output['recovery'][name] = json.loads(path.read_text())
        output['recovery']['exits'] = {p.name:p.read_text() for p in acquisition.glob('*.exit')}
        for path in acquisition.glob('collect_*.exit'):
            if path.read_text() != '0':
                output['recovery']['failure_tail'] = path.with_suffix('.log').read_text(errors='replace').splitlines()[-20:]
    return output


result = launch() if ACTION in ('launch','recover') else status()
result['observed'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
print(json.dumps(result))
