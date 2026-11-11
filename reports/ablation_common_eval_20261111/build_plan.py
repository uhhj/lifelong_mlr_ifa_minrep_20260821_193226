"""Bind the historical component table to its original-order DEBR evaluator."""
import ast
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
E = R.parent
AUDIT = E / '20261111_main_ablation_protocol_audit'
REVERSE = E / '20261110_reverse_order_main_protocol'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    inventory = read(AUDIT / 'CHECKPOINT_INVENTORY_AUDIT.json')
    evidence = AUDIT / inventory['evidence']
    assert sha(evidence) == inventory['evidence_sha256']
    checkpoints = read(evidence)['files']
    audit = read(AUDIT / 'AUDIT.json')
    parents = read(REVERSE / 'preflight_readonly_20261110_224104/EVIDENCE.json')['parents']
    reference_roots = {
        'OBJECT': E / '20261028_support_gate_full_s_object',
        'GOAL': E / '20261029_support_gate_full_s_goal',
    }
    evaluator = reference_roots['OBJECT'] / 'seed41/evaluate.py'
    # Git checks the historical files out with CRLF on Windows. The server
    # uses the original LF bytes recorded in the accepted seed contracts.
    evaluator_bytes = evaluator.read_bytes().replace(b'\r\n', b'\n')
    evaluator_sha = hashlib.sha256(evaluator_bytes).hexdigest()
    cases = {}
    comparisons = {}
    for suite, directory in reference_roots.items():
        ref = read(directory / 'ACTIVE_ROOT.json')
        final_path = directory / ref['latest_evidence'] / 'EVIDENCE.json'
        final = read(final_path)
        assert read(final_path.with_name('LOCAL_FINAL_AUDIT.json'))['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
        for seed in (41, 42, 43):
            key = f'{suite}_{seed}'
            seed_dir = directory / f'seed{seed}'
            assert (seed_dir / 'evaluate.py').read_bytes().replace(b'\r\n', b'\n') == evaluator_bytes
            source = ast.parse((seed_dir / 'runtime.py').read_text(encoding='utf-8'))
            paths = {node.targets[0].id: ast.literal_eval(node.value.args[0])
                     for node in source.body if isinstance(node, ast.Assign)
                     and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ('O', 'S')}
            parent = parents[key]
            assert paths == {'O': parent['debr_align_parent'], 'S': parent['debr_align_source']}
            assert parent['debr_eval_config']['data']['task_order_index'] == 0
            light = final['light'][f'seed{seed}/OUTCOMES.json']
            assert hashlib.sha256(light['text'].encode()).hexdigest() == light['sha256']
            records = json.loads(light['text'])
            assert len(records) == 200
            initials = {}
            for row in records:
                identity = f"{row['task']}_{row['initial_index']}"
                value = [row[k] for k in ('initial_file_sha256', 'initial_sha256')]
                assert initials.setdefault(identity, value) == value
            assert len(initials) == 80
            comparisons[key] = {
                method: audit['cases'][f'{method}/{suite}/{seed}']['outcome_sources'][0]
                for method in ('C_normal', 'Q', 'S_latest')
            }
            for method in ('MLR_IFA', 'RA'):
                case = f'{method}/{suite}/{seed}'
                assert audit['cases'][case]['evaluation_family'].startswith('A:')
                models = {str(row['stage']): {'path': row['path'], 'sha256': row['recorded_sha256'], 'bytes': row['bytes']}
                          for row in checkpoints if row['kind'] == 'stage_model' and row['case'] == case and row['present']}
                assert set(models) == set('6789')
                cases[f'{suite}_{method}_{seed}'] = dict(method=method, suite=suite, seed=seed, models=models,
                    config_root=paths['O'], source_root=paths['S'], reference_dir=f"{ref['root']}/seed{seed}",
                    reference_contract_sha256=ref['children'][str(seed)], official_sha256=parent['official_pretrain_sha256'],
                    eval_config=parent['debr_eval_config'], initials=initials)
    assert len(cases) == 12
    (R / 'evaluate.py').write_bytes(evaluator_bytes)
    spec = dict(name='ABLATION_COMMON_DEBR_EVAL_20261111_v2', reference_evaluator_sha256=evaluator_sha,
                predecessor=read(REVERSE / 'ACTIVE_ROOT.json')['root'], task_order_index=0,
                inventory_sha256=sha(evidence), table_audit_sha256=sha(AUDIT / 'AUDIT.json'),
                evaluation_cases=cases, comparison_outcomes=comparisons, episodes_per_case=200, episodes=2400)
    with (R / 'SPEC.json').open('x', encoding='utf-8') as file:
        json.dump(spec, file, indent=2, ensure_ascii=False)
        file.write('\n')


if __name__ == '__main__':
    main()
