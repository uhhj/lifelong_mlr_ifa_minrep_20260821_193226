"""Recompute existing equal-budget paired outcomes locally; no model or environment calls."""
from pathlib import Path
from fractions import Fraction as F
import datetime
import hashlib
import json
import math
import statistics

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
ROOT = WORK / 'execution/20261030_equal_optimizer_budget_cs_two_suites'
FINAL = ROOT / 'final_readonly_20261031_065009'
BINDINGS = {}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, parse=True):
    raw = path.read_bytes()
    BINDINGS[path.relative_to(WORK).as_posix()] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return json.loads(raw.decode('utf-8-sig')) if parse else raw.decode('utf-8-sig')


def light(evidence, name):
    entry = evidence['light'][name]
    assert hashlib.sha256(entry['text'].encode('utf-8')).hexdigest() == entry['sha256']
    return json.loads(entry['text'])


def metrics(matrix):
    return dict(
        AUC=sum(sum(matrix[s, t] for s in range(t, 10)) / (10-t) for t in range(6, 10)) / 4,
        FWT=sum(matrix[t, t] for t in range(6, 10)) / 4,
        NBT=sum(sum(matrix[t, t]-matrix[s, t] for s in range(t+1, 10)) / (9-t) for t in range(6, 9)) / 3,
        OldFinal=sum(matrix[9, t] for t in range(6, 9)) / 3,
        FinalAll=sum(matrix[9, t] for t in range(6, 10)) / 4,
        FinalNew=matrix[9, 9],
    )


def close(a, b):
    assert math.isclose(float(a), float(b), rel_tol=0, abs_tol=1e-9), (a, b)


def main():
    acceptance = read(ROOT / 'ACCEPTANCE.json')
    comparison = read(FINAL / 'PRIMARY_S_COMPARISON.json')
    ca = read(FINAL / 'LOCAL_FINAL_AUDIT.json')
    ce = read(FINAL / 'EVIDENCE.json')
    assert acceptance['audit_sha256'] == comparison['source_C_final_sha256'] == sha(FINAL / 'LOCAL_FINAL_AUDIT.json')
    assert acceptance['comparison_sha256'] == sha(FINAL / 'PRIMARY_S_COMPARISON.json')
    assert acceptance['report_sha256'] == sha(ROOT / 'FINAL_REPORT.md')
    assert ca['source_sha256'] == sha(FINAL / 'EVIDENCE.json')
    assert ca['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
    assert ca['training']['updates'] == 1012500 and ca['evaluation']['episodes'] == 1200
    assert ca['final_cpu']['strict_CPs'] == 24 and ca['final_cpu']['complete_recoveries'] == 6
    assert ca['final_cpu']['teacher_caches'] == 18
    for name in ['PROTOCOL.md', 'FINAL_REPORT.md', 'S_ACCEPTANCE.json', 'RUN_STATUS.json']:
        read(ROOT / name, parse=name.endswith('.json'))
    stopped = read(WORK / 'execution/20261027_support_gate_forks/RUN_STATUS.json')
    assert stopped['formal_episodes'] == 0 and stopped['mechanism_formal_evaluation_cancelled']
    read(WORK / 'execution/20261030_publication_readiness/REPORT.md', parse=False)
    cases, suites, negative = {}, {}, []
    production_total = 0
    for suite, reference in comparison['source_S'].items():
        folder = Path(reference['final_path'])
        sa, se = read(folder / 'LOCAL_FINAL_AUDIT.json'), read(folder / 'EVIDENCE.json')
        assert sha(folder / 'LOCAL_FINAL_AUDIT.json') == reference['audit_sha256']
        assert sha(folder / 'EVIDENCE.json') == reference['evidence_sha256'] == sa['evidence_sha256']
        assert sa['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE' and sa['formal_episodes'] == 600
        for seed in (41, 42, 43):
            case = f'{suite}_{seed}'
            cp, sp = case+'/', f'seed{seed}/'
            crows, srows = light(ce, cp+'OUTCOMES.json'), light(se, sp+'OUTCOMES.json')
            key = lambda x: (x['stage'], x['task'], x['initial_index'])
            crows, srows = {key(x): x for x in crows}, {key(x): x for x in srows}
            pairs = {key(x): x for x in comparison['cases'][case]['paired_outcomes']}
            assert crows.keys() == srows.keys() == pairs.keys() and len(pairs) == 200
            initial_c, initial_s = light(ce, cp+'INITIAL_C.json'), light(se, sp+'INITIAL_S.json')
            for field in ['tensors', 'rng', 'hardware', 'official_cp_sha256', 'parent_initial_sha256']:
                assert initial_c[field] == initial_s[field], (case, field)
            assert light(ce, cp+'TASK6_ACCEPTED_S_PARITY.json')['all_tensors_exact']
            ch, sh = light(ce, cp+'EPOCHS_C.json'), light(se, sp+'EPOCHS_S.json')
            assert len(ch) == 700 and len(sh) == 400
            stage_budgets = {}
            for stage in range(6, 10):
                cnum = sum(x['batches'] for x in ch if x['stage'] == stage)
                snum = sum(x['batches']*(1 if stage == 6 else 2) for x in sh if x['stage'] == stage)
                assert cnum == snum
                assert cnum == comparison['cases'][case]['stage_budgets'][str(stage)]['C_production']
                ci, si = light(ce, cp+f'ADMISSION_C_{stage}.json'), light(se, sp+f'ADMISSION_S_{stage}.json')
                assert ci['optimizer_calls'] == si['optimizer_calls'] == (4 if stage == 6 else 29)
                stage_budgets[stage] = cnum
            production_total += sum(stage_budgets.values())
            cmatrix, smatrix, cells = {}, {}, {}
            for k, pair in pairs.items():
                c, s = crows[k], srows[k]
                assert type(c['success']) is bool and type(s['success']) is bool
                assert pair['C'] == c['success'] and pair['S'] == s['success']
                for field in ['initial_file_sha256', 'initial_sha256', 'noise_sha256', 'env_seed', 'attempt']:
                    assert c[field] == s[field], (case, k, field)
                for field in ['initial_sha256', 'noise_sha256']:
                    assert pair[field] == c[field]
                for arm, row in [('C', c), ('S', s)]:
                    assert pair[arm+'_cp_sha256'] == row['cp_sha256']
            for stage in range(6, 10):
                for task in range(6, stage+1):
                    rows = [x for k, x in pairs.items() if k[:2] == (stage, task)]
                    assert len(rows) == 20 and {x['initial_index'] for x in rows} == set(range(20))
                    cmatrix[stage, task] = F(sum(x['C'] for x in rows), 20)
                    smatrix[stage, task] = F(sum(x['S'] for x in rows), 20)
                    delta = (smatrix[stage, task]-cmatrix[stage, task])*100
                    sf = sum(x['C'] and not x['S'] for x in rows)
                    fs = sum(x['S'] and not x['C'] for x in rows)
                    assert delta == 5*(fs-sf)
                    cell = f'{stage}_{task}'
                    stored = comparison['cases'][case]['cells'][cell]
                    close(delta, stored['S_minus_C_pp'])
                    assert sf == len(stored['C_success_S_fail']) and fs == len(stored['C_fail_S_success'])
                    cells[cell] = dict(delta_pp=float(delta), SF=sf, FS=fs)
                    if delta < 0:
                        negative.append(dict(case=case, cell=cell, **cells[cell]))
            cm, sm = metrics(cmatrix), metrics(smatrix)
            values = {arm: {k: float(v*100) for k, v in metric.items()} for arm, metric in [('C', cm), ('S', sm)]}
            delta = {k: float((sm[k]-cm[k])*100) for k in cm}
            for metric in ['AUC', 'FWT', 'NBT', 'OldFinal', 'FinalAll']:
                for arm in ['C', 'S']:
                    close(values[arm][metric], comparison['cases'][case][arm+'_metrics'][metric])
                close(delta[metric], comparison['cases'][case]['delta_pp'][metric])
            cases[case] = dict(**values, delta_pp=delta, stage_production_calls_each=stage_budgets, cells=cells)
        suites[suite] = {}
        for metric in cm:
            rows = [cases[f'{suite}_{s}'] for s in (41, 42, 43)]
            summary = {arm+'_mean': statistics.mean(x[arm][metric] for x in rows) for arm in ['C', 'S']}
            summary.update({arm+'_sample_sd': statistics.stdev(x[arm][metric] for x in rows) for arm in ['C', 'S']})
            summary['delta_pp'] = statistics.mean(x['delta_pp'][metric] for x in rows)
            if metric != 'FinalNew':
                close(summary['delta_pp'], comparison['suites'][suite][metric]['S_minus_C_pp'])
            suites[suite][metric] = summary
    assert production_total == 1012500 and len(negative) == len(comparison['all_negative_cells']) == 13
    out = dict(status='PASS_LOCAL_PUBLICATION_READINESS_EVIDENCE_REVIEW',
               observed_at=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
               scope='Existing local metadata and 1200 paired outcome records; no remote calls, model loading or rollouts',
               accepted_experiment_observation=acceptance['observation'],
               paired_records=1200, underlying_formal_outcomes=2400, cases=cases, suites=suites,
               negative_cells=negative, source_bindings=BINDINGS,
               AUC_positive_cases=sum(x['delta_pp']['AUC'] > 0 for x in cases.values()),
               FinalAll_positive_cases=sum(x['delta_pp']['FinalAll'] > 0 for x in cases.values()),
               OldFinal_positive_cases=sum(x['delta_pp']['OldFinal'] > 0 for x in cases.values()),
               OldFinal_tied_cases=sum(x['delta_pp']['OldFinal'] == 0 for x in cases.values()),
               production_optimizer_calls_each=production_total, isolated_calls_each=546,
               extra_S_training_collection=240, mechanism_formal_episodes=0,
               limits=['Development seeds and reused initial states', 'Equal optimizer calls, not FLOPs/walltime/training interaction',
                       'Full-method result, not isolated support/retention/gate causality', 'No new statistical significance claim'])
    (HERE/'EVIDENCE_REVIEW.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: out[k] for k in ['status', 'paired_records', 'AUC_positive_cases', 'FinalAll_positive_cases', 'OldFinal_positive_cases', 'OldFinal_tied_cases', 'production_optimizer_calls_each']}))
    print(json.dumps(suites, indent=2))


if __name__ == '__main__':
    main()
