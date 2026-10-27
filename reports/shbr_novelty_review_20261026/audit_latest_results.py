"""Recompute accepted scalar matrices only. No imports of model or simulator code."""
from pathlib import Path
from fractions import Fraction
import csv, datetime, hashlib, json, statistics

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
DIRS = {
    'GOAL': WORK/'execution/20261026_ccbp_v4_s_goal42_43/final_readonly_20261026_164119',
    'OBJECT': WORK/'execution/20261026_ccbp_v4_s_object41_42_43/final_readonly_20261027_120754'
}
MECHANISM = WORK/'execution/20261025_ccbp_v4_full/final_readonly_20261026_014654/LOCAL_FINAL_AUDIT.json'
bound = {}

def read(path):
    data = path.read_bytes()
    bound[path.relative_to(WORK).as_posix()] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    return json.loads(data.decode('utf-8-sig'))

def recalc(matrix):
    assert set(matrix) == {f'{stage}_{task}' for stage in range(6,10) for task in range(6,stage+1)}
    rates = {}
    for key, value in matrix.items():
        if isinstance(value, dict):
            assert value['episodes'] == 20
            rates[key] = Fraction(value['successes'], value['episodes'])
            assert abs(float(rates[key])*100-value['percent']) < 1e-10
        else:
            rates[key] = Fraction(str(value))/100
        assert 0 <= rates[key] <= 1 and (rates[key]*20).denominator == 1
    metrics = {
        'AUC': sum(sum(rates[f'{s}_{t}'] for s in range(t,10))/(10-t) for t in range(6,10))/4,
        'FWT': sum(rates[f'{t}_{t}'] for t in range(6,10))/4,
        'NBT': sum(sum(rates[f'{t}_{t}']-rates[f'{s}_{t}'] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,
        'OldFinal': sum(rates[f'9_{t}'] for t in range(6,9))/3,
        'FinalAll': sum(rates[f'9_{t}'] for t in range(6,10))/4
    }
    return {k: {'fraction': str(v), 'percent': float(v)*100} for k,v in metrics.items()}

def verify(matrix, expected):
    got = recalc(matrix)
    for k, v in got.items():
        assert Fraction(v['fraction']) == Fraction(expected[k]['fraction']), (k,v,expected[k])
        assert abs(v['percent']-expected[k]['percent']) < 1e-9
    return got

def main():
    suites = {}
    csv_rows = []
    for suite, directory in DIRS.items():
        audit = read(directory/'LOCAL_FINAL_AUDIT.json')
        assert audit['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
        comp = read(directory/'COMPARISON.json')
        reference = read(directory.parent/'REFERENCE_RESULTS.json')
        assert comp['final_audit_sha256'] == bound[(directory/'LOCAL_FINAL_AUDIT.json').relative_to(WORK).as_posix()]['sha256']
        assert comp['reference_sha256'] == bound[(directory.parent/'REFERENCE_RESULTS.json').relative_to(WORK).as_posix()]['sha256']
        runs = {}
        for seed, row in comp['S'].items():
            runs[f'S_{seed}'] = verify(row['matrix'], row['metrics'])
        for row in reference['baselines']['runs']:
            runs[f"{row['method']}_{row['seed']}"] = verify(row['matrix'], row['metrics'])
        summary = {}
        for method in ['MLR_IFA','RA','RA_ALIGN','S']:
            summary[method] = {}
            for metric in ['AUC','FWT','NBT','OldFinal','FinalAll']:
                values = [runs[f'{method}_{s}'][metric]['percent'] for s in [41,42,43]]
                summary[method][metric] = {'mean': statistics.mean(values), 'sample_sd': statistics.stdev(values)}
            csv_rows.append({'suite': suite, 'method': method, **{k:round(v['mean'],10) for k,v in summary[method].items()}})
        for metric, record in comp['three_seed_statistics'].items():
            for field in ['mean','sample_sd']:
                assert abs(summary['S'][metric][field]-record[field]) < 1e-9
        deltas = {baseline: {metric:summary['S'][metric]['mean']-summary[baseline][metric]['mean']
                            for metric in summary['S']} for baseline in ['MLR_IFA','RA','RA_ALIGN']}
        suites[suite] = {'runs': runs, 'summary': summary, 'S_minus_baseline_pp': deltas,
                         'all_negative_cells': comp['all_negative_cells'],
                         'limits': comp['limits']}
    mechanism = read(MECHANISM)
    assert mechanism['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
    m_metrics = {arm:verify(matrix, mechanism['metrics'][arm]) for arm,matrix in mechanism['matrix'].items()}
    output = {
        'status': 'PASS_RECOMPUTED_ACCEPTED_MATRICES_NOT_CAUSAL_OR_SIGNIFICANCE_TEST',
        'review_date': '2026-10-27',
        'computed_at': datetime.datetime.now().astimezone().isoformat(),
        'suites': suites,
        'goal41_C_D_O_S': {'metrics': m_metrics, 'matrix': mechanism['matrix'],
                          'support': mechanism['scientific_exploratory_support']},
        'source_bindings': bound,
        'scope': 'Pure local rational recomputation from saved accepted scalar matrices; zero models, gradients, rollouts or remote commands.'
    }
    (HERE/'LATEST_RESULTS_RECOMPUTED.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (HERE/'LATEST_METRICS.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['suite','method','AUC','FWT','NBT','OldFinal','FinalAll'])
        writer.writeheader();writer.writerows(csv_rows)
    print(json.dumps({'status':output['status'],'suites': {s:v['S_minus_baseline_pp'] for s,v in suites.items()}},ensure_ascii=False))

if __name__ == '__main__':
    main()
