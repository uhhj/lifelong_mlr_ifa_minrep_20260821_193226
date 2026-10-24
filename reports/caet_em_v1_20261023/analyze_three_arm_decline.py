"""Recompute accepted scalar evidence only. No model imports, forwards or remote work."""
import ast
import hashlib
import json
from fractions import Fraction
from pathlib import Path

R = Path(__file__).resolve().parent
F = R / 'final_CAET_EM_20261024_153414'
S = R / 'snapshot_20261024_153235'
sources = {}

def read(p):
    raw = p.read_bytes()
    sources[str(p.relative_to(R)) if p.is_relative_to(R) else str(p)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

closure = read(F / 'CLOSURE_INDEX.json')
assert closure['status'] == 'PASS_COMPLETE_LOCAL_CLOSURE'
for name in ('LOCAL_FINAL_AUDIT.json', 'OFFLINE_DIAGNOSTICS.json', 'MECHANISM_ACCOUNTING.json'):
    p = F / name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == closure['sources'][str(p.relative_to(R))]
final = read(F / 'LOCAL_FINAL_AUDIT.json')
assert final['status'] == 'PASS_COMPLETE_LOCAL_ACCEPTANCE'
download = read(S / 'DOWNLOAD.json')
assert sources[str((S / 'DOWNLOAD.json').relative_to(R))] == closure['sources'][str((S / 'DOWNLOAD.json').relative_to(R))]

def meta(name):
    p = S / 'metadata' / name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == download[name]['sha256']
    return read(p)

matrices = {a: {ast.literal_eval(k): Fraction(v, 20) for k, v in data['success_counts'].items()} for a, data in final['arms'].items()}
align = read(R.parent / '20261022_caet_v3_tac_eftr_object41/final_ALIGN_seed41_20261022_220252/LOCAL_FINAL_AUDIT.json')
matrices['ALIGN'] = {(r['stage'], r['task']): Fraction(r['current']) for r in align['matrix']}
comparisons = {}
for a, b in [('B0', 'ALIGN'), ('B1', 'B0'), ('B2', 'B1'), ('B1', 'ALIGN'), ('B2', 'ALIGN')]:
    contributions = {str(t): (sum(matrices[a][s,t] - matrices[b][s,t] for s in range(t,10)) / (10-t)) / 4 for t in range(6,10)}
    comparisons[a+'-'+b] = {'task_AUC_contribution_pp': {t: float(v*100) for t,v in contributions.items()}, 'AUC_difference_pp': float(sum(contributions.values())*100)}

# Match all four valid horizons across all diagnostic metrics, avoiding branch-specific denominators.
matched = {}
for arm in ('B1', 'B2'):
    matched[arm] = {}
    for task in range(6,10):
        rows = meta(f'{arm}/diagnostics/task{task}.json')['rows']
        rows = [r for r in rows if r['current'] and all(v['count'] == 4 for v in r['metrics'].values())]
        assert rows
        means = {k: sum(r['metrics'][k]['sum'] for r in rows)/(4*len(rows)) for k in rows[0]['metrics']}
        matched[arm][str(task)] = {'windows': len(rows), 'valid_positions': 4*len(rows), 'means': means,
            'branch_action_error_ratio_to_GT': {k: means[f'evidence_{k}_permuted_action_error']/means['GT_effect_action'] for k in ('action','response','visual')},
            'response_prediction_to_GT_reconstruction_ratio': means['predicted_response']/means['GT_response']}

teachers = {}
for t in (7,8,9):
    d = meta(f'B2/teachers/task{t}.json')
    teachers[str(t)] = {k: d[k] for k in ('accepted','current_members','old_members','tau_action','tau_evidence','failures')}
    teachers[str(t)]['accepted_fraction_of_current_population'] = d['accepted']/d['current_members']
    assert d['accepted'] in (80,613,164)

for name in ('model.py','train.py','readout.py','diagnose.py','transfer.py','retention.py'):
    p = R / 'caet_em_exp' / name
    sources[str(p.relative_to(R))] = hashlib.sha256(p.read_bytes()).hexdigest()
assert matrices['B1'][6,6] == matrices['B2'][6,6] == Fraction(1,4)
mechanism = read(F / 'MECHANISM_ACCOUNTING.json')
assert all(v == 0 for v in mechanism['B2']['6'].values())
out = {'status': 'PASS_EXISTING_SCALAR_ATTRIBUTION_LIMITED', 'source_sha256': sources,
       'comparisons': comparisons, 'matched_current_full4_diagnostics': matched, 'teacher_coverage': teachers,
       'new_model_forwards': 0, 'new_updates': 0, 'new_environment_steps': 0,
       'limitations': ['No one-factor readout/adapter/horizon/learning-rate ablation.',
       'No matched ALIGN/B1/B2 per-horizon GMM diagnosis; B0 diagnosis cannot be transferred to other arms.',
       'Interventions on training/RA windows are not closed-loop interventions or deployment-available GT effects.',
       'Matching all-four-valid metrics removes denominator differences but retains an exploratory subset and artificial perturbations.',
       'One seed and repeatedly inspected benchmark; no statistical significance or unique causal attribution.']}
(R / 'THREE_ARM_DECLINE_ANALYSIS.json').write_text(json.dumps(out, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print(json.dumps({'status':out['status'], 'comparisons': comparisons, 'teachers': teachers,
 'matched':{a:{t:{'windows':x['windows'],'branch_ratio':x['branch_action_error_ratio_to_GT'],'response_ratio':x['response_prediction_to_GT_reconstruction_ratio'],'predicted_action':x['means']['predicted_action'],'GT_action':x['means']['GT_effect_action']} for t,x in d.items()} for a,d in matched.items()}},ensure_ascii=False,indent=2))
