"""Describe fixed-ep99 controlled results using the existing metric definitions."""
import csv
import json
import statistics
from pathlib import Path
from summarize_final_paper_text_faithful import fixed_epoch_matrix, PAPER
from compute_object_metrics import metrics

root = Path(__file__).resolve().parents[1]
out = root / 'reports/own_reference_control'
out.mkdir(exist_ok=True)
all_results = {}
rows = []
for arm in ('mlr', 'release_ifa', 'corrected_ifa'):
    per_seed = {}
    for seed in (42,43,44):
        run = (root / f'runs/own_reference_control/ifa/seed{seed}') if arm == 'corrected_ifa' else (root / f'runs/final_paper_text_faithful/{"mlr" if arm == "mlr" else "ifa"}/seed{seed}')
        assert (run / 'PRIMARY_EVAL_COMPLETE').is_file(), run
        matrix,_ = fixed_epoch_matrix(run / 'primary_ep99.log')
        per_seed[str(seed)] = metrics(matrix)
        for i in range(4):
            for j in range(i+1):
                rows.append(dict(arm=arm,seed=seed,stage=i+6,task=j+6,success=matrix[i][j]))
    all_results[arm] = {'per_seed':per_seed, 'summary':{k:dict(mean=statistics.mean(per_seed[str(s)][k] for s in (42,43,44)), std=statistics.stdev(per_seed[str(s)][k] for s in (42,43,44))) for k in ('fwt','nbt','auc')}}
deltas = {}
for control in ('release_ifa','mlr'):
    per_seed = {str(s):{k:all_results['corrected_ifa']['per_seed'][str(s)][k]-all_results[control]['per_seed'][str(s)][k] for k in ('fwt','nbt','auc')} for s in (42,43,44)}
    deltas[control] = {'per_seed':per_seed,'mean':{k:statistics.mean(per_seed[str(s)][k] for s in (42,43,44)) for k in ('fwt','nbt','auc')}}
gaps = {k:dict(signed_gap=all_results['corrected_ifa']['summary'][k]['mean']-PAPER['ifa'][k][0], standardized_descriptive_gap=(all_results['corrected_ifa']['summary'][k]['mean']-PAPER['ifa'][k][0])/PAPER['ifa'][k][1]) for k in ('fwt','nbt','auc')}
result = dict(protocol='ep99; 20 trials; seeds42/43/44; sample std; original evaluation call order',results=all_results,corrected_minus_control=deltas,corrected_minus_paper=gaps,interpretation='Descriptive controlled results, not a significance test. Final causal interpretation requires validation logs and task6 negative-control review.')
(out/'CONTROLLED_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
with (out/'success_matrices.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['arm','seed','stage','task','success']);writer.writeheader();writer.writerows(rows)
lines=['# Controlled own-reference results','','Fixed ep99; 20 trials; sample standard deviations. No checkpoint selection.','','| Arm | Seed | FWT | NBT | AUC |','| --- | ---: | ---: | ---: | ---: |']
for arm,values in all_results.items():
    for seed,v in values['per_seed'].items():
        lines.append(f'| {arm} | {seed} | {v["fwt"]:.3f} | {v["nbt"]:.3f} | {v["auc"]:.3f} |')
    v=values['summary']
    lines.append('| '+arm+' | mean ± sample std | '+' | '.join(f'{v[k]["mean"]:.3f} ± {v[k]["std"]:.3f}' for k in ('fwt','nbt','auc'))+' |')
lines += ['', 'Full paired deltas, descriptive paper gaps and matrices are in CONTROLLED_RESULTS.json and success_matrices.csv. This is a numerical summary, not an automatically inferred final verdict.']
(out/'CONTROLLED_RESULTS.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(deltas,indent=2))
