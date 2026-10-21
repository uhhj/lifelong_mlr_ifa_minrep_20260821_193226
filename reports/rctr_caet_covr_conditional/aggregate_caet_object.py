"""Aggregate three independently accepted CAET runs; never consume partial scores."""
import argparse
import json
import statistics
from fractions import Fraction as F
from pathlib import Path
from audit_caet_progress import R, read, sha
from audit_rctr_completion_caet_progress import metrics, pack

def main():
    ap = argparse.ArgumentParser()
    for seed in (41,42,43):
        ap.add_argument(f'--seed{seed}', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = []; all_metrics = {k:[] for k in ('CAET','RA','paired_delta')}
    evidence = []
    for seed in (41,42,43):
        p = getattr(args,f'seed{seed}').resolve()
        assert p.parent == R
        suffix = '' if seed == 42 else f'_seed{seed}'
        audit_path = p/f'CAET_FINAL_AUDIT{suffix}.json'
        details_path = p/f'CAET_DETAILS{suffix}.json'
        a = read(audit_path); d = read(details_path); c = d['caet']
        receipt = c['completion']['receipt']
        assert (receipt['method'],receipt['suite'],receipt['seed']) == ('CAET','OBJECT',seed)
        assert f'CAET_OBJECT_seed{seed}_' in c['output']
        if 'seed' in a:
            assert a['seed'] == seed
        else:
            # Original seed42 audit predates the parameterized JSON schema.
            # Bind its identity through the hashed details and actual receipt;
            # never rewrite the original acceptance artifact.
            assert seed == 42 and p.name == 'progress_20261021_121832'
        assert a['engineering'] == 'LOCAL_INDEPENDENT_PASS'
        assert a['status'] in ('COMPLETED_GAIN','COMPLETED_NO_GAIN')
        assert a['source_details_sha256'] == sha(details_path)
        assert a['epochs'] == 400 and a['optimizer_steps'] == 105500
        assert len(a['checkpoints']) == 4 and a['formal']['records'] == 200
        assert a['formal']['completed_vectors'] == 20
        rates, m, _ = metrics(c['formal']['trials'],seed)
        base, b, _ = metrics(c['completion']['baseline_trials'],seed)
        assert all(F(a['metrics'][k]['fraction']) == v for k,v in m.items())
        assert all(F(a['baseline'][k]['fraction']) == v for k,v in b.items())
        delta = {k:m[k]-b[k] for k in m}
        for arm,v in (('CAET',m),('RA',b),('paired_delta',delta)):
            all_metrics[arm].append(v)
        rows.append(dict(seed=seed,metrics=pack(m),baseline=pack(b),delta=pack(delta),
            matrix=[dict(stage=s,task=t,current=str(v),baseline=str(base[s,t])) for (s,t),v in rates.items()]))
        evidence.append(dict(seed=seed,audit=str(audit_path.relative_to(R)),audit_sha256=sha(audit_path),
            details=str(details_path.relative_to(R)),details_sha256=sha(details_path)))
    summary = {}
    for arm,values in all_metrics.items():
        summary[arm] = {}
        for key in values[0]:
            mean = sum((v[key] for v in values),F(0))/3
            summary[arm][key] = dict(mean_fraction=str(mean),mean_percent=float(100*mean),
                sample_sd_pp=statistics.stdev(float(100*v[key]) for v in values),ddof=1)
    gate = F(summary['paired_delta']['AUC']['mean_fraction']) > 0
    output = args.output.resolve()
    assert output.is_relative_to(R) and not output.exists(), 'Preserve prior aggregation'
    output.mkdir(parents=True)
    result = dict(status='OBJECT_THREE_SEED_ACCEPTED',rows=rows,summary=summary,
        mean_auc_gate=gate,gate_rule='Exact equal-weight mean AUC strictly above matched RA; engineering PASS for all three',
        caet_epochs=1200,caet_main_updates=316500,caet_checkpoints=12,caet_formal_records=600,
        caet_vectors=60,evidence=evidence,scientific_changes=0,environment_calls=0)
    (output/'SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n')
    report = '# CAET OBJECT three-seed independent summary\n\n'
    report += '|Seed|RA AUC %|CAET AUC %|Delta pp|\n|---|---:|---:|---:|\n'
    for row in rows:
        report += f"|{row['seed']}|{row['baseline']['AUC']['percent']:.6f}|{row['metrics']['AUC']['percent']:.6f}|{row['delta']['AUC']['percent']:+.6f}|\n"
    report += '\n|Metric|RA mean +/- sample SD|CAET mean +/- sample SD|Paired delta mean +/- sample SD (pp)|\n|---|---:|---:|---:|\n'
    for key in summary['RA']:
        cells = [f"{summary[arm][key]['mean_percent']:.6f} +/- {summary[arm][key]['sample_sd_pp']:.6f}" for arm in ('RA','CAET','paired_delta')]
        report += '|'+key+'|'+'|'.join(cells)+'|\n'
    report += f'\nExact three-seed mean-AUC gate: {gate}. This is not evidence of statistical significance. GOAL admission remains a separate engineering/scientific applicability requirement.\n\nExploratory expansion after observing seed42; shared global action RNG, additional effect-model fitting, unknown pretraining state metadata, and schedule completion remain disclosed. Negative task results are included in SUMMARY.json.\n'
    (output/'REPORT.md').write_text(report,encoding='utf8')
    print(json.dumps(dict(status=result['status'],mean_auc_gate=gate,summary=summary),indent=2))

if __name__ == '__main__':
    main()
