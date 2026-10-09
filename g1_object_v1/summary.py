from fractions import Fraction

def exact_metrics(matrix):
    r=[[Fraction(str(x)) if x is not None else None for x in row] for row in matrix]
    n=len(r)
    fwt=sum(r[j][j] for j in range(n))/n
    nbt=sum(sum(r[j][j]-r[i][j] for i in range(j+1,n))/(n-j-1) for j in range(n-1))/(n-1)
    auc=sum(sum(r[i][j] for i in range(j,n))/(n-j) for j in range(n))/n
    return dict(auc=auc,fwt=fwt,nbt=nbt)


def floats(values):return {k:float(v) for k,v in values.items()}



def build_result(seed,matrix,baseline):
    values=exact_metrics(matrix);base=exact_metrics(baseline)
    return dict(seed=seed,r=matrix,metrics=floats(values),baseline_metrics=floats(base),
        exact_metrics={k:str(v) for k,v in values.items()},
        delta=floats({k:values[k]-base[k] for k in values}),
        run_additional_seeds=values['auc']>base['auc'])


def render(results):
    lines=['# G1-OBJECT-v1','',
        'Original mixed-batch BC and native weighted IFA, gated by decoder gradient dot sign. E1/P1 disabled.',
        '', '|Seed|G1 AUC|Historical IFA AUC|Delta AUC|FWT|NBT|',
        '|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append('|%d|%.9f|%.9f|%+.9f|%.9f|%.9f|'%(r['seed'],r['metrics']['auc'],r['baseline_metrics']['auc'],r['delta']['auc'],r['metrics']['fwt'],r['metrics']['nbt']))
    lines+=['','Seed42 gate: '+('POSITIVE_SINGLE_SEED_SIGNAL' if results[0]['run_additional_seeds'] else 'NO_OVERALL_BENEFIT'),
        'Exact rational AUC gate from the raw trial matrix; rounded display never controls continuation.',
        'Each seed: 400 epochs, four own ep99 checkpoints, 200 primary trials. No auxiliary evaluation or baseline rerun.',
        'Task6 has no G1 intervention: any historical difference at task6 is not attributable to gating.',
        'See MECHANISM_seed*.json for denominators, closure and own-trajectory IFA retention; this is not a baseline counterfactual.',
        'EXECUTION_COST.jsonl and per-seed training_cost record wall time. Probe/dot timings synchronize CUDA. No unsupported baseline speedup claim.',
        'A single-seed positive gate is an investment rule, not statistical significance. This does not prove shared knowledge or per-step forgetting prevention.']
    return '\n'.join(lines)+'\n'
