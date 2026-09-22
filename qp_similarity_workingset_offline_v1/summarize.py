"""Mechanical summaries: primary threshold, confirmation and real stress separated."""
import csv,json,math,os,statistics
from pathlib import Path
from qp_workingset_refinement_diag_v1.numeric import write

def read(p):return json.loads(Path(p).read_text())
def median(v):return statistics.median(v) if v else None

def atomic_csv(path,rows):
    tmp=path.with_name(path.name+'.tmp');keys=[]
    for r in rows:
        for k in r:
            if k not in keys:keys.append(k)
    with tmp.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');w.writeheader();w.writerows(rows)
    os.replace(tmp,path)

def summarize(out,complete=False):
    out=Path(out);rows=[];times=[];trace_rows=[]
    for done in sorted((out/'cases').glob('*/tau_*/COMPLETE.json')):
        info=read(done);spec=info['fixture'];cell=done.parent;tau=info['tau']
        for arm,seed in [('FULL_WS',None),('SIM_ADD',None),('RANDOM_ADD',0),('RANDOM_ADD',1),('RANDOM_ADD',2),('SIM_ONESHOT',None)]:
            directory=cell/('SIM_ADD' if arm=='SIM_ONESHOT' else (arm if seed is None else arm+'_'+str(seed)))
            repeats=[read(directory/('repeat%02d.json'%i)) for i in range(10)];raw=repeats[0];first=raw['first']
            one=arm=='SIM_ONESHOT'
            val=raw.get('oneshot_validation' if one else 'validation',{})
            if one:
                meta=dict(status='PASS' if first['full_kkt_pass'] and first['kernel_pass'] else 'FULL_CHECK_REJECTED',
                    full_kkt_pass=bool(first['full_kkt_pass'] and first['kernel_pass']),
                    k0=raw['k0'],k_final=len(first['C']),final_nonzero_support=first['nonzero_support'],inner_iterations=first['inner_iterations'],
                    initial_working_set_size=0,final_working_set_size=first['working_set_size'],
                    peak_linear_dimension=first['peak_linear_dimension'],linear_solves=first['linear_solves'],
                    subqp_rounds=1,added_constraints=0,first_missing_violations=first['missing_violation_count'],
                    first_missing_max=first['missing_violation_max'],float64=first['float64'],decimal80=first['decimal80'])
            else:meta=raw
            row=dict(fixture=spec['id'],source=spec['source'],split=spec['split'],kind=spec['kind'],m=spec['m'],
                tau=tau,arm=arm,random_seed=seed,cond_A=info['cond_A'],reference_status=info['reference_status'])
            for key in ('status','full_kkt_pass','k0','k_final','final_nonzero_support','initial_working_set_size',
                        'final_working_set_size','peak_linear_dimension','linear_solves','inner_iterations',
                        'subqp_rounds','added_constraints','first_missing_violations','first_missing_max'):
                row[key]=meta.get(key)
            for precision in ('float64','decimal80'):
                for key,v in meta[precision].items():row[precision+'_'+key]=v
            row.update(E_update=val.get('E_update'),objective_difference=val.get('objective_difference'),
                update_budget_pass=val.get('update_budget_pass',False),validation_status=val.get('status'),
                N_crosscheck_pass=val.get('N_crosscheck_pass'),
                deterministic=all(r.get('deterministic_match',True) for r in repeats),
                trace_file=str(directory/'constraint_trace.json'),linear_systems_file=str(directory/'linear_systems.json'))
            row['certified']=bool(row['full_kkt_pass'] and row['update_budget_pass'] and row['deterministic'] and info['reference_status']=='PASS')
            for field in ('total_seconds','grouping_seconds','selection_seconds','construction_seconds','subqp_seconds','full_check_seconds'):
                values=[(r['first']['elapsed_seconds'] if field=='total_seconds' else r['first']['timing'][field]) if one else r[field] for r in repeats]
                row[field+'_median']=median(values);row[field+'_min']=min(values);row[field+'_max']=max(values)
            rows.append(row)
            for rep,r in enumerate(repeats):
                time_row=dict(fixture=spec['id'],source=spec['source'],split=spec['split'],kind=spec['kind'],m=spec['m'],tau=tau,
                              arm=arm,random_seed=seed,repeat=rep,derived_first_round=one)
                for f in ('total_seconds','grouping_seconds','selection_seconds','construction_seconds','subqp_seconds','full_check_seconds'):
                    time_row[f]=(r['first']['elapsed_seconds'] if f=='total_seconds' else r['first']['timing'][f]) if one else r[f]
                times.append(time_row)
            if not one:
                for tr in read(directory/'constraint_trace.json'):
                    trace_rows.append(dict(fixture=spec['id'],tau=tau,arm=arm,random_seed=seed,**tr))
    atomic_csv(out/'per_instance.csv',rows);atomic_csv(out/'timing.csv',times)
    tmp=out/'constraint_trace.jsonl.tmp'
    with tmp.open('w') as f:
        for r in trace_rows:f.write(json.dumps(r,allow_nan=False,separators=(',',':'))+'\n')
    os.replace(tmp,out/'constraint_trace.jsonl')
    primary=[r for r in rows if r['tau']==.999]
    ratios=[]
    for sim in [r for r in primary if r['arm']=='SIM_ADD']:
        same=[r for r in primary if r['fixture']==sim['fixture']]
        full=next(r for r in same if r['arm']=='FULL_WS');rand=[r for r in same if r['arm']=='RANDOM_ADD']
        count_random=median([r['linear_solves'] for r in rand]);tr=median([r['total_seconds_median'] for r in rand])
        count_ratio=(sim['linear_solves']/count_random) if count_random else (1. if sim['linear_solves']==0 else None)
        ratios.append(dict(fixture=sim['fixture'],source=sim['source'],kind=sim['kind'],split=sim['split'],
            certified=sim['certified'],comparison_certified=full['certified'] and all(r['certified'] for r in rand),
            k0_fraction=sim['k0']/sim['m'],k_final_fraction=sim['k_final']/sim['m'],
            T_SIM_FULL=sim['total_seconds_median']/full['total_seconds_median'],T_SIM_RANDOM=sim['total_seconds_median']/tr,
            solves_SIM_RANDOM=count_ratio))
    atomic_csv(out/'paired_ratios.csv',ratios)
    def group_metrics(selected):
        if not selected:return dict(n=0)
        rt=median([r['T_SIM_RANDOM'] for r in selected]);rf=median([r['T_SIM_FULL'] for r in selected])
        counts=[r['solves_SIM_RANDOM'] for r in selected];rc=median(counts) if all(x is not None for x in counts) else None
        comparisons=all(r['comparison_certified'] for r in selected)
        return dict(n=len(selected),numerical_all=all(r['certified'] for r in selected),comparison_certified=comparisons,
            median_k_final_fraction=median([r['k_final_fraction'] for r in selected]),
            median_T_SIM_FULL=rf,median_T_SIM_RANDOM=rt,median_solves_SIM_RANDOM=rc,
            H3=bool(comparisons and (rt<=.90 or (rc is not None and rc<=.80 and rt<=1.))))
    subsets={}
    for split in ('implementation','confirmation'):
        subsets[split+'_high_mixed']=group_metrics([r for r in ratios if r['split']==split and r['kind'] in ('high','mixed')])
        subsets[split+'_low']=group_metrics([r for r in ratios if r['split']==split and r['kind']=='low'])
    hm=group_metrics([r for r in ratios if r['kind'] in ('high','mixed')]);low=group_metrics([r for r in ratios if r['kind']=='low'])
    real=[r for r in ratios if r['source']=='real'];rg=group_metrics(real)
    if real:
        rg.update(geometric_mean_T_SIM_FULL=math.exp(sum(math.log(r['T_SIM_FULL']) for r in real)/len(real)),
                  slowest_T_SIM_FULL=max(r['T_SIM_FULL'] for r in real),per_fixture=real)
    ready=len(ratios)==32
    gates={}
    label='IN_PROGRESS'
    if ready:
        gates=dict(H1=all(r['certified'] for r in ratios) and read(out/'UNIT_TESTS.json')['passed'],H2=hm['median_k_final_fraction']<=.75,
            H3=hm['H3'],H4_synthetic=bool(hm['comparison_certified'] and low['comparison_certified'] and hm['median_T_SIM_FULL']<=.80 and low['median_T_SIM_FULL']<=1.10),
            real_advance=bool(len(real)==5 and rg['numerical_all'] and rg['comparison_certified'] and rg['geometric_mean_T_SIM_FULL']<=.90))
        if not gates['H1']:label='NUMERICAL_OR_REFERENCE_UNRESOLVED'
        elif all(gates.values()):label='FEASIBLE_AND_PROMISING'
        elif hm['median_k_final_fraction']>.90:label='HEAVY_REINSERTION'
        elif gates['H2'] and not gates['H3']:label='CONSTRAINT_SELECTION_WORKS_SIMILARITY_UNPROVEN'
        else:label='EXACT_BUT_NO_SPEEDUP'
    summary=dict(status='COMPLETE' if complete else 'RUNNING',primary_label=label,gates=gates,
        completed_cells=len(list((out/'cases').glob('*/tau_*/COMPLETE.json'))),expected_cells=96,
        primary_instances=len(ratios),high_mixed=hm,low=low,real_stress=rg,subsets=subsets,
        primary_failures=[r for r in ratios if not r['certified']],
        timing_repeats_are_not_independent_fixtures=True,training_active=False,formal_solver_modified=False,model_writeback=False,GPU_required=False)
    write(out/'SUMMARY.json',summary)
    if complete:
        sensitivity={}
        for tau in (.99,.9999):
            ss=[r for r in rows if r['tau']==tau and r['arm']=='SIM_ADD']
            sensitivity[str(tau)]=dict(n=len(ss),certified=sum(r['certified'] for r in ss),
                high_mixed_median_final_fraction=median([r['k_final']/r['m'] for r in ss if r['kind'] in ('high','mixed')]))
        summary['sensitivity_descriptive_only']=sensitivity;write(out/'SUMMARY.json',summary)
        text=['# Similarity-guided constraint working-set offline v1','',
              'Primary label: **'+label+'**','',
              'All results below use tau=0.999. Sensitivity thresholds cannot replace this decision.',
              '', '## Frozen gates','', '```json',json.dumps(gates,indent=2),'```','',
              '## Separate synthetic and real evidence','',
              '```json',json.dumps(dict(high_mixed=hm,low=low,confirmation=subsets,real_stress=rg),indent=2),'```','',
              '## Limits','',
              '- Five selected real hard QPs all have64 valid constraints; no ordinary real32/96 set was added.',
              '-27 synthetic fixtures are generated as A=N.T@N,b=N.T@v. The third seed provides9 confirmation cases.',
              '- All timed methods start from zero; cached/full refined references are evaluation-only.',
              '- RANDOM_ADD seeds0/1/2 are reported separately; paired attribution uses their median, never their best.',
              '- FULL_WS dynamically chooses its own support. Full inner certification is reused without a redundant Decimal pass.',
              '- SIM_ONESHOT reuses the first SIM_ADD result and never contributes an accepted solution when full certification fails.',
              '- Time includes the inherited kernel condition-number and trace instrumentation. Extra whole-matrix spectra and reference comparisons are outside measured calls.',
              '-10 timing repetitions measure CPU timing stability, not10 independent QPs. No GPU/backpropagation speedup or closed-loop benefit is tested.',
              '- No float32 writeback or formal training. Original solver/recovery files remain unchanged.',
              '', '`training_active=false; formal_solver_modified=false; model_writeback=false; GPU_required=false`','']
        (out/'FINAL_REPORT.md').write_text('\n'.join(text))
