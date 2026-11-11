"""Exact manuscript metrics and reverse-order task identities after acceptance."""
from pathlib import Path
from fractions import Fraction as F
import datetime,json,statistics,sys
from audit_progress import decode,read
R=Path(__file__).resolve().parent;d=Path(sys.argv[1]);a=read(d/'LOCAL_FINAL_AUDIT.json');e=read(d/'EVIDENCE.json');obj=decode(e)
assert a['status']=='PASS_COMPLETE_REVERSE_LOCAL_ACCEPTANCE'
spec=read(R/'SPEC.json');binding=read(R/'MAIN_TABLE_BINDING.json');results={};all_negative=[]
def metrics(r):
    return dict(FWT=sum(r[j,j] for j in range(6,10))/4,NBT=sum(sum(r[j,j]-r[i,j] for i in range(j+1,10))/(9-j) for j in range(6,9))/3,AUC=sum(sum(r[i,j] for i in range(j,10))/(10-j) for j in range(6,10))/4,OldFinal=sum(r[9,j] for j in range(6,9))/3,NewFinal=r[9,9],FinalAll=sum(r[9,j] for j in range(6,10))/4)
for case,entry in spec['cases'].items():
    outcomes=obj[case+('/NORMALIZED_OUTCOMES.json' if entry['method']=='MLR' else '/OUTCOMES.json')]
    cells={}
    for x in outcomes:cells.setdefault((x['stage'],x['task']),[]).append(x)
    assert len(cells)==10 and all(len(v)==20 for v in cells.values())
    matrix={k:F(sum(v['success'] for v in values),20) for k,values in cells.items()}
    results[case]=dict(matrix={str(s)+'_'+str(t):dict(successes=int(v*20),episodes=20,percent=float(v*100),original_task=spec['order'][t],current_original_task=spec['order'][s]) for (s,t),v in matrix.items()},metrics={k:dict(fraction=str(v),percent=float(v*100)) for k,v in metrics(matrix).items()},cost=e['cpu'][case])
summary={};differences={};negative_cells=[]
for suite in ['OBJECT','GOAL']:
    summary[suite]={}
    for method in ['MLR','DEBR']:
        runs=[results['%s_%s_%d'%(suite,method,seed)] for seed in [41,42,43]]
        summary[suite][method]={k:dict(mean_percent=statistics.mean(x['metrics'][k]['percent'] for x in runs),sample_sd_percent=statistics.stdev(x['metrics'][k]['percent'] for x in runs)) for k in runs[0]['metrics']}
    differences[suite]={}
    for seed in [41,42,43]:
        m=results['%s_MLR_%d'%(suite,seed)];g=results['%s_DEBR_%d'%(suite,seed)]
        cells={k:g['matrix'][k]['percent']-m['matrix'][k]['percent'] for k in m['matrix']}
        mr=obj['%s_MLR_%d/NORMALIZED_OUTCOMES.json'%(suite,seed)];gr=obj['%s_DEBR_%d/OUTCOMES.json'%(suite,seed)]
        paired={}
        for stage in range(6,10):
            for task in range(6,stage+1):
                mm=[x for x in mr if x['stage']==stage and x['task']==task];gg=[x for x in gr if x['stage']==stage and x['task']==task]
                assert len(mm)==len(gg)==20
                for x,y in zip(mm,gg):assert all(x[k]==y[k] for k in ['initial_index','initial_file_sha256','initial_sha256','noise_sha256','env_seed','attempt'])
                paired['%d_%d'%(stage,task)]=dict(DEBR_success_MLR_failure=sum(y['success'] and not x['success'] for x,y in zip(mm,gg)),MLR_success_DEBR_failure=sum(x['success'] and not y['success'] for x,y in zip(mm,gg)),both_success=sum(x['success'] and y['success'] for x,y in zip(mm,gg)),both_failure=sum(not x['success'] and not y['success'] for x,y in zip(mm,gg)))
        differences[suite][str(seed)]=dict(metric_delta_pp={k:g['metrics'][k]['percent']-m['metrics'][k]['percent'] for k in m['metrics']},cell_delta_pp=cells,paired_discordance=paired,comparison='Common full-DEBR evaluation with verified identical initial-state and per-trajectory Normal inputs; method-specific training compute/interactions retained')
        negative_cells.extend(dict(suite=suite,seed=seed,cell=k,original_task=m['matrix'][k]['original_task'],DEBR_minus_MLR_pp=v) for k,v in cells.items() if v<0)
out=dict(status='PASS_ACCEPTED_REVERSE_ORDER_ANALYSIS',observed=datetime.datetime.now().astimezone().isoformat(),evaluation_protocol='COMMON_DEBR_EVALUATION_20261111_V1',user_authorized_amendment='EVALUATION_PROTOCOL_AMENDMENT.json',order=spec['order'],new_final_original_task=6,old_final_original_tasks=[9,8,7],metric_definition='Same per-task temporal AUC and NBT as manuscript audit; NBT lower is better',results=results,summary=summary,DEBR_minus_MLR=differences,negative_cells=negative_cells,original_order_main_reference=binding['main_summary'],main_table_binding='MAIN_TABLE_BINDING.json',limitations=['Same development seeds and benchmark initial states reused','User unified the new reverse-order formal evaluations before the first episode; historical original-order MLR still uses its earlier native evaluator and is not a pure order-only comparator','Reverse order changes task populations and accepted update totals at fixed100 epochs/task','DEBR incurs acquisition interactions and candidate compute; resource differences itemized'])
(R/'FINAL_ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rows=[]
for suite in summary:
    for method in summary[suite]:rows.append('|%s|%s|%s|'%(suite,method,'|'.join('%.3f ± %.3f'%(summary[suite][method][k]['mean_percent'],summary[suite][method][k]['sample_sd_percent']) for k in ['FWT','NBT','AUC','OldFinal','NewFinal','FinalAll'])))
report='# 逆序实验最终报告\n\n12run实际完成并通过独立CPU及本地验收。基底任务0–5和官方预训练权重保留，增量原任务9→8→7→6。每格20回合，三seed均值±样本标准差，单位%%。\n\n|套件|方法|FWT↑|NBT↓|AUC↑|OldFinal↑|NewFinal↑|FinalAll↑|\n|---|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\nOldFinal对应原9/8/7，NewFinal对应原6。完整逐seed/原任务矩阵、负单元、配对SF/FS及成本见FINAL_ANALYSIS.json。按用户在首个正式回合前的修订，MLR和DEBR均使用完整DEBR评估，实际核同初态／Normal／env_seed。纯MLR训练保留，已完成训练未重复。原顺序历史MLR仍为旧评估口径，不能把其与新逆序MLR的差全归于顺序；当前开发seed与初态复用不构成独立确认。\n'
(R/'REPORT.md').write_text(report,encoding='utf-8')
status=dict(status='COMPLETE_ACCEPTED_ANALYSIS_PENDING_PUBLICATION',observed=out['observed'],evidence_directory=str(d.resolve()),actual_final_cpu_acceptance=True,contract=a['contract'],totals=a['totals'],summary=summary)
(R/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=out['status'],summary=summary,negative_cells=len(negative_cells)),ensure_ascii=False))
