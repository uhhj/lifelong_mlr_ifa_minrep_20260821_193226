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
        differences[suite][str(seed)]=dict(metric_delta_pp={k:g['metrics'][k]['percent']-m['metrics'][k]['percent'] for k in m['metrics']},cell_delta_pp=cells,comparison='Separate native main-table evaluation streams; not an identical-noise paired counterfactual')
        negative_cells.extend(dict(suite=suite,seed=seed,cell=k,original_task=m['matrix'][k]['original_task'],DEBR_minus_MLR_pp=v) for k,v in cells.items() if v<0)
out=dict(status='PASS_ACCEPTED_REVERSE_ORDER_ANALYSIS',observed=datetime.datetime.now().astimezone().isoformat(),order=spec['order'],new_final_original_task=6,old_final_original_tasks=[9,8,7],metric_definition='Same per-task temporal AUC and NBT as manuscript audit; NBT lower is better',results=results,summary=summary,DEBR_minus_MLR=differences,negative_cells=negative_cells,original_order_main_reference=binding['main_summary'],main_table_binding='MAIN_TABLE_BINDING.json',limitations=['Same development seeds and benchmark initial states reused','Original main-table native evaluators preserved separately; no claim of exact Normal-noise pairing across methods','Reverse order changes task populations and accepted update totals at fixed100 epochs/task','DEBR incurs acquisition interactions and candidate compute; resource differences itemized'])
(R/'FINAL_ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rows=[]
for suite in summary:
    for method in summary[suite]:rows.append('|%s|%s|%s|'%(suite,method,'|'.join('%.3f ± %.3f'%(summary[suite][method][k]['mean_percent'],summary[suite][method][k]['sample_sd_percent']) for k in ['FWT','NBT','AUC','OldFinal','NewFinal','FinalAll'])))
report='# 逆序实验最终报告\n\n12run实际完成并通过独立CPU及本地验收。基底任务0–5和官方预训练权重保留，增量原任务9→8→7→6。每格20回合，三seed均值±样本标准差，单位%%。\n\n|套件|方法|FWT↑|NBT↓|AUC↑|OldFinal↑|NewFinal↑|FinalAll↑|\n|---|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\nOldFinal对应原9/8/7，NewFinal对应原6。完整逐seed/原任务矩阵、负单元与全部成本见FINAL_ANALYSIS.json。MLR和DEBR分别继承主表原生评估流，不能把回合编号相同写成严格同噪声配对；当前开发seed与初态复用不构成独立确认。\n'
(R/'REPORT.md').write_text(report,encoding='utf-8')
status=dict(status='COMPLETE_ACCEPTED_ANALYSIS_PENDING_PUBLICATION',observed=out['observed'],evidence_directory=str(d.resolve()),actual_final_cpu_acceptance=True,contract=a['contract'],totals=a['totals'],summary=summary)
(R/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=out['status'],summary=summary,negative_cells=len(negative_cells)),ensure_ascii=False))
