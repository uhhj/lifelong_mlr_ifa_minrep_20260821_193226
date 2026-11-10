"""Accepted stripped-S comparisons with exact metrics and paired outcome changes."""
from pathlib import Path
from fractions import Fraction as F
import datetime,json,statistics,sys
R=Path(__file__).resolve().parent;d=Path(sys.argv[1])
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
a=J(d/'LOCAL_FINAL_AUDIT.json');e=J(d/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['formal']==2400
full=J(R/'FULL_S_BASELINE_OUTCOMES.json')['cases'];historical=J(R/'BASELINE_OUTCOMES.json')['cases'];records={};results={}
def metrics(r):
    return dict(FWT=sum(r[j,j] for j in range(6,10))/4,NBT=sum(sum(r[j,j]-r[i,j] for i in range(j+1,10))/(9-j) for j in range(6,9))/3,AUC=sum(sum(r[i,j] for i in range(j,10))/(10-j) for j in range(6,10))/4,OldFinal=sum(r[9,j] for j in range(6,9))/3,NewFinal=r[9,9],FinalAll=sum(r[9,j] for j in range(6,10))/4)
def analyze(rows):
    cells={}
    for x in rows:cells.setdefault((x['stage'],x['task']),[]).append(x)
    assert len(cells)==10 and all(len(v)==20 for v in cells.values())
    matrix={k:F(sum(x['success'] for x in v),20) for k,v in cells.items()}
    return dict(matrix={str(s)+'_'+str(t):dict(successes=int(v*20),episodes=20,percent=float(v*100)) for (s,t),v in matrix.items()},metrics={k:dict(fraction=str(v),percent=float(v*100)) for k,v in metrics(matrix).items()})
for suite in ['OBJECT','GOAL']:
    for seed in [41,42,43]:
        for variant in ['MLR','ALIGN']:
            case='%s_%s_%d'%(suite,variant,seed);records[case]=json.loads(e['light'][case+'/OUTCOMES.json']['text']);results[case]=analyze(records[case])
        key=suite+'_'+str(seed);case=suite+'_FULL_S_'+str(seed);records[case]=full[key];results[case]=analyze(records[case])
        results[suite+'_HISTORICAL_MLR_'+str(seed)]=analyze(historical[key])
summary={};pairs={};negative=[]
for suite in ['OBJECT','GOAL']:
    summary[suite]={}
    for method in ['MLR','ALIGN','FULL_S','HISTORICAL_MLR']:
        runs=[results['%s_%s_%d'%(suite,method,k)] for k in [41,42,43]]
        summary[suite][method]={name:dict(mean_percent=statistics.mean(x['metrics'][name]['percent'] for x in runs),sample_sd_percent=statistics.stdev(x['metrics'][name]['percent'] for x in runs)) for name in runs[0]['metrics']}
    pairs[suite]={}
    for first,second in [('ALIGN','MLR'),('FULL_S','MLR'),('FULL_S','ALIGN')]:
        name=first+'_minus_'+second;pairs[suite][name]={}
        for seed in [41,42,43]:
            left=records['%s_%s_%d'%(suite,first,seed)];right=records['%s_%s_%d'%(suite,second,seed)]
            index=lambda rr:{(x['stage'],x['task'],x['initial_index']):x for x in rr}
            l,r=index(left),index(right);assert l.keys()==r.keys()
            for key in l:
                assert all(l[key][field]==r[key][field] for field in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed'])
            changes={}
            for stage in range(6,10):
                for task in range(6,stage+1):
                    rr=[(l[stage,task,i]['success'],r[stage,task,i]['success']) for i in range(20)]
                    sf=sum(x and not y for x,y in rr);fs=sum(y and not x for x,y in rr)
                    changes['%d_%d'%(stage,task)]=dict(improved=sf,degraded=fs,both_success=sum(x and y for x,y in rr),both_failed=sum(not x and not y for x,y in rr),delta_pp=5*(sf-fs))
                    if fs>sf:negative.append(dict(suite=suite,comparison=name,seed=seed,stage=stage,task=task,delta_pp=5*(sf-fs),improved=sf,degraded=fs))
            lm=results['%s_%s_%d'%(suite,first,seed)]['metrics'];rm=results['%s_%s_%d'%(suite,second,seed)]['metrics']
            pairs[suite][name][str(seed)]=dict(metric_delta_pp={k:lm[k]['percent']-rm[k]['percent'] for k in lm},cells=changes,initial_and_standard_Normal_stream_hashes_exact=True)
out=dict(status='PASS_ACCEPTED_STRIPPED_S_ANALYSIS',observed=datetime.datetime.now().astimezone().isoformat(),results=results,summary=summary,paired=pairs,negative_cells=negative,new_formal_episodes=2400,unique_full_S_baseline_episodes=1200,full_S_retrained=False,full_S_reevaluated=False,shared_full_S_pairing_not_independent_replications=True,actual_cpu_acceptance=a,limits=['Full S has RA/execution/retention/support/gating and additional training interactions; this run isolates ALIGN only inside the stripped full-S implementation','Historical main-table MLR uses its original evaluator and is background context','CUDA hardware differences recorded; same development seeds and evaluation initial states reused','Three seeds do not establish universal taskwise dominance'])
(R/'FINAL_ANALYSIS.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rows=[]
for suite in summary:
    for method in ['MLR','ALIGN','FULL_S']:
        rows.append('|%s|%s|%s|'%(suite,method,'|'.join('%.3f ± %.3f'%(summary[suite][method][k]['mean_percent'],summary[suite][method][k]['sample_sd_percent']) for k in ['FWT','NBT','AUC','OldFinal','NewFinal','FinalAll'])))
report='# 完整S实现剥离后的MLR/ALIGN配对实验最终报告\n\n12个新run通过实际48CP/12完整恢复与2400正式初态/噪声、预算独立验收。三seed均值±样本标准差，单位%。MLR与ALIGN均来自相同完整S实现，删除RA/E/R/q/gate/训练采集；两者仅监督动作与合法窗口定义不同。完整S使用已接受的1200条基线记录，无额外训练或补评。\n\n|套件|方法|FWT↑|NBT↓|AUC↑|OldFinal↑|NewFinal↑|FinalAll↑|\n|---|---|---:|---:|---:|---:|---:|---:|\n'+'\n'.join(rows)+'\n\n完整逐seed矩阵、改善/退化配对和所有负单元见FINAL_ANALYSIS.json。完整S共享的1200基线回合各用于两种比较，不计为2400独立基线。历史主表MLR的实现与评估差异保留为背景，不混为本轮新MLR。\n'
(R/'REPORT.md').write_text(report,encoding='utf-8')
status=dict(status='COMPLETE_ACCEPTED_ANALYSIS_PENDING_PUBLICATION',observed=out['observed'],evidence_directory=str(d.resolve()),actual_final_cpu_acceptance=True,contract=a['contract'],summary=summary,totals=a)
(R/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(status=out['status'],summary=summary,negative_cells=len(negative)),ensure_ascii=False))
