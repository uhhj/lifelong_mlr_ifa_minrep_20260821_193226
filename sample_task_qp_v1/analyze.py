"""Descriptive paired analysis, no effectiveness decision or closed-loop claim."""
import statistics as st
from collections import defaultdict
from .common import *
def mean(xs):return sum(xs)/len(xs)
def run():
    p=read(OUT/'PROTOCOL_FROZEN.json');updates=[];rates=[];windows=[];groups=defaultdict(list);demo_groups=defaultdict(list)
    for name in p['points']:
        dest=OUT/'measurements'/name;c=read(dest/'COMPLETE.json')
        for fn,h in c['files'].items():assert sha(dest/fn)==h
        updates+=rows(dest/'UPDATE_METRICS.jsonl');rates+=rows(dest/'SAMPLE_RATES.jsonl')
        for fn in ['PROBE_WINDOWS.jsonl','AB_WINDOWS.jsonl']:
            for r in rows(dest/fn):
                bank=r.get('bank','P');x={k:r[k] for k in ['point','seed','current_task','arm','eval_task','demo','nll','initial_nll','MLR_nll','delta_vs_initial','delta_vs_MLR','epsilon']};x['bank']=bank
                x['window']=r.get('window',r.get('record_index'));windows.append(x)
                groups[name,x['arm'],bank,x['eval_task']].append(x)
                demo_groups[name,x['arm'],bank,x['eval_task'],x['demo']].append(x)
    task_rows=[];demo_rows=[]
    for target,source in [(task_rows,groups),(demo_rows,demo_groups)]:
        for key,rs in source.items():
            first=rs[0];row={k:first[k] for k in ['point','seed','current_task','arm','bank','eval_task']}
            if len(key)==5:row['demo']=key[-1]
            row.update(windows=len(rs),demos=len({r['demo'] for r in rs}),nll=mean([r['nll'] for r in rs]),initial_nll=mean([r['initial_nll'] for r in rs]),MLR_nll=mean([r['MLR_nll'] for r in rs]),delta_vs_initial=mean([r['delta_vs_initial'] for r in rs]),delta_vs_MLR=mean([r['delta_vs_MLR'] for r in rs]),
              actual_harmed_windows=sum(r['delta_vs_initial']>r['epsilon'] for r in rs),worse_than_MLR_windows=sum(r['delta_vs_MLR']>r['epsilon'] for r in rs),better_than_MLR_windows=sum(r['delta_vs_MLR']<-r['epsilon'] for r in rs),
              max_window_delta_MLR=max(r['delta_vs_MLR'] for r in rs),median_window_delta_MLR=st.median(r['delta_vs_MLR'] for r in rs))
            target.append(row)
    origin_rows=[]
    for name in p['points']:
        weights=read(ROOT/'reports/group_joint_update_v1/measurements'/name/'SELECTION.json')['mixed_batch_counts']
        for arm in ARMS:
            for bank in ['P','A','B']:
                rs=[r for r in task_rows if r['point']==name and r['arm']==arm and r['bank']==bank];old=[r for r in rs if r['eval_task']<r['current_task']];new=next(r for r in rs if r['eval_task']==r['current_task'])
                W=sum(weights.get(str(r['eval_task']),0) for r in old)
                origin_rows.append(dict(point=name,seed=new['seed'],current_task=new['current_task'],arm=arm,bank=bank,old_balanced_delta=mean([r['delta_vs_MLR'] for r in old]),old_replay_weighted_delta=sum(r['delta_vs_MLR']*weights.get(str(r['eval_task']),0) for r in old)/W if W else None,
                    new_delta=new['delta_vs_MLR'],new_native_gain=new['initial_nll']-new['MLR_nll'],worst_old_task_delta=max(r['delta_vs_MLR'] for r in old),
                    old_actual_harm_fraction=sum(r['actual_harmed_windows'] for r in old)/sum(r['windows'] for r in old),old_worse_than_MLR_fraction=sum(r['worse_than_MLR_windows'] for r in old)/sum(r['windows'] for r in old)))
    update_lookup={(r['point'],r['arm']):r for r in updates};origin_lookup={(r['point'],r['arm'],r['bank']):r for r in origin_rows};comparisons=[]
    for name in p['points']:
        for space in ['GRAD','STEP']:
            task=update_lookup[name,'TASK_'+space+'_QP'];sample=update_lookup[name,'SAMPLE_'+space+'_QP']
            row=dict(point=name,seed=task['seed'],current_task=task['current_task'],space=space,
              task_actual_norm_ratio=task['actual_norm_ratio'],sample_actual_norm_ratio=sample['actual_norm_ratio'],task_projected_norm_ratio=task['projected_norm_ratio'],sample_projected_norm_ratio=sample['projected_norm_ratio'],
              task_actual_correction_ratio=task['actual_correction_ratio'],sample_actual_correction_ratio=sample['actual_correction_ratio'],
              task_actual_sample_violations=task['sample_positive_resolvable_actual'],sample_actual_sample_violations=sample['sample_positive_resolvable_actual'])
            for bank in ['P','A','B']:
                a=origin_lookup[name,task['arm'],bank];b=origin_lookup[name,sample['arm'],bank]
                for metric in ['old_balanced_delta','new_delta','old_actual_harm_fraction']:row[bank+'__sample_minus_task__'+metric]=b[metric]-a[metric]
            comparisons.append(row)
    summaries=[]
    for seed in ['ALL',45,46,47]:
        for arm in ARMS:
            us=[u for u in updates if u['arm']==arm and (seed=='ALL' or u['seed']==seed)]
            for bank in ['P','A','B']:
                rs=[r for r in origin_rows if r['arm']==arm and r['bank']==bank and (seed=='ALL' or r['seed']==seed)]
                summaries.append(dict(seed=seed,arm=arm,bank=bank,origins=len(rs),old_balanced_delta=mean([r['old_balanced_delta'] for r in rs]),old_median_delta=st.median(r['old_balanced_delta'] for r in rs),new_delta=mean([r['new_delta'] for r in rs]),old_actual_harm_fraction=mean([r['old_actual_harm_fraction'] for r in rs]),actual_norm_ratio=mean([u['actual_norm_ratio'] for u in us]),projected_norm_ratio=mean([u['projected_norm_ratio'] for u in us]),actual_correction_ratio=mean([u['actual_correction_ratio'] for u in us]),near_stall_origins=sum(u['near_stall'] is True for u in us),actual_sample_violations_mean=mean([u['sample_positive_resolvable_actual'] for u in us])))
    for name,data in [('update_metrics.csv',updates),('sample_rates.csv',rates),('window_losses.csv',windows),('task_losses.csv',task_rows),('demo_losses.csv',demo_rows),('origin_metrics.csv',origin_rows),('paired_comparison.csv',comparisons),('summary.csv',summaries)]:csvout(OUT/name,data)
    write(OUT/'SUMMARY.json',dict(status='COMPLETE',points=36,horizon=1,rows=summaries,independent_units='Origin and development training seed; windows and demos within a carrier are correlated.',no_rollout=True))
    md=['# 逐样本与任务平均QP：单步对照','',
        '36个相同真实起点，seed45/46/47；只比较一次更新，不进行闭环或多步续训。负BC差值表示相对MLR改善。样本指原buffer时间窗口，相关窗口不当作独立实验。','',
        '| seed | arm | A旧任务ΔBC | A新任务ΔBC | B旧任务ΔBC | B新任务ΔBC | 实际位移范数/MLR | 位移≤MLR10%的起点 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for seed in [45,46,47,'ALL']:
        for arm in ARMS:
            a=next(r for r in summaries if r['seed']==seed and r['arm']==arm and r['bank']=='A');b=next(r for r in summaries if r['seed']==seed and r['arm']==arm and r['bank']=='B')
            md.append('| %s | %s | %.7g | %.7g | %.7g | %.7g | %.5g | %d |'%(seed,arm,a['old_balanced_delta'],a['new_delta'],b['old_balanced_delta'],b['new_delta'],a['actual_norm_ratio'],a['near_stall_origins']))
    md+=['','解释顺序：先看逐样本约束是否在QP变量中满足，再看AdamW或参数写入后的真实位移，再看逐窗口完整NLL，最后看隔离A/B。不能把其中一层的改善自动传递给下一层。',
      'paired_comparison.csv给出同一起点、同一干预位置下逐样本减任务平均的差；sample_rates.csv与window_losses.csv区分一阶预测和实际损失。demo_losses.csv保留演示层面的差异。',
      '本轮是单步机制诊断；无持续性与闭环效果证据，不能判断某一控制器已经有效。三个seed均为开发seed，A/B为反复使用的开发监测库。']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    print('ANALYSIS_COMPLETE',len(updates),len(windows))
if __name__=='__main__':run()
