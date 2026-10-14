"""Mechanical reporting; descriptive gates are not ABCR efficacy claims."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import write,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();out=a.out
    assert json.loads((out/'ANALYSIS_COMPLETE.json').read_text())['status']=='PASS'
    from .verify import verify
    validation=verify(out)
    p1=[];p2=[];p3=[]
    for task in [6,7,8,9]:
        stage=out/('task%d'%task)
        p1+=json.loads((stage/'P1.json').read_text());p2+=json.loads((stage/'P2.json').read_text());p3+=json.loads((stage/'P3.json').read_text())
    main_p1=[r for r in p1 if r['beta']==.5]
    comparisons={}
    for name in [('C_physical_gate_then_visual','H_full_cosine'),('unmod_visual_only','H_full_cosine'),('C_physical_gate_then_visual','H_same_physical_gate')]:
        selected=[r for r in p2 if r['beta']==.5 and r['method']==name[0] and r['baseline']==name[1]]
        comparisons[name[0]+'_vs_'+name[1]]=selected
    summary=dict(status='COMPLETED_P1_P2_P3',P1_count_gate_primary_all_stages=all(r['count_gate_primary'] for r in main_p1),
        P1_semantic_validity='NOT_ESTABLISHED_BY_QUANTILE_COVERAGE',primary_beta=.5,
        P2_paired_comparisons=comparisons,P2_FiLM_causality='NOT_IDENTIFIED_BY_C_VS_H_ALONE',
        P3_primary=[r for r in p3 if r['beta']==.5],P3_false_sharing_ground_truth='NOT_ESTABLISHED; action-horizon divergence measured',
        P4='NOT_RUN_USER_DEFERRED',independent_check=validation['status'],abcr_training_started=False,formal_training_resumed=False)
    write(out/'SUMMARY.json',summary,immutable=True)
    lines=['# OBJECT 前置实验 P1–P3','',
        'P4 按用户后续指令暂停，包括采样模拟和 20-epoch 训练。本轮不训练、不修改主干、MLR 或正式 buffer。','',
        '## 数据与时序','',
        '使用同一冻结 OBJECT 基础 checkpoint。诊断池每个旧任务 demo0–4，当前 Task6/7/8/9 各 demo0–9，全部时间步。',
        '这是本次方案指定的池（包含基础任务0–5），不是正式 MLR 实际只包含已学增量任务的回放快照。',
        '70 条轨迹的 obs[t] 与 states[t+1] 关节位置逐位一致。真正未来从 actions[t+1] 开始；目标物体坐标读 states[t+1]。',
        '这与历史 B 的同下标动作定义不同，历史报告不改写。末尾没有真实后继的项目不填充；缺失数量显式报告。',
        '官方数据转换源码同样先 step(action) 后保存 obs：[create_dataset.py](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py)。','',
        '## P1：数量覆盖，而非物理真实性认证','',
        '归一化本体状态为 7 个关节与 2 个夹爪位置，尺度只来自基础池。每阶段全量跨任务对精确计算 Q15/Q85；',
        '可比对内 q4 动作差异 Q25/Q75。动作差异为标准化 7 维命令的 RMS，运动尺度基础 IQR，下限0.1，夹爪完整幅度2。',
        '主 beta=0.5，0.3/0.7 均单独报告，不挑最优。30%状态覆盖、10%共享覆盖、10%分叉覆盖为主数量门，8%仅为原设计低端敏感性。','',
        '|当前任务|状态覆盖|共享锚点覆盖|分叉锚点覆盖|数量门|',
        '|---|---:|---:|---:|---|']
    for r in main_p1: lines.append('|%d|%.2f%%|%.2f%%|%.2f%%|%s|'%(r['task'],100*r['state_anchor_coverage'],100*r['share_anchor_coverage'],100*r['bif_anchor_coverage'],r['count_gate_primary']))
    lines+=['','分位数会按定义产生动作两端类别，因此 pair 级约25%/25%不是独立成功证据。anchor覆盖仍需实测，可能集中于重复时间步或特定旧任务。',
        '这些阈值使用当前阶段自身分布，是探索性标定，不是对独立确认集的物理匹配验证。它不会推翻实验 B 已记录的负结果。','',
        '## P2：检索物理误差','',
        'H 使用原生 CompletePolicy 当前4-token输出；实际 use_state=False，第四token是另一组FiLM(eih视觉)，不是本体状态token。',
        '为保证配对比较，两侧从相同原图、相同 checkpoint、eval、无随机增强重新导出，未混用旧随机增强缓存。',
        'C 主版本先按同一 Ds 门选候选再按视觉排序。因此其 EEF 误差优势可能来自输入本体状态，不能直接归因为去除 FiLM。',
        '补充相同物理候选门下 H 以及纯未调制视觉检索两组控制。下表是 beta0.5、按当前轨迹配对的平均误差差（负值表示前者更好）。','',
        '|任务|比较|指标|均值差(m)|轨迹bootstrap95%CI|',
        '|---|---|---|---:|---|']
    for name,rows in comparisons.items():
        for r in rows: lines.append('|%d|%s|%s|%.6f|%s|'%(r['task'],name,r['metric'],r['mean'],r['ci95']))
    lines+=['','物体误差使用目标物体free-joint原点到末端的向量/距离；不同物体原点并非统一语义接触点。',
        '每个新轨迹为统计单元，旧检索池固定且被重复使用；不能把全部时间步当独立样本，也不能仅凭一个 seed 的配对区间推广所有模型。','',
        '## P3：后续动作分叉','',
        '所有 q 固定使用各阶段 q4 得到的同一 tau+/tau-，不按 q 重设分位数。主对比使用相同 q6 完整对集，避免末尾删除造成假改善。',
        '“单步相同但后续动作不同”是时序动作分叉代理，不是经人工/环境验证的伪共享真值；不预设25%–40%或低于5%的结果。','',
        '|任务|q|q1共享后变为分叉|当前q共享、后续至q6变分叉|该q截断比例|',
        '|---|---:|---:|---:|---:|']
    fmt=lambda x:'NA' if x is None else '%.2f%%'%(100*x)
    for r in p3:
        if r['beta']==.5: lines.append('|%d|%d|%s|%s|%s|'%(r['task'],r['q'],fmt(r['q1_shared_now_bif_rate']),fmt(r['later_up_to_6_bif_rate']),fmt(r['pair_truncation_fraction'])))
    lines+=['','q6 后的分叉未观测，因此不把其0次观测当作低假阳性率。各 q 的可用数量与共同分母见原始表。','',
        '## 解释边界与产物','',
        '输出 P1/P2/P3 全部 beta 结果、逐检索锚点、逐新轨迹误差、旧任务贡献、q窗口曲线、固定阈值与运行成本。',
        'KDE只用于展示：全量阈值先计算，绘图随后固定种子抽样2000对，使用Scott带宽；不据图调阈值或挑beta。',
        '本轮不证明真实技能共享、动力学冲突、ABCR有效性、未调制特征必胜或最优q；没有改动历史结果。',
        '`formal_training_resumed=false`。']
    temporary=out/'FINAL_REPORT.md.tmp';temporary.write_text('\n'.join(lines)+'\n',encoding='utf-8');temporary.replace(out/'FINAL_REPORT.md')
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    kde_errors=[]
    for ax,r in zip(axes.ravel(),main_p1):
        with np.load(out/('task%d'%r['task'])/'P1_plot_beta0.5.npz') as z: x,y=z['visual'],z['action']
        try:
            fit=gaussian_kde(np.stack([x,y]),bw_method='scott')
            xx,yy=np.meshgrid(np.linspace(x.min(),x.max(),100),np.linspace(max(0,y.min()),y.max(),100))
            zz=fit(np.stack([xx.ravel(),yy.ravel()])).reshape(xx.shape);ax.contourf(xx,yy,zz,levels=15,cmap='viridis')
        except np.linalg.LinAlgError as e:
            kde_errors.append(dict(task=r['task'],error=str(e)));ax.scatter(x,y,s=2)
        ax.axvline(r['tau_v'],color='red',ls='--');ax.axhline(r['tau_plus'],color='white',ls='--');ax.axhline(r['tau_minus'],color='orange',ls='--')
        ax.set(title='Task %d (physical-gated pairs)'%r['task'],xlabel='Weighted visual cosine, beta=0.5',ylabel='Future q4 action RMS')
    fig.suptitle('Exploratory joint density: count thresholds, not semantic ground truth');fig.tight_layout();fig.savefig(out/'P1_JOINT_KDE.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for ax,r in zip(axes.ravel(),main_p1):
        with np.load(out/('task%d'%r['task'])/'P1_conditional_action_hist_beta0.5.npz') as z:
            counts,edges=z['counts'],z['edges']
        ax.stairs(counts/(counts.sum()*np.diff(edges)),edges,fill=True,alpha=.7)
        ax.axvline(r['tau_plus'],color='green',ls='--',label='q25')
        ax.axvline(r['tau_minus'],color='red',ls='--',label='q75')
        ax.set(title='Task %d, all comparable q4 pairs'%r['task'],xlabel='Normalized action RMS',ylabel='Density');ax.legend()
    fig.tight_layout();fig.savefig(out/'P1_CONDITIONAL_ACTION_HISTOGRAM.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for task in [6,7,8,9]:
        rows=[r for r in p3 if r['task']==task and r['beta']==.5]
        axes[0].plot([r['q'] for r in rows],[r['q1_shared_now_bif_rate'] for r in rows],marker='o',label='Task%d'%task)
        axes[1].plot([r['q'] for r in rows],[r['pair_truncation_fraction'] for r in rows],marker='o',label='Task%d'%task)
    axes[0].set(xlabel='Future horizon q',ylabel='q1-shared pairs now above tau-',title='Fixed q6-complete cohort')
    axes[1].set(xlabel='Future horizon q',ylabel='Unavailable fraction',title='No terminal padding')
    for ax in axes: ax.legend()
    fig.tight_layout();fig.savefig(out/'P3_HORIZON_CURVES.png',dpi=150);plt.close(fig)
    write(out/'PLOT_DIAGNOSTICS.json',dict(kde_errors=kde_errors,thresholds_not_fit_on_plot_sample=True),immutable=True)
    timing={}
    for name in ['LANGUAGE_COMPLETE.json','FEATURES_COMPLETE.json','ANALYSIS_COMPLETE.json','INDEPENDENT_CHECK.json']:
        if (out/name).exists():
            rec=json.loads((out/name).read_text());timing[name]={k:rec[k] for k in ['seconds','peak_rss_kib'] if k in rec}
    write(out/'EFFICIENCY.json',dict(stages=timing,device='CPU',threads=2,
        model_training_seconds=0,formal_training_resumed=False),immutable=True)
    write(out/'STATUS.json',dict(status='COMPLETED_P1_P2_P3',p4='DEFERRED_BY_USER',formal_training_resumed=False))
    write(out/'COMPLETE.json',dict(status='COMPLETED_P1_P2_P3',
        outputs={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md','.png') and p.name!='COMPLETE.json'},
        formal_training_resumed=False),immutable=True)

if __name__=='__main__': main()
