"""Descriptive tables and final report; never changes the preregistered gates."""
import argparse, collections, csv, platform, shutil
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from .common import read, write, table, atomic_text, sha


def stats(values):
    x = np.asarray(values, float)
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                p90=float(np.quantile(x,.9)), maximum=float(x.max())) if len(x) else dict(n=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); a = ap.parse_args(); out=a.out
    A=read(out/'R5_A_REPORT.json'); B=read(out/'R5_B_REPORT.json'); C=read(out/'R5_C_REPORT.json')
    integrity=read(out/'INPUT_AND_RESULT_INTEGRITY.json'); assert integrity['status']=='PASS'
    starts=read(out/'START_MANIFEST.json')['starts']; b=[read(p)['row'] for p in sorted((out/'B').glob('*/*.json'))]
    c=[read(p) for p in sorted((out/'C').glob('*/*.json'))]; native={r['start_id']:r for r in c if r['arm']=='MLR'}
    details=[]; trajectories=[]; task_details=[]
    for r in c:
        original={(v['kind'],v['origin']):v for v in native[r['start_id']]['observations']}
        groups=collections.defaultdict(list)
        for v in r['observations']: groups[v['kind']].append(v)
        m=dict(start_id=r['start_id'],seed=r['seed'],task=r['task'],arm=r['arm'],pair_counts=r['pair_counts'])
        for kind,vv in groups.items():
            x=np.array([v['delta'] for v in vv]); y=np.array([original[kind,v['origin']]['delta'] for v in vv]); db=np.array([v['deadband'] for v in vv]); harm=x>db; old_harm=y>db
            m[kind]=dict(**stats(x),mean_positive_damage=float(np.maximum(x,0).mean()),harm_fraction=float(harm.mean()),
                positive_damage_delta_from_native=float((np.maximum(x,0)-np.maximum(y,0)).mean()),
                mean_delta_from_native=float((x-y).mean()),harm_prevented_count=int((old_harm&~harm).sum()),
                harm_introduced_count=int((~old_harm&harm).sum()),native_harm_count=int(old_harm.sum()))
            if kind=='new':
                m[kind]['progress_retention']=float(x.mean()/y.mean()) if y.mean() < -db.mean() else None
                m[kind]['native_mean_improvement_resolvable']=bool(y.mean() < -db.mean())
            for trajectory in sorted({v['trajectory'] for v in vv}):
                xx=[v['delta'] for v in vv if v['trajectory']==trajectory]
                trajectories.append(dict(start_id=r['start_id'],seed=r['seed'],arm=r['arm'],kind=kind,trajectory=trajectory,**stats(xx)))
            for task in sorted({v['task'] for v in vv}):
                xx=[v['delta'] for v in vv if v['task']==task]
                task_details.append(dict(start_id=r['start_id'],seed=r['seed'],arm=r['arm'],kind=kind,probe_task=task,**stats(xx)))
        details.append(m)
    write(out/'R5_C_EFFECT_DETAILS.json',details); table(out/'R5_C_TRAJECTORY_METRICS.csv',trajectories); table(out/'R5_C_TASK_METRICS.csv',task_details)
    activation=[]
    for e in starts:
        ps=e['batch_pairs']; counts=collections.Counter(p['label'] for p in ps if p['eligible'])
        activation.append(dict(start_id=e['start']['start_id'],seed=e['start']['seed'],task=e['start']['task_id'],
            new_slots=sum(r['task']==e['start']['task_id'] for r in e['batch_records']),old_slots=sum(r['task']<e['start']['task_id'] for r in e['batch_records']),
            cross_pairs=len(ps),physical_pass=sum(p['physical_pass'] for p in ps),eligible_share=counts['SHARE'],eligible_bif=counts['BIF'],eligible_mid=counts['MID']))
    table(out/'R5_C_ACTIVATION.csv',activation)
    direct=[]
    for sid in sorted({r['start_id'] for r in b}):
        rr=[r for r in b if r['start_id']==sid and r['physical_pass']]; y=np.array([r['old_delta'] for r in rr]); db=np.array([r['deadband'] for r in rr])
        for key in ['action_distance','state_distance','negative_gradient_cosine','h_dot_d']:
            x=np.array([r[key] for r in rr]); valid=(abs(x)>db)&(abs(y)>db)
            direct.append(dict(start_id=sid,seed=rr[0]['seed'],task=rr[0]['task'],signal=key,n=len(rr),spearman=float(spearmanr(x,y)[0]),
                hdot_sign_accuracy=float((np.sign(x[valid])==np.sign(y[valid])).mean()) if key=='h_dot_d' and valid.any() else None,
                note='Descriptive raw signal; no fitted calibration; distance sign is not a harm classifier'))
    table(out/'R5_B_DIRECT_SIGNAL_METRICS.csv',direct)
    active=[r for r in details if r['arm']=='SHARE_ONLY' and r['pair_counts']['SHARE']]
    assert len(active)==1
    ar=active[0]; nr=next(r for r in details if r['arm']=='MLR' and r['start_id']==ar['start_id'])
    relative=-ar['old']['positive_damage_delta_from_native']/nr['old']['mean_positive_damage']
    cost=dict(device='CPU',threads=1,python=platform.python_version(),
        B_measured_seconds_sum=sum(r['seconds'] for r in b),C_measured_seconds_sum=sum(r['seconds'] for r in c),
        B_per_pair_seconds=stats([r['seconds'] for r in b]),C_per_branch_seconds=stats([r['seconds'] for r in c]),
        scope='Recorded per-unit time includes calculation and restore; excludes startup, preparation, image pack, AI review and reporting. Not a total project wall-clock estimate.',
        audit_engineering_failure='First simulator pack attempt cached too many scenes and was killed (exit137). Resume reconstructs one scene per side, deletes it, and preserves completed records. No pair or threshold changed.',
        peak_RAM='NOT_INSTRUMENTED',GPU_required=False,CUDA_initialized=False,disk_free_bytes=shutil.disk_usage(out).free)
    write(out/'EFFICIENCY.json',cost)
    context=read(out/'EXPERIMENT_CONTEXT.json'); protocol=read(out/'PROTOCOL.json')
    cells=B['primary']['seed_task_cells']; positive=sum(r['delta_mean'] is not None and r['delta_mean']>0 for r in cells)
    n_uniq_pairs=len({r['pair_id'] for r in b}); ntr=len({t for r in b for t in [r['new_trajectory'],r['old_trajectory']]})
    lines=[
        '# R5：标签可靠性、干扰预测与真实混合批次反事实验证',
        '',
        '本轮完成可用资产范围内的 CPU 离线验证，结论是不推进完整 ABCR/BIF 训练。动作 SHARE/BIF 标签没有建立稳定的实际旧任务干扰预测增益；原生混合批次中有效配对过少，干预效果证据不足。',
        '',
        '## 协议修订与范围',
        '',
        '- 用户授权以 AI 多帧审查替代人工审查；所有标签明确记为 AI，不能当作人工真值或独立人工复现。仍是同一助手，虽更换盲化 ID、隐藏类别/分数/旧标签/更新结果，不能声称独立审查者。',
        '- 用户授权补全 ABCR 定义。仅新增独立诊断代码，使用固定探索性损失；没有将它装入正式训练。',
        '- R5-A：OBJECT Task7/8/9，72 对既有分层样本；每对新旧两侧各三个时刻、两视角，仿真器只恢复保存状态并 forward，不推进仿真。',
        '- R5-B/C：seeds45/46/47 × Task7/8 × 各最早/最晚直接保存的完整 AdamW anchor = 12 起点。它们是现存 anchor 的两端，epoch30–78，不能称为完整训练早中晚覆盖。Task9 缺完整 AdamW 状态，因此未执行；没有重建优化器矩或重放前序 batch。',
        f'- 运行仓库 HEAD `{context["git_head"]}`，分支 `{context["git_branch"]}`；新增代码基于本地提交4c72198，运行源码的逐文件 SHA 见 ENGINE_BINDING.json，不能把远端 Git HEAD 单独当作运行源码版本。Python {context["python"]} / PyTorch {context["torch"]}；CPU、单线程、未初始化 CUDA。',
        '- 使用原 decoder+GMM head、官方终端 GMM NLL、原 clip/AdamW/scheduler；原生候选在 CPU 内核上逐位对齐。不是跨 GPU 历史数值轨迹逐位复现，也不是完整 MLR+IFA 收益验证：这些合法 anchor 的 IFA 为关闭状态。',
        '',
        '## R5-A：多帧局部可比性',
        '',
        '|任务|物理门通过 SHARE：可比/总数|物理门通过 BIF：可比/总数|',
        '|---|---:|---:|']
    for task in [7,8,9]:
        ss=[next(r for r in A['by_stratum'] if r['task']==task and r['physical_pass'] and r['action_label']==cl) for cl in ['SHARE','BIF']]
        lines.append('|%d|%d/%d|%d/%d|'%(task,ss[0]['comparable'],ss[0]['n'],ss[1]['comparable'],ss[1]['n']))
    lines += ['',f'全部72对：可比{A["counts"]["COMPARABLE"]}、不可比{A["counts"]["NOT_COMPARABLE"]}、不确定{A["counts"]["UNCERTAIN"]}。物理门通过子集 SHARE 15/24=62.5%（描述性 Wilson95%：42.7%–78.8%），BIF 9/24=37.5%（21.2%–57.3%）。均未达到冻结的80%且下界60%规则，不确定按未认证计。',
        '',
        '多帧中可见持物/释放/接近阶段错位，以及末端相近但目标相对位姿不同。机械臂末端、朝向、夹爪门并不等价于目标交互状态一致；也出现物理门不通过但局部相位可比的样本。不同物体身份本身未作为否决理由。该分层多样性样本不是总体精度的无偏估计；AI 可比也不证明动作关系等于知识共享/冲突。',
        '',
        '## R5-B：预测实际旧任务损失',
        '',
        f'完成150条检查点—样本对测量，涉及{n_uniq_pairs}个不同样本对、{ntr}个任务限定轨迹ID；主物理通过子集98条、29对、12检查点。其余既定样本因未唯一进入相应 seed 的真实缓存而排除，理由逐条记录，不按测量收益筛选。',
        '',
        '每次从完整状态构造单个新窗口 BC 的真实 AdamW 位移，旧窗口只评价。B 是受控 new-only 探测；真实 batch10 混合更新在 C 单独测量。评价目标是旧 NLL 的实际变化，绝非 SHARE/BIF 标签互相预测。',
        '',
        '每个测试观测的训练集同时排除同 checkpoint 和任何相同的新/旧任务限定轨迹。训练折标准化、ridge=1固定，不调参；以下为每检查点 Spearman 的中位数。',
        '',
        '|信号（固定交叉拟合）|主集 Spearman 中位数|',
        '|---|---:|']
    for name,label in [('action','q4动作距离'),('physical','物理状态特征'),('physical_plus_action','物理+动作距离'),('gradient','负原始BC梯度余弦'),('actual_update','旧梯度 · 实际AdamW位移')]:
        lines.append('|%s|%.4f|'%(label,B['primary']['predictors'][name]['median_spearman']))
    ci=B['primary']['checkpoint_mean_delta95']
    lines += ['',f'加入动作距离相对物理基线的配对 ΔSpearman 中位数为 {B["primary"]["action_added_spearman_median"]:.4f}；检查点聚类 bootstrap 的平均差95%区间为 [{ci[0]:.4f}, {ci[1]:.4f}]，只有{positive}/6个 seed×task 单元为正。各方法中位数之差不等于配对差中位数；本判断使用后者。',
        '',
        '轨迹对聚类的 MAE 改善敏感性区间也跨0。AI可比且物理通过子集仅44条、12对，能计算方向相关的检查点不足8，判证据不足；不将较好的子集描述数代替主判定。全部150条的次要分析也未建立动作距离增益。',
        '',
        '这些结果支持在本次受控探测中直接测量优化相关信号；不证明动作距离在所有非线性预测器或其他阶段必然无用。12个检查点来自6条相关训练轨迹，另有重复样本对，聚类区间仍受小样本和交叉依赖限制。',
        '',
        '## R5-C：真实混合批次反事实',
        '',
        '冻结探索性定义：z为decoder末端512维表示的L2归一化；SHARE损失均值 `1−z_new·stopgrad(z_old)`；BIF损失均值 `relu(z_new·stopgrad(z_old)−0.5)`；各权重0.05。只使用原 batch 中满足冻结门的真实样本对，空集合为0，不检索补样、不重配比例。',
        '',
        '完整 MLR / SHARE_ONLY / BIF_ONLY / ABCR 共48分支，每分支12个旧任务窗口+12个新任务窗口，并非24条独立轨迹。探针与整批次在相应任务内 trajectory-disjoint；各分支相同 RNG、native train mode，评价关闭 dropout。每次 AdamW/scheduler 各+1后恢复，正式模型和状态不提交。',
        '',
        f'12批原生新旧槽位可形成{sum(r["cross_pairs"] for r in activation)}个跨任务组合，但最终只有1个 SHARE、0个 BIF。SHARE_ONLY/ABCR仅1/12起点触发，另外11点逐位等于MLR；BIF_ONLY全部12点逐位等于MLR。不能把这些零干预当作保护成功。',
        '',
        f'唯一活跃起点 `{ar["start_id"]}`：旧探针平均正损伤由 {nr["old"]["mean_positive_damage"]:.9f} 降到 {ar["old"]["mean_positive_damage"]:.9f}，相对下降 {100*relative:.4f}%；新探针平均 ΔNLL 由 {nr["new"]["mean"]:.9f} 到 {ar["new"]["mean"]:.9f}，仍为正。该点不能声称新任务学习改善，也不能外推总体效果。完整ABCR与SHARE_ONLY相同，因为没有BIF项。',
        '',
        '冻结的最少8个活跃检查点门未满足，各干预均 INCONCLUSIVE_INSUFFICIENT_ACTIVATION。逐任务、逐轨迹、尾部p90/max、原生有害样本的防止/新引入计数见 C_EFFECT_DETAILS、C_TASK_METRICS 和 C_TRAJECTORY_METRICS；不靠 pooled 窗口制造统计量。',
        '',
        '## 验证、成本与复现边界',
        '',
        f'- 单元测试8/8通过；12个MLR分支与原生observe参数、AdamW、scheduler及RNG逐位一致；150+48次完整状态恢复通过。{integrity["exact_no_op_branches"]}个未触发辅助损失分支的结果与原生逐位一致。',
        f'- {integrity["input_files_verified"]}个定向输入文件、12个原始anchor、代码和协议哈希均验证；{integrity["original_physical_state_frames"]}个原始物理状态帧回查；72图像哈希匹配AI标签。{integrity["crossfit_test_rows"]}个交叉拟合测试行的排除规则和预测由另一路最小二乘实现复算。并非独立研究者端到端复现。',
        '- 432个计划状态帧中431个可恢复；A013旧侧+4时刻没有保存的后动作状态，明确记为未知，没有把它当作“无接触”。该对其余时刻和图像仍可审查。',
        f'- B逐单元记录合计 {cost["B_measured_seconds_sum"]:.1f} 秒，C合计 {cost["C_measured_seconds_sum"]:.1f} 秒；不含建模、映射、图包、审查、报告，不能当作全部项目用时。峰值RAM未埋点，不编造数值。',
        '- 初次图包生成因缓存多个MuJoCo场景被系统kill；保留失败日志，改为逐场景释放后只续做缺失图片，没有改样本或研究门。',
        '- 原始HDF5、特征与完整AdamW anchor未提交到Git；START_MANIFEST、INPUT_AND_RESULT_INTEGRITY给出路径和SHA。仅克隆仓库不足以端到端复现；本机deliveries保留审查图包。',
        '',
        '## 最终判断',
        '',
        '- R5-A：LABEL_RELIABILITY_NOT_ESTABLISHED（AI探索性审查，无人工真值）。',
        '- R5-B：ACTION_RELATION_INCREMENTAL_VALUE_NOT_ESTABLISHED。',
        '- R5-C：INCONCLUSIVE_INSUFFICIENT_ACTIVATION。',
        '- 不推进完整BIF/ABCR训练；本轮结束，不增加几何变体、阈值或检索抽样来争取通过。',
        '',
        '`formal_training_resumed = false`',
        '']
    atomic_text(out/'FINAL_REPORT.md','\n'.join(lines))
    summary=read(out/'SUMMARY.json'); summary.update(MEASUREMENT_VALIDITY='PASS_WITH_STATED_SCOPE_LIMITATIONS',INPUT_AND_RESULT_INTEGRITY='PASS',UNIT_TESTS='8/8',
        audit_pairs=72,optimizer_checkpoints=12,B_records=len(b),C_branches=len(c),AI_review_not_human=True,
        limitations=['No Task9 complete optimizer state','12 available anchors, not unbiased trajectory sampling','AI labels not independent human truth','One SHARE-active checkpoint, zero BIF-active checkpoints','CPU parity does not establish historical GPU trajectory parity'],
        NEXT_STEP='NO_FULL_ABCR_OR_BIF_TRAINING',report='FINAL_REPORT.md')
    write(out/'SUMMARY.json',summary)
    # Small standalone research plots; all numbers come from saved records.
    import matplotlib
    matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'figure.dpi':130})
    fig,ax=plt.subplots(figsize=(7,4)); x=np.arange(3); w=.35
    for off,cl,col in [(-w/2,'SHARE','#187a7a'),(w/2,'BIF','#d5772b')]:
        vals=[next(r['fraction'] for r in A['by_stratum'] if r['task']==task and r['physical_pass'] and r['action_label']==cl) for task in [7,8,9]]
        ax.bar(x+off,vals,w,label=cl,color=col)
    ax.set(xticks=x,xticklabels=['Task7','Task8','Task9'],ylim=(0,1.1),ylabel='AI-comparable fraction',title='Physical-PASS stratum (8 pairs / task / class)'); ax.axhline(.8,color='gray',ls='--');ax.legend();fig.tight_layout();fig.savefig(out/'r5_a_comparability.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4));names=list(B['primary']['predictors']);ax.bar(range(5),[B['primary']['predictors'][k]['median_spearman'] for k in names],color=['#888888']*3+['#e19b3b','#187a7a']);ax.set(xticks=range(5),xticklabels=['Action','Physical','Physical+action','Gradient','Actual update'],ylabel='Median checkpoint Spearman',ylim=(-1,1),title='Held-checkpoint + held-trajectory cross-fitting');ax.axhline(0,color='black',lw=.5);fig.tight_layout();fig.savefig(out/'r5_b_prediction.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));ax.bar([r['arm'] for r in C['arms']],[r['active_checkpoints'] for r in C['arms']],color=['#187a7a','#d5772b','#546ab0']);ax.set(ylim=(0,12),ylabel='Active checkpoints / 12',title='Native batch10: no retrieval or resampling');ax.axhline(8,color='gray',ls='--',label='Frozen minimum');ax.legend();fig.tight_layout();fig.savefig(out/'r5_c_activation.png');plt.close(fig)
    code={p.name:sha(p) for p in Path(__file__).parent.glob('*') if p.suffix in ['.py','.json','.md']}
    write(out/'DELIVERY_CODE_MANIFEST.json',dict(code=code,note='Final analysis/verification/report code manifest; executed numerical engine remains the original ENGINE_BINDING.'))
    print('FINAL_REPORT complete',flush=True)


if __name__=='__main__':main()
