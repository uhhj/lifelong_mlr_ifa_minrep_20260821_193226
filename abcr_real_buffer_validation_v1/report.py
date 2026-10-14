import argparse,csv,collections,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import *

def rows(p):return list(csv.DictReader(p.open()))
def ci(vals):
    if not vals:return None
    x=np.array(vals);rng=np.random.RandomState(42);b=x[rng.randint(len(x),size=(2000,len(x)))].mean(1)
    return dict(n_trajectories=len(x),mean=float(x.mean()),ci95=np.quantile(b,[.025,.975]).tolist())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();p=a.out
    coverage=rows(p/'R1_COVERAGE.csv');batch=rows(p/'R1_NATIVE_BATCH_COVERAGE.csv');physical=rows(p/'R2_PHYSICAL_AUDIT.csv');r3=rows(p/'R3_NONOVERLAP.csv');trajectories=rows(p/'R3_TRAJECTORY.csv');r4=rows(p/'R4_RESULTS.csv')
    labels={r['pair_id']:r for r in rows(p/'semantic_audit/AI_SEMANTIC_LABELS.csv')};selection=read(p/'AUDIT_SELECTION_PRIVATE.json')['pairs'];joined=[]
    for r in selection:joined.append({**r,**labels[r['pair_id']]})
    # 'label' is action label in selection but semantic label in annotation: preserve both explicitly.
    for r,original in zip(joined,selection):r['action_label']=original['label'];r['semantic_label']=r.pop('label')
    table(p/'R2_SEMANTIC_AUDIT.csv',joined);group=[]
    for task in [7,8,9]:
        for action in ['SHARE','BIF']:
            for passed in [True,False]:
                rr=[r for r in joined if r['task']==task and r['action_label']==action and r['physical_pass']==passed]
                counts=collections.Counter(r['semantic_label'] for r in rr);group.append(dict(task=task,action_label=action,physical_pass=passed,n=len(rr),**{k:counts[k] for k in ['COMPARABLE','NOT_COMPARABLE','UNCERTAIN']}))
    write(p/'R2_REPORT.json',dict(physical_population=physical,semantic_strata=group,annotator='AI, not human',limitations=['Blinded visual judgment is not human ground truth or simulator contact-state validation.','One sample per trajectory pair first: audit is diversity-oriented, not uniform random pairs; no unbiased population semantic precision or CI claimed.','Pose gate checks EEF/gripper only; semantic review inspects target relation.','Physical thresholds are frozen engineering criteria, not a proven task equivalence theorem.']))
    differences=[]
    for seed in [42,43,44]:
        for task in [7,8,9]:
            for gate in ['quantile','physical']:
                for label in ['SHARE','BIF']:
                    subset=[r for r in trajectories if int(r['seed'])==seed and int(r['task'])==task and r['gate']==gate and r['label']==label]
                    by={(int(r['q']),int(r['demo'])):r for r in subset};delta=[]
                    for demo in range(10):
                        one,four=by[(1,demo)],by[(4,demo)]
                        if one['rate'] and four['rate']:delta.append(float(four['rate'])-float(one['rate']))
                    differences.append(dict(seed=seed,task=task,gate=gate,label=label,paired_q4_minus_q1=ci(delta)))
    write(p/'R3_REPORT.json',dict(cells=r3,trajectory_paired=differences,meaning='later opposite-command rate, not true semantic false sharing; q1/q4 initial subsets differ, later comparison segment identical and disjoint',limitations=['q4 sees three more and more recent actions before the shared future segment; cannot isolate averaging from recency.','Fixed q4-calibrated threshold applied to both lengths; no claim of globally optimal q.','Repeated old endpoints and shared raw trajectories across feature seeds are not independent replicates.']))
    r4summ=[]
    for task in [7,8]:
        for label in ['SHARE','BIF']:
            rr=[r for r in r4 if int(r['task'])==task and r['action_label']==label]
            r4summ.append(dict(task=task,label=label,n=len(rr),harm=sum(r['old_outcome']=='HARM' for r in rr),benefit=sum(r['old_outcome']=='BENEFIT' for r in rr),neutral=sum(r['old_outcome']=='NEUTRAL' for r in rr),mean_gradient_cosine=float(np.mean([float(r['gradient_cosine']) for r in rr])) if rr else None))
    strict=0;rawsign=0
    for r in r4:
        dy=float(r['old_delta']);tau=float(r['deadband'])
        strict+=abs(dy)>tau and np.sign(dy)==np.sign(float(r['old_gradient_dot_actual_update']))
        rawsign+=abs(dy)>tau and np.sign(dy)==-np.sign(float(r['gradient_cosine']))
    strict=int(strict);rawsign=int(rawsign)
    write(p/'R4_REPORT.json',dict(status='INCONCLUSIVE_SMALL_SELECTED_SAMPLE',groups=r4summ,n_pairs=len(r4),complete_states=2,seed=45,tasks=[7,8],task9='NOT_RUN_NO_COMPLETE_OPTIMIZER_STATE',actual_displacement_sign_matches=strict,negative_gradient_cosine_sign_matches=rawsign,limitations=['Two shared anchor states; five pairs are not five independent training runs.','Single-new-window BC diagnostic with original AdamW moments; not a submitted 10-window mixed MLR+IFA update.','Deterministic CPU eval; no claim of bitwise historical GPU training-trajectory reproduction.','Only retained, uniquely matched historical plan windows used; no replacement after eligibility failures.','Action labels describe genuine future actions after terminal observation; official BC target is native aligned terminal expert action.','A BIF pair that decreases old NLL disproves a universal BIF=>optimization-harm rule; rates are not estimated reliably.'],formal_training_resumed=False))
    summary=dict(experiment='abcr_real_buffer_validation_v1',status='COMPLETED_WITH_SCOPE_LIMITATIONS',R1='COMPLETE_9_NATIVE_BUFFER_CELLS',R2='PHYSICAL_SCAN_COMPLETE_AI_SEMANTIC_AUDIT_72_NOT_HUMAN_GROUND_TRUTH',R3='NONOVERLAPPING_FUTURE_COMPARISON_COMPLETE',R4='INCONCLUSIVE_5_PAIRS_2_COMPLETE_ADAMW_STATES_TASK9_NOT_RUN',ABCR_TRAINING_SUPPORTED=False,formal_training_resumed=False,training_active=False)
    write(p/'SUMMARY.json',summary)
    # Small, reproducible scientific plots, no matrix edits.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,label in zip(axes,['share','bif']):
        for gate in ['quantile','physical','physical_object_proxy']:
            values=[np.mean([float(r[label+'_coverage']) for r in coverage if int(r['task'])==t and r['gate']==gate]) for t in [7,8,9]]
            ax.plot([7,8,9],values,'o-',label=gate)
        ax.set(title=label.upper()+' window coverage',xlabel='Current task',ylabel='Fraction');ax.set_xticks([7,8,9]);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(p/'R1_coverage.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for q in [1,4]:
        values=[np.mean([float(r['later_opposite_rate']) for r in r3 if int(r['task'])==t and r['gate']=='physical' and r['label']=='SHARE' and int(r['q'])==q]) for t in [7,8,9]]
        ax.plot([7,8,9],np.array(values)*100,'o-',label='Initial q=%d'%q)
    ax.set(xlabel='Current task',ylabel='Later BIF among initial SHARE (%)',title='Same non-overlapping future actions 5-8');ax.set_xticks([7,8,9]);ax.legend();fig.tight_layout();fig.savefig(p/'R3_nonoverlap.png',dpi=160);plt.close(fig)
    text=['# 正式 Buffer 与状态—动作关系验证 R1–R4','',
          '本轮仅执行 CPU 统计、AI 图像审查与可恢复的 BC 单步探针。未训练 ABCR、未运行 P4、未恢复正式训练。',
          '', '## R1：真实历史 Buffer', '',
          '正式历史仅含增量任务：Task7 回放 Task6；Task8 回放 Task6/7；Task9 回放 Task6/7/8。按 seeds42/43/44 原生50%窗口选择。seed42 使用已提交A1缓存的同一原生选择并逐文件校验；43/44为原配置和原选择函数确定性重建，不冒称遗留基线快照。',
          '沿用 P1 的 beta=0.5、原分位数阈值与归一化，不在新池重标定。这里的覆盖指终端帧有完整未来q4的原生窗口；分母包含无未来片段或无法唯一定位的窗口。',
          '', '|Task|原候选 SHARE/BIF 覆盖（三seed范围）|独立物理门后 SHARE/BIF|再加物体相对位置5cm代理门|', '|---|---|---|---|']
    def rangeval(task,gate,label):
        x=[100*float(r[label+'_coverage']) for r in coverage if int(r['task'])==task and r['gate']==gate];return '%.2f–%.2f%%'%(min(x),max(x))
    for task in [7,8,9]:text.append('|%d|%s / %s|%s / %s|%s / %s|'%(task,*[rangeval(task,g,l) for g in ['quantile','physical','physical_object_proxy'] for l in ['share','bif']]))
    text+=['','物理门为末端位置≤3cm、姿态≤15°、夹爪开度差≤10%，不把越界夹爪裁回。5cm物体相对位置仅为单列的几何代理，不是语义真值。','',
           '按原生混合数据集和 DistributedSampler 重建100个epoch的批次，批次数与历史lr_protocol逐阶段一致；不是强行5+5采样，也没有训练。','',
           '|Task|原批次同时有 SHARE+BIF（候选）|物理门后同时命中|','|---|---|---|']
    for task in [7,8,9]:
        vals=[]
        for gate in ['quantile','physical']:
            x=[float(r['both_batch_fraction'])*100 for r in batch if int(r['task'])==task and r['gate']==gate];vals.append('%.3f–%.3f%%'%(min(x),max(x)))
        text.append('|%d|%s|%s|'%(task,*vals))
    text+=['','## R2：独立物理与语义核验','',
           '全部候选做独立物理扫描。另对seed42事前固定的72对做双视角盲审，隐藏动作类别、相似分数和物理门结果。AI结果：27对可比、40对不可比、5对不确定。该分层且优先轨迹对多样性的审查不是总体均匀抽样，不能把27/72当无偏总体匹配准确率。',
           '常见问题是初始机械臂姿态与背景相似，但目标物体相对夹爪位置不同；抓取、搬运、放置阶段也可能错配。机器人位姿接近并不足以证明目标交互关系相同。AI不是独立人工真值；单帧也无法可靠认证接触、速度和所有隐状态。完整逐对理由与图像SHA已保存。','',
           '## R3：不重叠后续动作','',
           '同一q8可用对集合内，初始q1=第1步、初始q4=第1–4步，统一评价第5–8步。所有类别仍用原q4阈值；不补齐轨迹尾部、不重新标定。','',
           '|Task|物理门后：q1 SHARE→后续 BIF|q4 SHARE→后续 BIF|','|---|---|---|']
    for task in [7,8,9]:
        vals=[]
        for q in [1,4]:
            x=[100*float(r['later_opposite_rate']) for r in r3 if int(r['task'])==task and r['gate']=='physical' and r['label']=='SHARE' and int(r['q'])==q];vals.append('%.3f–%.3f%%'%(min(x),max(x)))
        text.append('|%d|%s|%s|'%(task,*vals))
    text+=['','这支持q4在本固定短时动作指标上较少出现后续分叉；不证明q4全局最优，也不支持“25%–40%伪共享”的预设。q4获得更多且更近的动作信息，不能把全部改善归因为去噪。轨迹级差值及区间见R3_REPORT.json；三个特征seed共用原始轨迹，不是三套独立机器人数据。','',
           '## R4：真实 AdamW 状态上的 BC 单步探针','',
           '采用seed45两处最早直接保存的完整Task7/Task8状态；不重放前序batch。Task9仅有模型权重，未伪造优化器，因此未测。R2物理+AI可比的Task7/8候选共10对，只有5对能在seed45历史原生计划中唯一定位且旧样本确实被保留；不补抽替代。',
           '完整decoder+head、官方GMM终端NLL、原AdamW矩、LR、weight decay和grad clip；统一CPU eval关闭dropout。每对只用新窗口BC产生一次候选更新，再测旧窗口损失，随后恢复模型、optimizer、scheduler、RNG和模式。它是受控单窗口探针，不是原10窗口MLR+IFA混合更新。首次对照原生observe得到参数、优化器和scheduler逐位一致。','',
           '|Pair|Task|动作标签|梯度cos|旧NLL变化|h_old·实际位移|结果|','|---|---|---|---|---|---|---|']
    for r in r4:text.append('|%s|%s|%s|%.6f|%+.6f|%+.6f|%s|'%(r['pair_id'],r['task'],r['action_label'],float(r['gradient_cosine']),float(r['old_delta']),float(r['old_gradient_dot_actual_update']),r['old_outcome']))
    text+=['',f'实际位移一阶预测符号符合 {strict}/{len(r4)}；负梯度夹角符号符合 {rawsign}/{len(r4)}。小样本不能估计泛化预测率。BIF既出现损伤也出现改善，不能把动作分类直接等同于优化关系。实际位移预测也有有限步长失配实例。',
           '动作标签来自终端观测之后的未来片段；官方BC目标仍为原生终端专家动作，未更改训练对齐。因此两种标签对象的差异同样限制解释。','',
           '## 裁决与限制','',
           '- R1：正式Buffer确有候选，但独立物理核验后覆盖显著降低，原生随机混合批次的双类供应仍稀疏。不能沿用先前含基础任务大池的高覆盖结论。',
           '- R2：存在明显关系错配风险；图像审查为AI探索性证据，尚非人工认证的状态可比性。',
           '- R3：q4在固定后续动作分叉指标上有优势；仅限此离线时间尺度与阈值。',
           '- R4：INCONCLUSIVE。5对、2个历史状态、1个seed不足以建立可靠动作关系→优化干扰预测；已有反例否定无条件等价。',
           '- 不建议依据本轮直接开始ABCR完整训练。没有修改正式buffer、BC损失、优化器算法或QP。',
           '', '所有输入/源码定向哈希、身份映射、恢复检查、逐窗口/轨迹统计与核验脚本一并保存。大特征与原图保留于服务器和本地交付目录，不纳入Git。',
           '', 'formal_training_resumed=false']
    (p/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__':main()
