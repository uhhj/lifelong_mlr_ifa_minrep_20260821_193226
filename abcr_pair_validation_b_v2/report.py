"""Render this completed base validation; no tuning or incremental measurement."""
import argparse
import csv
import json
from pathlib import Path

from abcr_pair_validation_b_v1.original_experiment_b import dump_json,sha256


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    out=args.out
    report=json.loads((out/'BASE_VALIDATION_REPORT.json').read_text())
    fr=json.loads((out/'frozen_original_rules.json').read_text())
    cal=report['roles']['calibration']
    val=report['roles']['validation']
    export=json.loads((out/'FEATURE_EXPORT_COMPLETE.json').read_text())
    checks=json.loads((out/'CROSS_ENVIRONMENT_RECHECK.json').read_text())
    c,v=cal['metrics_relative_to_ai'],val['metrics_relative_to_ai']
    # This report documents the observed complete failure to recover validation
    # positives. It does not introduce a tunable acceptance cutoff for future runs.
    assert v['tp']==0 and v['fn']>0 and v['fp']>0
    assert checks['status']=='PASS'
    summary=dict(status='STOPPED_AT_BASE_VALIDATION',measurement_validity='PASS',
        base_matching_validation='NOT_SUPPORTED_RELATIVE_TO_AI_LABELS',
        label_source='AI_VISUAL_ANNOTATION',human_verified=False,
        reason='No validation positive was recovered; the only accepted validation pair is AI-labeled nonmatch. Original protocol stops on poor semantic-state validation.',
        tau_v=fr['visual']['tau_v'],calibration=c,validation=v,
        incremental_screen='NOT_RUN_PROTOCOL_STOP',incremental_pt_feature_alignment='NOT_RUN',
        annotation_uncertain=dict(calibration=cal['unresolved'],validation=val['unresolved']),
        abcr_training_started=False,formal_training_resumed=False,
        limitation='Results relative to one AI annotator on selected physical hard pairs, not human-verified physical ground truth; threshold not retuned.')
    dump_json(out/'SUMMARY.json',summary)
    dump_json(out/'STATUS.json',summary,overwrite=True)
    annotated_fr=dict(fr, label_source='AI_VISUAL_ANNOTATION',human_verified=False,
        amendment_sha256=json.loads((out/'RUN_PROTOCOL.json').read_text())['bindings']['amendment_sha256'],
        original_rules_frozen_sha256=sha256(out/'frozen_original_rules.json'),
        action_scale_scope='unique calibration-role direct HDF5 base anchors only')
    dump_json(out/'frozen_v2.json',annotated_fr)
    text=f'''# ABCR 实验 B v2：基础校准与轨迹隔离验证

结论：**基础匹配验证未获支持，按原协议在此停止，不进入 Task7/8/9 增量筛查，不训练 ABCR。**
不是工程阻塞：563 个真实 HDF5 锚点、1126 张图像均完成离线 CPU 特征导出，数值复核通过。
本次执行遵循原协议与 AI 标签修订；没有按结果更换标签、改变门限或重新抽样。

## 结果

冻结视觉阈值 **{fr['visual']['tau_v']:.16f}**，来自校准负例双视角最小余弦的 higher Q0.95 + 1e-7。
固定物理门为 3 cm / 15° / 0.10 夹爪行程，所有 450 对原本就是按物理门筛选的待审计对，
所以本报告的 450/450 物理通过不能解释为全体跨任务窗口都有可比位姿。

| 数据角色 | AI正 / AI负 / U | TP / FP / TN / FN | 精确率 | 召回率 | 误接收率 |
|---|---:|---:|---:|---:|---:|
| 校准 | 74 / 213 / 13 | 2 / 10 / 203 / 72 | {c['precision']:.2%} | {c['recall']:.2%} | {c['fpr']:.2%} |
| 轨迹隔离验证 | 46 / 99 / 5 | 0 / 1 / 98 / 46 | {v['precision']:.2%} | {v['recall']:.2%} | {v['fpr']:.2%} |

全部指标是**相对于 AI 视觉标签**，`human_verified=false`，不是人工或完整物理状态真值。
18 个 U 保留在原始记录和覆盖率分母中，不强制二分类。逐任务对结果见 `BASE_VALIDATION_BY_TASK_PAIR.csv`。
验证集唯一通过的对来自任务 3↔5，AI 标签为负；动作规则将其归为 UNCERTAIN_ACTION。
验证中 SHARE=0、BIF=0。校准中门后 SHARE=4、BIF=1、UNCERTAIN_ACTION=7，不能用校准结果替代验证。

## 为什么停止

这不只是为了压低误接收率而损失一点召回：46 个明确正例一个也没有通过，唯一接收的也是负例。
原协议仅规定“验证显示语义状态可比性差则停止”，没有定量成功线；本报告没有新增或调节通过阈值。
按这一既有定性停止条件，当前规则不足以支持增量候选关系可靠性。

正例的分数并不系统高于负例。校准正/负中位数分别为
{cal['score_quantiles']['1'][2]:.6f} / {cal['score_quantiles']['0'][2]:.6f}，验证为
{val['score_quantiles']['1'][2]:.6f} / {val['score_quantiles']['0'][2]:.6f}。
这是实测分布。可能的解释包括表征偏重物体外观、背景，或 AI 对“关系可比”的判断有误；
本实验不能区分这些解释，也不能据此宣称视觉匹配或 ABCR 的所有路线均失效。
没有反转相似度、降低阈值、改用单视角或在增量任务上寻找有利工作点。

## 来源、隔离与数值核验

- 原始脚本 SHA 保持不变，校准和动作规则直接复用原函数。
- 原始基础 checkpoint：`object_multitask_model_ep99.pth`，SHA `{export['binding']['checkpoint_sha256']}`。
- 原生 CLIP 视觉模块 + rank8/alpha8 LoRA + image_proj，共 250 个 state 键严格加载。
- 512 维输出位置为 `image_proj(image_encoders(image))`，在 FiLM 之前。
- 原始 OpenGL 图像按照正式 robomimic 方式转 float/255/CHW，不作展示用垂直翻转；
  正式 Resize/CenterCrop/Normalize、eval forward 与独立重复逐位一致。
- 每个锚点重新读取原始像素 SHA、robot_states、物理夹爪与未来 3 步动作，与准备清单逐项相同。
- 基础校准/验证分别为 demo0–24 / demo25–49，轨迹数 {report['trajectory_counts']['calibration']} / {report['trajectory_counts']['validation']}，交集 0。
  编码器基础预训练已使用基础演示；隔离指标签校准/验证，不能称编码器未见验证演示。
- 动作 IQR 只用校准角色的基础锚点，未用验证动作。阈值封存后才计算验证分数。
- 输入、代码、checkpoint 和输出 SHA 绑定；6/6 单元测试通过。
- 450 对独立标量复算通过。另在 Windows Python/NumPy 环境只从保存数组重算阈值与混淆矩阵，
  不调用原数值代码，结论一致。不是独立人工标注或完整原始图像端到端复现。

## 不确定性与证据范围

按任务内轨迹重采样 2000 次，样本对权重为两个端点轨迹的抽样次数之积。
验证 recall/precision 的经验 bootstrap 区间退化为 [0,0]：这仅因本样本没有任何 TP，
**不表示总体真实召回率被证明精确等于零**。precision 仅有 1 个接收样本，尤其不稳定。
有效重采样次数与区间原值保留在 `BASE_VALIDATION_REPORT.json`。
数据为经过物理筛选的刻意采样对，不能估计全体窗口覆盖率。
单 AI 标注可能存在系统偏差，没有人工核验、标注者间一致性或完整动态状态真值。

## 成本与保留资产

CPU 特征构造（不含之前原始文件哈希读取）共 {export['seconds']:.2f} 秒，
峰值进程 RSS {export['peak_rss_kib']/1024**2:.3f} GiB，CUDA 未初始化。
预处理/编码器来源、输入清单与哈希见 `EXPERIMENT_CONTEXT.json`、`INPUT_ASSET_CHECK.json`、`FEATURE_BINDING.json`。
服务器保留 `reports/abcr_pair_validation_b_v2/` 中合并 records.npz 和逐锚点缓存，便于复核，
模型与 HDF5 不提交到 Git。仓库提交小报告、逐对表、冻结阈值及复算代码。

增量 `.pt` 的像素/特征映射还未执行，不能把本次直接 HDF5 锚点核验冒称其已完成。
Task6↔Task8 增量复核、低 A 分数增量审计、GOAL/LIBERO-50 均未运行。
后续如研究新度量，应先另立基础任务协议并独立人工复核标签，而不是继续消耗增量数据调参。

`formal_training_resumed=false`。
'''
    (out/'FINAL_REPORT.md').write_text(text,encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(out/'BASE_PAIR_SCORES.csv',encoding='utf-8',newline='') as f:
        rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,role in zip(axes,['calibration','validation']):
        for label,color in [('0','#bc4749'),('1','#386641')]:
            values=[float(r['min_cos']) for r in rows if r['role']==role and r['label']==label]
            ax.hist(values,bins=[i/20 for i in range(21)],alpha=.55,color=color,label='AI label '+label)
        ax.axvline(fr['visual']['tau_v'],color='black',linestyle='--',label='Frozen threshold')
        ax.set(title=role,xlabel='Minimum of two pre-FiLM cosine similarities',ylabel='Selected pair count',xlim=(0,1))
        ax.legend(fontsize=8)
    fig.suptitle('Base-task validation: AI labels, not human-verified ground truth')
    fig.tight_layout()
    fig.savefig(out/'base_visual_score_distributions.png',dpi=160)
    plt.close(fig)
    selected=[p for p in out.iterdir() if p.is_file() and p.suffix in ('.json','.csv','.md','.png') and p.name!='COMPLETE.json']
    dump_json(out/'COMPLETE.json',dict(status='BASE_VALIDATION_COMPLETED_PROTOCOL_STOP',
        report_files={p.name:sha256(p) for p in selected},
        derived_feature_files=export['outputs'],formal_training_resumed=False))


if __name__=='__main__':
    main()
