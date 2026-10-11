"""Render descriptive evidence from a completed, read-only event audit."""
import csv
import hashlib
import json
import os
from pathlib import Path


def write(p, value):
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, p)


def main():
    root = Path(__file__).resolve().parents[1]
    report = root/'reports/v1_object_tsr'
    out = report/'posthoc_failure_audit_v1'
    a = json.loads((out/'AUDIT.json').read_text(encoding='utf-8'))
    result = json.loads((report/'SUMMARY.json').read_text(encoding='utf-8'))
    sample = list(csv.DictReader((out/'SAMPLING_EXPOSURE.csv').open()))
    coverage = list(csv.DictReader((out/'EPOCH_COVERAGE.csv').open()))
    epochs = list(csv.DictReader((out/'EPOCH_EVENT_SUMMARY.csv').open()))
    curve = list(csv.DictReader((out/'BASELINE_LOSS_COMPARISON.csv').open()))
    anchor = list(csv.DictReader((out/'IFA_ANCHOR_EXPOSURE.csv').open()))
    assert len(coverage) == 1000 and len(epochs) == len(curve) == 400
    assert sum(int(x['batches']) for x in epochs) == 105500
    assert sum(int(x['actual_draws']) for x in sample) == 1052400
    for task in range(6,10):
        t = a['stages'][str(task)]['totals']
        assert sum(int(x['draws']) for x in coverage if int(x['stage']) == task) == t['current_windows']+t['old_windows']
        assert sum(int(x['actual_ifa_current_window_terms']) for x in anchor if int(x['stage']) == task) == t.get('ifa_window_terms',0)
    assert all(x['exactly_equal_to_task_0_pt'] == x['unique_anchors'] for x in a['language_anchor_checks'])
    paired = list(csv.DictReader((report/'PAIRED_TRIALS.csv').open()))
    outcomes = []
    for stage in range(6,10):
        for task in range(6,stage+1):
            rows = [r for r in paired if int(r['stage'])==stage and int(r['task'])==task]
            assert len(rows)==20
            v = sum(r['v1_success']=='True' for r in rows)
            b = sum(r['baseline_success']=='True' for r in rows)
            assert 5*v == result['matrices']['v1'][stage-6][task-6]
            assert 5*b == result['matrices']['baseline'][stage-6][task-6]
            outcomes.append(dict(stage=stage,task=task,v1_successes=v,baseline_successes=b,trials=20,
                v1_only=sum(r['v1_success']=='True' and r['baseline_success']=='False' for r in rows),
                baseline_only=sum(r['baseline_success']=='True' and r['v1_success']=='False' for r in rows)))
    sampling_table = '\n'.join('| {stage} | {source_task} | {p0:.2f}% → {p:.2f}% | {actual:.2f}% | {native:,} → {actual_n:,} | {change:+.2f}% |'.format(
        stage=x['stage'],source_task=x['source_task'],p0=100*float(x['p0_old_only']),p=100*float(x['p_old_only']),
        actual=100*float(x['actual_share_old_only']),native=int(x['native_schedule_draws']),actual_n=int(x['actual_draws']),change=float(x['change_percent']))
        for x in sample if x['role']=='replay')
    ifa_table='\n'.join('| %s | %s | %s | %s | %.3f%% | %.5f |' % (t,
        '无' if not s['selected_old_task_calls'] else ','.join('旧'+k for k in s['selected_old_task_calls']),
        format(s['totals'].get('ifa_batches',0),','),format(s['totals'].get('ifa_window_terms',0),','),
        s['active_fraction_per_current_window_term']*100,s['mean_ifa_added_per_all_batch']) for t,s in a['stages'].items())
    text = '''# V1-OBJECT seed42 失败复盘：回放与 IFA 曝光

本报告只读取既有日志、配置、缓存张量及结果；不构造模型，不运行前向/反向，不训练、不重评 rollout，不修改任何实验源码或历史结果。

## 结论与证据等级

**最强证据指向：Task9 阶段主动削减了 Task8 的回放，伴随更好的 Task9 获取和更差的 Task8 保持。Task6 的退化不能用“回放不足”解释。** 任务来源路由还间接改变了历史 IFA 的正参考身份，因此并非只有各任务 BC 曝光量在变化。现有日志能定位这些变化，但不能隔离每条通路的因果贡献。

未发现概率实现错误、漏跑 epoch 或减少当前任务样本：400 个 epoch 原生顺序/current-index SHA 全匹配；105,500 个 batch 的新旧槽位和当前样本 ID 全匹配；所有逐事件来源计数与采样日志、总报告一致。Task6/7 的实际完整 batch 身份序列均等于原生 sampler，实质路由干预从 Task8 才开始。

证据标记：**实测**＝既有训练/rollout 记录；**重建**＝在同一缓存、原生顺序和 seed 下 CPU 重建 sampler，非历史基线逐步日志；**源码推论**＝由已验证输入与现行实际源码直接推出；**推测**＝仍无法用现有数据证明因果的解释。

## 1. 成功率变化发生在哪里

以下均为 PRIMARY_EP99、每格 20 次，比较本轮配对重评的基线保存 checkpoint；不要混用旧历史矩阵。

| 学完阶段 | V1：Task6 / 7 / 8 / 9 | 配对基线：Task6 / 7 / 8 / 9 |
|---|---|---|
| Task6 | 30 / — / — / — | 35 / — / — / — |
| Task7 | 40 / 90 / — / — | 50 / 100 / — / — |
| Task8 | 20 / 95 / 95 / — | 35 / 100 / 100 / — |
| Task9 | 0 / 90 / 65 / 95 | 20 / 95 / 100 / 80 |

- 最终 Task9：+15 pp，对应同初态配对 3 次仅 V1 成功、0 次仅基线成功。
- 最终 Task8：−35 pp，0 次仅 V1 成功、7 次仅基线成功。V1 在进入 Task9 前是 95%，之后降至 65%；基线同期 100%→100%。因此 30 pp 是该阶段新增的保持差距，另 5 pp 已在之前存在。
- 最终 Task6：−20 pp，0 次仅 V1 成功、4 次仅基线成功。V1 从 Task7 后 40%→20%→0%；基线 50%→35%→20%。Task8/9 两阶段各多退化 5 pp，且更早已有差距。
- 最终 Task7：90% 对 95%。它在 Task8 阶段也被大幅削减回放，但没有像 Task8 一样崩落；不能把采样次数与成功率理解为简单线性关系。

总体 AUC 72.2917 对 78.3333（−6.0417 pp），四阶段 FWT 77.50 对 78.75（−1.25 pp），最终十任务均值 30 对 35。Task9 单格改善不等于总体迁移改善。只有一个 seed、每格 20 次，本文不宣称统计因果显著性。

## 2. 冻结概率与真实次数

概率均为**旧样本槽位内部的条件概率**，不是全部 batch 的占比。原生次数由同缓存的原生无放回 sampler 重建：每个缓存窗口每 epoch 恰好出现一次。当前可用的历史基线仅有 epoch 聚合日志，无逐 batch 来源记录；重建数不能冒充历史逐步实测。

| 学习阶段 | 回放来源 | 原 p0 → 设定 p | 实际旧槽位占比 | 原生预算 → V1 实测次数 | 变化 |
|---|---|---|---|---|---|
SAMPLING_TABLE

当前 Task6/7/8/9 窗口分别出现 156,200 / 148,000 / 156,000 / 131,900 次，均完全未变；旧窗口总次数 460,300 也未变。不是给 Task9 增加了新样本或训练步，而是重新分配了旧样本组成。

Task9 与固定参考签名的余弦：旧6=0.25295、旧7=0.25610、旧8=0.03362。固定 τ=0.1、ε=0.5 的带符号 softmax 混合把旧8从 33.90% 压到 19.62%。实际 19.6975%，与设定吻合。旧8在全部训练窗口中的占比从 21.5470% 降至 12.5204%。

Task6 在三个后续阶段累计回放 283,952 次，而原生预算为 234,300，**增加 21.19%**；Task8 阶段增加43.11%、Task9阶段增加20.46%。因此否定“Task6因为被少抽而退化”。

## 3. 覆盖范围和重复性

- Task8 阶段：旧6每 epoch 全覆盖781个缓存窗口，额外抽样为重复；旧7平均只见403.3/740=54.50%，每 epoch范围355–450。
- Task9 阶段：旧6、旧7每 epoch全覆盖其781/740个窗口并重复；旧8平均只见453.24/780=58.11%，范围401–511。
- 100个epoch累计，各旧任务缓存成员仍100%至少见过；不能说删掉了某些窗口。变化是覆盖频率和epoch内重复结构。旧8每个窗口累计40–76次，而原生恰100次；旧6在Task9每窗口108–138次。
- 缓存仍只有原始50%窗口，没有补充新的轨迹/状态。已保存特征身份没有可核验的trajectory映射，本轮不能报告轨迹覆盖或关键状态覆盖改善。更多重复不等于覆盖更多机器人状态。

## 4. IFA 实际曝光：次数、激活与参考身份必须分开

IFA 只处理当前任务窗口的末端表示；旧任务提供固定语言参考。以下“窗口项”是反复训练中的项数，不是独立样本数；active按当前窗口加权。平均附加loss按所有batch平均。零曝光阶段不把active=0解释成已满足约束。

| 阶段 | 当前任务实际使用的旧参考 | 有IFA的batch | IFA当前窗口项 | 正hinge比例 | 每batch平均加权IFA |
|---|---|---:|---:|---:|---:|
IFA_TABLE

**实测：Task9 的36,200个batch全部没有IFA项。** 不是后期偶尔失活。Task8使用旧7，Task7使用旧6。选中的旧任务并不需要出现在当前batch：Task8有7,704个IFA batch没有采到任何旧7窗口，仍正常计算旧7参考项。不能把旧7回放减少45.5%写成“旧7相关IFA减少45.5%”。

固定mask来自完整缓存及当前特征，非本epoch路由频率。按同一冻结输入与源码，重建原生sampler的IFA调用数/当前窗口项相同；但**历史基线的实际hinge大小、激活率和IFA梯度没有保存，无法做实测的baseline→V1梯度变化比较**。现有缓存同规则重建不等于取回已缺失的历史日志。

### 隐含耦合：混合batch首样本充当正参考

源码 `native_observe.py` 沿用了原 `base.py` 的语义：对当前任务所有窗口，正参考取混合batch的 `cur_feat[0,0,0,:]`，负参考取选中旧任务的 `feature_dir/<old>/0.pt`。`ifa_own_reference_fix` 未启用。本轮没有修改此行为。

Task8 的30,807个IFA batch中：

| 正参考来自 | 原生sampler重建 | V1实际 | 实际占比 |
|---|---:|---:|---:|
| 当前Task8 | 15,595 | 15,595 | 50.62% |
| 旧Task7（也是负参考） | 7,318 | 4,060 | 13.18% |
| 旧Task6 | 7,894 | 11,152 | 36.20% |

旧6正参考batch增加41.27%；旧7正参考batch减少44.52%。实际IFA中还有51,020个当前Task8窗口项，正参考取自Task6、负参考取自Task7。这是已观测到的**参考语义曝光变化**，不能等同于“旧6受到了更多保护”。

进一步只读取了所有实际使用过的唯一首样本语言张量：Task7阶段781+1,480项；Task8阶段781+736+1,560项，全部与对应任务0.pt逐位相同。

因此，当正参考恰是被选为负参考的旧任务时，两参考张量完全相同。该hinge的两个角度项在数学上相消，只剩常量margin，不能凭active=100%当作有方向的保护梯度。Task7有7,844个这种batch；Task8有4,060个。后者平均hinge=0.276162且active=100%。这是**输入核验+源码推论**；本轮没有额外反向传播实测梯度。其他batch的有效梯度幅度/方向仍未知。

Stage8中使用非相同正负参考的batch从原生重建23,489增至实测26,747，但这只是非退化参考调用次数，不是梯度有效性认证。路由不仅改变BC任务权重，也改变了IFA对当前任务表示的拉/推目标。不能将本结果干净解释为“只改变任务BC比例”的独立消融。

## 5. 尚未干预的阶段已经有轨迹差异

V1前两阶段完全使用原生采样；同状态单步工程parity已通过。但完整训练轨迹与历史基线不是逐位复现：Task6 epoch0平均总loss差1.33636e−7，epoch1差−0.00131351，epoch2差−0.0565341。Task6 ep99差+0.0117363，Task7 ep99差−0.137075。

400个epoch的batch数和末端LR均匹配；没有预算差异。历史CSV列名`mean_bc_loss`实际上累积了observe返回的BC+IFA总loss，本报告与V1的`total_loss`比较，不误当纯BC。

这证明对照在路由介入前已经有数值轨迹/表现差异，但现有日志不能进一步区分设备/内核非确定性、环境差异或尚未记录的执行上下文。不能未经证据称为GPU差异，也不能由单步parity宣称历史训练全轨迹相同。该限制必须计入Task6最终归因。

## 6. 为什么Task9上升、Task8和Task6下降

**Task8：证据最完整，但仍是机制解释而非独立因果检验。** 固定预训练签名认为旧8与新9最不相似，于是旧8少抽32,676次、每epoch覆盖降至58.11%；恰在这一阶段旧8成功率下降30 pp而基线保持不变，Task9获得3/20额外成功。合理解释是优化器更偏向新9及较相容的旧6/7，减少了对旧8的维持压力。低相似度不是“无需保护”，也不是旧技能安全性指标。**不能证明这32,676次减少单独造成了全部30 pp损失**，因为阶段起点和其他路由/IFA历史也不同。

**Task6：只能排除样本数量不足，无法定位唯一根因。** 它被加量、每epoch全覆盖，仍逐阶段下降，说明按固定参考梯度“与新任务更相似”不保证这个旧任务在当前学生/AdamW状态中被维持。Task6起始闭环表现本已较弱；新增曝光只重复原缓存，并无新状态覆盖。共享参数竞争、缓存拟合与rollout分布差异、前期IFA参考偏移、早于干预的数值轨迹差异，均为可能因素；现有日志不足以排列其因果贡献。

IFA也不是显式旧策略/旧loss守恒约束：Task8的IFA处理当前Task8表示，Task9无IFA。不能因Task6被更多回放或曾是IFA参考，就推出它的闭环技能应被保护。Task9阶段没有IFA不能单独解释V1比基线差，因为这是同冻结输入规则的共同性质，而非V1新关闭了一项保护。

训练中旧样本聚合BC仍下降，但只保存了“所有旧任务混合均值”，没有每旧任务的逐样本loss/梯度。不同来源权重变化后，该均值也不可用来声称Task6拟合改善。日志中99%以上batch触发既有100范数裁剪；更不能把采样数量比例直接当作AdamW最终位移贡献比例。IFA标量很小或hinge长期为正同样不能证明其梯度无关紧要/主导退化。

## 7. 本轮能确认与不能确认

能确认：采样实现兑现协议；Task8被显著降频；Task6未被饿死；Task9当前数据预算未增加；Task9无IFA；Task8的IFA正参考组成随路由改变；整体结果是单seed的获取—保持权衡，原推进门未通过。

不能确认：路由各通路的独立因果贡献；Task6具体在哪类状态失效；签名是否在当前学生上依然对齐；历史基线IFA梯度有多强；这些差异是否跨seed稳定。不要以缺失的逐任务梯度/轨迹日志编造原因，也不要把Task9单格上升解释成总体成功。

没有启动新实验、补seed、重训基线或修正IFA。后续是否测试任何改动，另行决定。本次仅归档取证。

## 证据文件

- `AUDIT.json`：各阶段全量计数与校验。
- `SAMPLING_EXPOSURE.csv`、`EPOCH_COVERAGE.csv`：概率、次数、窗口级覆盖与重复。
- `IFA_ANCHOR_EXPOSURE.csv`、`LANGUAGE_ANCHOR_CHECK.json`：正参考组成与缓存语言张量核验。
- `EPOCH_EVENT_SUMMARY.csv`：全部400epoch的日志聚合。
- `BASELINE_LOSS_COMPARISON.csv`：既有基线与V1训练曲线，不新增训练。
- `INPUT_PROVENANCE.json`：本次400个输入事件文件及相关少量源码的定向SHA；非全仓hash。
- `REVIEW_SUMMARY.json`：机器可读结论和配对试验计数。

formal_training_resumed = false
new_experiments_started = false
'''.replace('SAMPLING_TABLE',sampling_table).replace('IFA_TABLE',ifa_table)
    path=out/'FAILURE_REVIEW.md'
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(text,encoding='utf-8');os.replace(tmp,path)
    write(out/'REVIEW_SUMMARY.json',dict(
        scope='POSTHOC_LOG_AND_EXISTING_TENSOR_AUDIT_ONLY',new_experiments_started=False,
        formal_training_resumed=False,source_results_commit='e599a4defbedf250eb6829510c7c8b6deb9d474f',
        sampling_implementation='VERIFIED_AGAINST_105500_EVENTS',
        baseline_source_counts='DETERMINISTIC_NATIVE_SCHEDULE_RECONSTRUCTION_NOT_BASELINE_PER_STEP_LOGS',
        task8_underexposure='MEASURED_41.8923_PERCENT_FEWER_DRAWS_DURING_TASK9',
        task6_starvation_explanation='CONTRADICTED_BY_MEASURED_21.1916_PERCENT_MORE_CUMULATIVE_REPLAY',
        ifa_stage9='ZERO_TERMS_MEASURED',ifa_reference_semantics='MIXED_BATCH_FIRST_SAMPLE_INHERITED',
        ifa_anchor_composition_change='MEASURED_V1_VS_RECONSTRUCTED_NATIVE_SCHEDULE',
        pre_intervention_training_divergence=True,causal_attribution='PARTIAL_NOT_ISOLATED',
        paired_outcomes=outcomes,aggregate_checks='PASS',
        audit_code_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (root/'v1_object_tsr_posthoc').glob('*.py')}))
    print('Report generated; all aggregate consistency checks passed.')


if __name__ == '__main__':
    main()
