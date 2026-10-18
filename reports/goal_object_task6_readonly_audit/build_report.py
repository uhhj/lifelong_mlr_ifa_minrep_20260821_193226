"""Render audit tables from existing evidence; no policy execution."""
import json,statistics
from pathlib import Path
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text(encoding='utf-8-sig'))
x=read('RESULTS.json');d=read('DIAGNOSTICS.json');j=read('JOINT_RNG.json');b=read('BUFFER_VERIFICATION.json')
assert x['status']==d['status']==j['status']==b['status']=='PASS_READONLY'
L=['# 为什么 Coreset 在 OBJECT 增益、在 GOAL 退步：Task6 只读审计','',
 '**判断：已发现一种比“随机波动”或“漏掉事件”更具体、与遗忘时序一致的解释，但尚未证明唯一因果。** GOAL 的 Coreset 确实改善了自己优化的几何覆盖，却明显改变 Task6 回放的时间段与末步夹爪指令分布。OBJECT 上同类分布偏移较小；且 OBJECT 总体改善也不等于 Task6 每个 seed 都改善。','',
 '本轮只读现有产物，未构建模型、optimizer 或环境；0 次训练更新、0 次模型前向、0 条新增闭环、0 次新特征导出。仅 CPU 读取已有特征与 RNG 快照。','',
 '## 1. 先排除统计口径造成的反转','',
 '|比较范围|C−R AUC均值 ± 样本SD（pp）|Task6对总AUC差的贡献|','|---|---:|---:|']
for label in ('OBJECT_five_first','OBJECT_common_first','OBJECT_common_repeat43','GOAL_common'):
 v=x['suite_summary'][label];L.append(f"|{label}|{v['auc_delta_mean']:+.6f} ± {v['auc_delta_sample_sd']:.6f}|{v['task_contribution_mean']['6']:+.6f}|")
L+=['','共同 seed42–44 下反转仍存在；将 OBJECT43 repeat1 作为单独敏感性替代，OBJECT 仍平均 +3.6111 pp。历史首次结果未覆盖，未因高分剔除43。这里跨套件不是同任务配对实验，不能把差值当一个因素的因果效应。','',
 'GOAL 的总 AUC 差 −3.020833 pp，Task6 单独贡献 −3.4375 pp，Task7 为0、Task8三seed平均为0、Task9为 +0.416667 pp。负结果主要集中于 Task6 保持。','',
 '|OBJECT seed|总AUC C−R|Task6全阶段平均 C−R|Task6最终 Random / Coreset|','|---|---:|---:|---|']
for s,run in [(s,'first') for s in range(40,45)]+[(43,'repeat1')]:
 v=x['runs'][f'OBJECT_{s}_{run}'];L.append(f"|{s} {run}|{v['delta_C_minus_R']['AUC']:+.4f}|{4*v['task_auc_contribution_pp']['6']:+.4f}|{v['arms']['random']['matrix'][3][0]:g} / {v['arms']['coreset']['matrix'][3][0]:g}|")
L+=['','OBJECT 的 seed40/44 其实也有 Task6 退步，特别是40最终60→20；总体AUC由其他任务补偿。不能以OBJECT总平均有效来假定Task6已被稳定保护。','',
 '## 2. GOAL 的差距在回放开始后形成','',
 '|Seed|方法|刚学完6|学完7后|学完8后|学完9后|','|---|---|---:|---:|---:|---:|']
for s in (42,43,44):
 for a in ('random','coreset'):
  v=x['runs'][f'GOAL_{s}_first']['arms'][a];L.append(f'|{s}|{a}|'+ '|'.join(f'{r[0]:g}' for r in v['matrix'])+'|')
L+=['','两组 Task6 入库前实际模型张量与初始化RNG一致（原完成验收凭证）；stage6 的20条结果也一致。三seed平均 Random 曲线为68.33→70.00→65.00→71.67，Coreset为68.33→51.67→56.67→43.33。首次差距出现在学Task7后，此时Task6已经改用两套不同记忆。Task9自身入库发生在训练结束后，不能解释这一轮Task6退步。','',
 '按同初态比较“刚学完Task6→最终阶段”。SS=一直成功，SF=成功转失败，FS=失败转成功，FF=一直失败；这是结果翻转，不自动等于纯模型遗忘。','',
 '|Seed|方法|SS|SF|FS|FF|原成功初态保留率|','|---|---|---:|---:|---:|---:|---:|']
for s in (42,43,44):
 for a in ('random','coreset'):
  v=x['runs'][f'GOAL_{s}_first']['arms'][a]['task6_relative_to_first']['9'];L.append(f"|{s}|{a}|{v['SS']}|{v['SF']}|{v['FS']}|{v['FF']}|{v['retention_fraction']*100:.2f}%|")
L+=['','尤其seed44：Coreset初始成功的12个初态，最终只保留1个，11个转失败；Random保留8个。全三seed描述性合计Coreset有18个SF，Random有7个，不能把这些初态重复当独立训练seed。全部相邻阶段和跨方法配对数保存在RESULTS.json。','',
 '## 3. 入库是真的变了，但没有损坏或缺整个演示','',
 f"已重新核验 {sum(c['N'] for c in d['cells']):,} 个 Task6 feature 实际SHA256与原清单、{sum(v['count'] for v in b['rows']):,} 个实际buffer载荷，以及四个阶段Task6成员/哈希持续不变。HDF5全文件、元数据、feature和buffer均读取前后哈希不变。",
 'GOAL每seed997个窗口取499个，Random/Coreset重叠240/252/257个；约一半成员被改变。OBJECT每seed1562取781。所有组十条演示都保留；GOAL三个seed的Coreset还都保留了每条演示至少一个末端目标窗口。不存在“整个演示或所有结束窗口被扔掉”的证据。',
 'GOAL通过完整float32八步动作精确对应HDF5获得2991/2991唯一位置；OBJECT利用既有精确WINDOW_BINDING并逐动作重验，7810/7810可定位。文件编号不是时间编号，没有按pt文件名猜操作阶段。','',
 '|GOAL seed|Random末端目标窗口数|Coreset末端目标窗口数|Random padding窗口|Coreset padding窗口|','|---|---:|---:|---:|---:|']
for c in d['cells']:
 if c['suite']!='GOAL':continue
 counts={a:sum(w['selected'][a] and w['candidates'][0]['target']==w['candidates'][0]['length']-1 for w in c['window_bindings']) for a in ('random','coreset')}
 L.append(f"|{c['seed']}|{counts['random']}|{counts['coreset']}|{c['methods']['random']['profile']['padding_n']}|{c['methods']['coreset']['profile']['padding_n']}|")
L+=['','这更像重复监督频率下降，而不是完全失去末端支持。减少padding冗余本身未必有害；不能仅凭该数量认定它导致失败。','',
 '## 4. 几何覆盖改善，时间/控制分布却更偏','',
 '|套件|方法|原距离最大最近距离|原距离平均最近距离|时间四分区TV|演示TV|30维控制均值偏差RMS|末步正夹爪比例|','|---|---|---:|---:|---:|---:|---:|---:|']
for suite in ('GOAL','OBJECT'):
 for a in ('random','coreset'):
  v=x['diagnostic_summary'][suite][a];L.append(f"|{suite}|{a}|{v['combined_max']:.4f}|{v['combined_mean']:.4f}|{v['quartile_total_variation']:.4f}|{v['demo_total_variation']:.4f}|{v['descriptor_mean_bias_rms']:.4f}|{v['positive_gripper_fraction']*100:.2f}%|")
L+=['','GOAL取三seed平均，OBJECT取五seed平均。TV是相对完整窗口分布的总变差距离；控制偏差先按本任务完整窗口均值/总体SD标准化。它们是描述性尺度，不是成功率或显著性。覆盖半径仅在同seed标定距离内比较，跨套件绝对数值没有统一物理单位。','',
 '原visual/action距离、标定参数与最后覆盖半径均复现。GOAL最大最近距离由约0.537降到0.279；视觉、动作分量的最大距离也改善。独立的状态token余弦最近距离没有出现Coreset统一恶化，因此当前数据不支持直接归罪于“选择器没加state所以状态覆盖坏了”。状态token余弦距离不等于机器人状态空间或任务关键性认证。','',
 '**关键偏移是目标时间与夹爪命令的联合构成。** GOAL原始正夹爪目标占41.42%，Random40.55%，Coreset54.84%；OBJECT原始54.80%，Random55.47%，Coreset55.52%。正负号只表示实际动作命令，不把它当抓取成功标签。','',
 '|GOAL目标时间段|完整窗口数|Random平均保留率|Coreset平均保留率|','|---|---:|---:|---:|']
cells=[c for c in d['cells'] if c['suite']=='GOAL']
for q in range(4):
 n=cells[0]['population']['quartile_counts'][str(q)]
 rates={a:statistics.mean(c['methods'][a]['profile']['quartile_counts'].get(str(q),0)/n*100 for c in cells) for a in ('random','coreset')}
 L.append(f"|{q*25}–{(q+1)*25}%|{n}|{rates['random']:.2f}%|{rates['coreset']:.2f}%|")
L+=['','GOAL的50–75%时段249个目标全部是正夹爪命令，Coreset保留约71.35%；最初四分之一181个目标全部负命令，只保留约36.65%。最后四分区也从Random约51.14%保留降到40.79%。即Coreset把固定预算从部分前段/后段挪向中后段正命令窗口；不是单纯均匀去重。','',
 '源码实际路径 `BriefPolicy.forward → GMMHead.loss_fn` 只监督 `target[:,-1,:]`；配置里其他head名字不改变这个硬编码实际路径。故末步命令构成直接改变BC回放监督频率，不只是任意描述量。原Coreset同样以末步动作参与距离；整个8步视觉上下文仍被使用。相关BriefPolicy/BasePolicy/GMMHead和selector文件在OBJECT43与GOAL42逐字节相同，源哈希见SOURCE_COMPARISON.json。','',
 '整个八步窗口并集的原始时间覆盖，GOAL Random平均98.52%，Coreset96.33%，OBJECT两组约98.8%。这说明大量上下文仍在，主要差别是哪些末步目标被反复监督，不宜说Coreset把轨迹大段完全删除。逐demo、时间交叉表、空档和成员ID均保存。','',
 'OBJECT的控制偏差主要来自动作变化幅度；GOAL除变化幅度偏高外，正夹爪及其与平移命令的联合统计也明显偏高。两者“更分散”的记忆不一定有相同训练作用。原目标是最大化几何代表性，既不保证保持原密度，也不保证保存最需要抵抗遗忘的监督频率。','',
 '## 5. 不是简单的切换事件漏采，也不是新的IFA配对','',
 '|套件|方法|切换±4步内目标数（均值）|前后1–4步均有目标的切换数（均值）|','|---|---|---:|---:|']
for suite in ('GOAL','OBJECT'):
 for a in ('random','coreset'):
  v=x['diagnostic_summary'][suite][a];L.append(f"|{suite}|{a}|{v['near_switch_n']:.2f}|{v['events_both_sides']:.2f} / 20|")
L+=['','这里只统计正负夹爪命令转换；没有重新采用Event模块的检测/配额，也没有额外选样。GOAL Coreset两侧覆盖19/20、20/20、20/20个事件，比Random更完整。因此“再强制保留更多切换窗口”没有被当前证据支持。这个结果不否定一切事件方法，也不证明转换两侧已有正确行为。','',
 'GOAL所有seed、两组IFA选择一致：stage7=[6,7]，stage8=[6,8]，stage9空。不能说Coreset选出了不同IFA对导致本轮下降。OBJECT大多数stage8/9为空，唯一这里观察到的配对差是OBJECT42 Random stage8=[7,8]而Coreset为空；全部IFA原日志见DIAGNOSTICS.json。','',
 '相同IFA配对仍可能有不同输入分布/共享模型梯度；GOAL gamma0.7，OBJECT0.3。不能用两套件的相关性证明IFA导致反转，也不能据配对相同排除其与回放的交互。','',
 '每个已选Task6窗口在后续三个任务各完整遍历100次，成员分布差会持续进入训练；这只是样本出现次数，不宣称梯度贡献严格等权。最后Task9阶段Task6槽位占比GOAL499/3365≈14.83%，OBJECT781/3600≈21.69%。两套件都遵守自己的完整遍历协议，同套件两组比例相同；GOAL最后任务相对更大可能放大旧任务保持困难，但它不能单独解释C−R差。','',
 '## 6. RNG与结果边界','',
 '重新只读解析64份Task6评估入口的Python/NumPy/CPU/CUDA随机状态快照，所有同seed同stage的Random/Coreset入口状态一致。检查的是实际状态/张量，不是整个pth文件哈希。原评估器每stage重置，Task6是该阶段第一个评估任务，两套件入口/rollout代码相同。','',
 'GOAL除seed43 Random stage9的第二个向量组102步提前全成功外，其余Task6组均600步。该例第一组双方均600步，第二组提前结束不会倒过来改变此前Task6采样；它可能影响之后评估的其他任务。原采样仍是十路固定形状全局Normal流，未逐轨迹隔离，未保存每步噪声/状态trace；不能严格证明实际噪声和物理执行逐步一致，但也不应不加核查就把这次共同Task6退步全归为“随机流不同”。','',
 '每单元20个初态、训练seed有限，OBJECT的43重跑本身显示可观波动。GOAL43的向量提前结束需要单独披露；没有补跑或重试任何正常失败。状态元数据unknown、3090/A100/历史4090差异、调度补全、跨任务/跨套件内容不同等既有边界继续保留。','',
 '## 7. 当前证据支持的解释层级','',
 '已证实：几何覆盖改善、GOAL监督构成偏移更强、Task6遗忘集中发生于后继学习、存储未损坏、没有遗漏整条演示/全部末端、切换事件覆盖未恶化、GOALIFA选中对没有组间改变。',
 '最合理的工作假说：固定几何Coreset在GOAL Task6上过度降低了部分高频负命令与前后段目标的回放权重，偏向中后段正命令/变化较大的窗口。这些频率变化可能削弱原有策略在后继任务更新中的保持；OBJECT的任务分布下同类选择更接近原命令比例，故多样性收益可能占优。',
 '尚未证明：具体哪个窗口造成哪次失败，前段不足还是末段不足更关键，补齐它们是否提升，以及IFA交互和模型优化各占多少。不能将演示时间分区直接命名为经过验证的抓取/接触/释放成功阶段，也不能从本轮相关性给失败初态分配训练权重。',
 '因此，当前可以说“几何覆盖更好不代表持续学习回放更好，GOAL存在明确的监督构成偏移风险”；还不能说已找到唯一原因或已有有效修复。此次不设计或启动新实验。','',
 '## 8. 完整结果与复算','',
 '|Run|方法|AUC|FWT|NBT|Task6最终|Task7最终|Task8最终|Task9最终|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for key,v in x['runs'].items():
 for a in ('random','coreset'):
  m=v['arms'][a]['metrics'];L.append('|'+key+'|'+a+'|'+'|'.join(f"{m[k]['percent']:.6f}" for k in ('AUC','FWT','NBT'))+'|'+'|'.join(f'{z:g}' for z in v['arms'][a]['matrix'][3])+'|')
L+=['','RESULTS.json还含各组全部四阶段矩阵和精确分数、相邻阶段配对与跨方法配对，没有省略Task7/8/9，也没有将repeat43替换首次报告。','',
 '命令与真实退出：','', '```sh',
 '# 远端独立只读目录 /root/mlr_ifa/analyses/GOAL_OBJECT_Task6_readonly_20261018',
 'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /data/mlr_ifa/env/bin/python -u collect_readonly.py',
 '/data/mlr_ifa/env/bin/python -u inspect_joint_rng.py',
 '/data/mlr_ifa/env/bin/python verify_buffer_readonly.py',
 '# 本地或发布目录（标准库）',
 'python -X utf8 analyze.py',
 'python -X utf8 build_report.py',
 'python -X utf8 independent_verify.py',
 '```','',
 '远端三个脚本退出均0；首次提交后台采集的SSH控制通道30秒超时，但实际采集33.54秒后退出0，未重复采集。首次本地分析发生str/Path工具类型错误，修复路径转换后退出0；没有修改原科学数据或重跑训练/评估。','',
 'EVIDENCE_MANIFEST.json追踪322份原轻量证据，DIAGNOSTICS/JOINT_RNG/BUFFER_VERIFICATION追踪远端实际输入SHA256。SOURCE_COMPARISON记录检查的代码来源；ARTIFACT_MANIFEST覆盖本报告与脚本。原始权重、视觉/状态特征、演示、buffer载荷与RNG原文件均不下载/发布。窗口身份、来源哈希、派生统计及原episode结果允许发布。']
(R/'REPORT.md').write_text('\n'.join(L)+'\n',encoding='utf-8')
print('REPORT.md generated')
