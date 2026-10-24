from pathlib import Path
import json
R=Path(__file__).resolve().parent
assert json.loads((R/'RUN_STATUS.json').read_text())['status']=='COMPLETE_DIAGNOSTIC_ACCEPTED_NO_PROTECTION_TRAINING'
report='''# RA / ALIGN 共同窗口与配对轨迹诊断：完成

2026-10-24。离线对照与240新回合已独立验收；worker/supervisor自然exit0，19:29:55 CST确认无自身进程、GPU空闲、源合同及12检查点hash不变。0训练更新，没有重训、保护训练或额外seed扩展。状态COMPLETE_DIAGNOSTIC_ACCEPTED_NO_PROTECTION_TRAINING。

## 配对闭环结果

GOAL Task8，Stage8→Stage9；三seed各20初态，共同逐轨迹动作噪声。下表是新诊断协议，不能与旧global-RNG正式AUC混算。

|seed|方法|Stage8成功/20|Stage9成功/20|成功→失败|失败→成功|
|---|---|---|---|---|---|
|41|RA|16|16|1|1|
|41|ALIGN|18|12|7|1|
|42|RA|13|15|2|4|
|42|ALIGN|13|13|2|2|
|43|RA|14|15|3|4|
|43|ALIGN|14|14|4|4|

RA合计43/60→46/60（+5个百分点）；ALIGN45/60→39/60（−10个百分点）。ALIGN净下降主要由seed41贡献，42/43净变化为0但仍存在互相抵消的退化和改善。这是描述性结果，不宣称统计显著或跨seed统一退化机制。

## 原先的漂移现象不是ALIGN特有

共同452窗口/seed、总1356窗口中，RA也存在分量切换（28/32/28）、有效夹爪异号（6/6/3）。RA15次异号中10次相对专家动作的Huber改善、5次恶化；ALIGN9次均改善。RA平均Huber在41/42改善、43略恶化；RA的next-action mixture NLL三seed均恶化，不能概括为所有离线误差改善。RA原训练目标a[u]、本诊断共同目标a[u+1]，这些值不是RA训练损失。详见[离线报告](OFFLINE_REPORT.md)。

## 同一历史上的不同决策

全部120配对初始实际观测相同；step0执行动作已有数值差异，step1两驱动轨迹观测开始分歧。因此后续不能将各自轨迹同时间步的动作直接相减归因。本次每一条实际历史都同时查询Stage8/9，并在服务器上从私有帧重建输入、噪声及执行动作，核验54646个有效决策步。

跨检查点分量不同、夹爪异号同时存在于保持成功和失败→成功轨迹。ALIGN seed41 init12的Stage9轨迹600步没有夹爪异号仍成功→失败；同seed init5在step77出现异号却于79步成功，属于失败→成功。seed43 init7在step62出现分量及夹爪差异随后超时，仅是候选关联，不能认定step62造成失败。

成功轨迹通常更短，全程事件发生率有暴露时长偏差。补充事后固定前32步描述核对后，ALIGN成功→失败轨迹的平移动作差异也未普遍大于保持成功轨迹。没有用测试结果选择保护阈值。原始RGB关键窗口私有保留，本轮标量分析未进行图像语义判读；也没有不可逆失败时刻标签或同状态动作干预，不能声称定位了抓取/接触的具体因果。

已观察到多次有害结果变化，但尚未得到可供选择性保护使用、能够区分纠错的可重复机制。条件保护训练不启动；任何后续三组分叉须先单独固定有限协议、选择规则和Task8保持/Task9学习检查。保留重复探索限制，不作保护保证。

## 验收与成本

[最终本地审计](final_20261024_193008/LOCAL_FINAL_AUDIT.json)通过240回合、120配对、24vector及54646步。服务器最终只读复算通过历史/噪声/执行动作、源及检查点hash；本地初态记录绑定此前已接受的初态文件和逐初态字节SHA，未将本次记录绑定冒充新增初态文件独立读取。12检查点在本次GPU准入严格加载，最终hash再次复核，未声称新增CPU模型加载。

24vector合计1771.090092秒（29.518168分钟），不含准备、准入、下载及审计；13881批次前端、27762次Temporal/head前向，138810环境lane推进（含84164个已结束lane的零动作推进），另1200预热lane推进；有效策略步54646。准入208参数路径/16前端调用另列。RA离线174 CPU批次/2712模型窗口。工程01在import失败，0前向/环境，原失败保留。

下载295白名单文件共61236066字节；private图像/动作/特征/噪声/权重未下载发布。完整逐步标量light留本地，只发布其SHA清单、汇总和关键事件标量。服务器私有产物681783296字节保留。正式闭合与成本见[CLOSURE_AND_COST_AUDIT](final_20261024_193008/CLOSURE_AND_COST_AUDIT.json)；详细分层与限制见[机制分析](final_20261024_193008/MECHANISM_REPORT.md)。

本报告及轻量证据按AGENTS发布到MLR，是否push成功以本地PUBLICATION_RECEIPT及远端SHA核验为准。验收发布后删除ra-align监控，不再轮询已闭合诊断。
'''
(R/'REPORT.md').write_text(report,encoding='utf8')
(R/'MONITOR_HANDOFF.md').write_text('''# 诊断闭合，不再启动或轮询

2026-10-24：240回合/120配对/54646有效步、本地最终审计与机制分析完成；见REPORT.md与final_20261024_193008。RA43/60→46/60，ALIGN45/60→39/60。尚无可区分纠错的因果保护规则，0训练更新，不启动保护分叉。原私有证据及工程失败保留，无重复评估。完成本轮publish.py及remoteSHA/receipt核验后删除ra-align监控。不要重新deploy或恢复旧队列。
''',encoding='utf8')
with (R/'IMPLEMENTATION_NOTES.md').open('a',encoding='utf8') as f:f.write('\n## 2026-10-24 最终覆盖\n\n上述“仅AST”仅描述运行前状态：final_20261024_193008的远端私有重构及本地完整审计均已实际PASS，无包装失败或重跑；新增analyze_mechanisms.py及close_diagnostic.py已对全量标量实际运行。最终状态与限制见REPORT.md，禁止将描述性关联当保护训练准入。\n')
(R/'LOCAL_RECOMPUTATION.txt').write_text('PASS: final_readonly private recomputation; audit_final_local 240 episodes/120 pairs/54646 steps; analyze_mechanisms all trajectories; close_diagnostic accepted initial-hash binding, natural exit0 and costs. No new training or environment calls for final audits.\n',encoding='utf8')
print('Final report and handoff written; publication pending.')
