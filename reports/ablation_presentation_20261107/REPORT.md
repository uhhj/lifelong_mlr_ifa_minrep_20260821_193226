# 消融展示方案（讨论稿）

以“成功历史保持与受控候选选择如何影响持续学习”为主线。正文建议一张核心消融表、门控机制图和系数敏感性图；所有统计使用已有三seed结果，不新增实验。

## 核心表

四行：C（常规更新预算、无新增成功历史保持）、Q（保留逐任务q加权保持、关闭门控）、C matched optimizer budget（无新增保持、延长训练匹配S物理optimizer调用）、Ours（完整S）。用描述性名称搭配内部C/Q映射，不把C写成去掉全部IFA，也不把延长训练C写成普通w/o模块。

OBJECT与GOAL各列AUC、FWT、NBT，与主表一致，均值±跨seed样本SD。OldFinal、NewFinal和FinalAll放配套成功率图/补充表，完整逐任务差异保留。表注说明Q/S同接受更新及采集次数而物理optimizer成本不同；C-budget/S同物理调用而训练交互、FLOPs与墙钟不相等。

核心读数：Ours−Q的OBJECT/GOAL AUC为+1.875/+3.75pp，FinalAll为+2.0833/+4.5833pp。GOAL NBT为5.2778对4.4444，不能写全部指标改善。Ours−等预算C的OBJECT/GOAL AUC为+5.5903/+3.5417pp，说明该增加更新预算的C对照没有达到Ours的均值；不推出计算因素完全排除。Q相对常规C的OBJECT AUC略低、GOAL略高，不叙述为每个新增模块都单调提升。

## 机制图

门控图用S内部共同起点的C/Q候选探针差、接受条件及逐阶段接受率。按真实定义标坐标符号，并明确候选与独立完整C/Q训练不同。训练探针解释选择行为，正式成功率支持效果；不把通过率当作闭环收益。正文展示两个套件相同坐标的简洁面板，完整密度/条件组合和负单元移至补充。

## 敏感性

lambda_R取0.05/0.1/0.2，固定三个点，展示三seed散点与均值，避免平滑曲线制造额外采样。正文优先2×2面板：OBJECT/GOAL × AUC/FinalAll；OldFinal/NewFinal放次级面板或补充。既定0.1的两套件AUC/FinalAll均值最高，不称普适最优或不敏感。GOAL0.2提高OldFinal但降低NewFinal，说明新旧任务取舍；OBJECT0.2主要损失集中Task6，并非更大系数必定更好保持。不同GPU及Task6同模型闭环差异作为敏感性实验设置说明，不能将全部差值唯一归于系数。

## RA/ALIGN与早期设计

如果RA和ALIGN也是论文主贡献，可单列紧凑组件对照表：复现MLR+IFA、+RA、+RA+ALIGN、完整方法。不要把历史RA+ALIGN自动等同于最新C；它们有不同版本/协议。该表定位为累计配置比较，不伪装成完整开关因子实验。GOAL上历史ALIGN并未形成所有指标的单调增益，正文与完整表一致。

早期D/O/S、路由及方向锚等单seed探索放补充设计分析，限定各自协议，避免与最终三seed结果混成一条因果阶梯。当前没有独立的最终三seedw/o-q、统一缓存w/o-success-filter实验，不凭已有Q/S或不同lambda构造这些行。若未来需要这类声明，另行明确授权和设计。

来源：20261101_ablation_summary/REPORT.md；20261031_full_q_vs_latest_s_two_suites/PROTOCOL.md；20261030_equal_optimizer_budget_cs_two_suites/PROTOCOL.md；OBJECT及GOAL lambda最终报告；publication_figures_20261106/README.md。此次只读取已接受资料并提出展示方案，未新增训练、评估或更改科学数据。
