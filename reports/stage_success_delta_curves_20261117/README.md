# 逐阶段闭环成功率提升 Δ 图

按用户要求，将此前的绝对成功率图转换为相对 MLR + IFA 的百分点差值。保留旧图、全部原始结果与正式统计口径，不进行新的训练、仿真或评估。

## 定义

对于每个套件、种子 r 和完成阶段 s：

`Δ_r(s) = mean_success_method,r(s) − mean_success_MLR+IFA,r(s)`。

两个 mean_success 均为该种子在阶段 s 的已见增量任务（Task6 至 Tasks）成功率均值。曲线显示三个种子 41/42/43 的差值均值；阴影表示三个种子差值的样本标准差（ddof=1）。标准差先做逐种子相减再计算，不能直接用两个标准差相减或沿用绝对图的标准差。这里的相同种子索引不代表历史基线与新方法已经核验共同动作随机流。

纵轴单位为百分点（pp），不是相对增长百分比。零虚线表示 MLR + IFA：+10 pp 表示比该基线高 10 个百分点，负值表示低于基线。所有方法减去同一基线，因此任意两方法之间的均值差距保持不变。相对坐标减少绝对数值的背景高度，使差异更容易阅读，不改变实验效果大小。

OBJECT 与 GOAL 共用 −15 至 +35 pp 的纵轴，完整包含所有均值±标准差及零基线；未截断负值、删除不利阶段或平滑曲线。CADL 仍为论文的标准预算版本，未换成增加优化器调用预算后的 C。

## English caption

**Stage-wise closed-loop success-rate gains over MLR + IFA.** At each completed training stage, success rates are averaged over the tasks encountered from Task 6 up to that stage. Differences from MLR + IFA are computed separately for each corresponding training seed. Curves show the mean difference across seeds 41–43, and shaded regions indicate one sample standard deviation of the seed-wise differences. The dashed horizontal line denotes the MLR + IFA reference. Values are percentage-point differences, and each task–stage pair uses 20 evaluation episodes. CADL uses the standard training budget.

## 中文图注

相对 MLR + IFA 的逐阶段闭环成功率提升。在每个训练阶段结束后，对 Task6 至当前阶段的已见任务成功率取平均，并分别计算每个对应种子相对 MLR + IFA 的差值。曲线为种子 41–43 的差值均值，阴影为种子级差值的一个样本标准差；水平虚线表示 MLR + IFA 参考水平。单位为百分点，每个任务–阶段组合评估 20 回合。CADL 使用标准训练预算。

## 来源与文件

从前一版已审查、已发布的 4,800 条正式成功/失败记录重新计算全部阶段均值，再计算 72 个逐种子差值和 24 个差值均值。核验原始验收源 SHA、已发布标量文件 SHA，以及与绝对图的均值关系。未用训练损失、训练采集或聚合 AUC 反推曲线。历史基线及硬件、算力、数据复用边界沿用绝对图 README，不新增因果或统计显著性结论。

- figures/DEBR_stage_success_delta.svg：可编辑文字、曲线与阴影；无栅格嵌图。
- figures/DEBR_stage_success_delta.png：600 dpi，6960 × 3150。
- figures/DEBR_stage_success_delta.pdf：矢量版。
- figures/preview.png：同版预览。
- data/delta_per_seed.csv、data/delta_summary.csv：逐种子差值和均值/样本标准差。
- DATA_LAYOUT_AUDIT.json、SOURCE_MANIFEST.json：实际数据与版式验收、来源 SHA。

全部文字为 Times New Roman，横向两个子图，各自标题和公共图例放在下方。SVG/PNG/PDF 使用同一绘图对象导出，已经目视核对预览。使用配置了 numpy/matplotlib 的 Python 执行 make_figure.py；当前依赖可通过 PYTHONPATH 指向项目 tmp/publication_plot_deps。
