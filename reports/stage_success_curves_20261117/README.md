# 逐阶段闭环成功率曲线

基于最新版论文主结果与组件消融口径，比较 MLR + IFA、CADL（标准训练预算）、CADL + SHBR w/o Gating 和 DEBR。两个子图横向排列，全部采用 Times New Roman，子图标题与公共图例均在图下方。SVG 中文字和曲线均可单独编辑；PNG、PDF、SVG 从同一数据与绘图对象导出。

## 统计定义

对于种子 r，在完成阶段 s 后，先计算 Task6 至 Tasks 的任务成功率算术平均：

`m_r(s) = mean_{t=6,...,s} [100 × successes(r,s,t) / 20]`。

实线/虚线点为三个种子（41、42、43）的 `mean_r m_r(s)`，阴影为该种子级指标的样本标准差（ddof=1），不是置信区间。每个任务/阶段使用正式初态编号 0–19 的 20 条闭环评估记录。每个种子每个方法共 200 条，24 个 run 共 4,800 条。没有使用训练采集编号 20–29，没有补采、重训、重评或模型推理。

纵轴为已见增量任务平均成功率，不是论文 AUC。随着阶段推进，平均所包含的任务集合扩大；曲线下降本身不能单独解释成旧任务遗忘。离散点间仅用直线连接，没有平滑、插值生成新的评估点或截断标准差带。两个子图共用 0–100% 完整纵轴。

## 论文英文图注

**Stage-wise closed-loop success rates on LIBERO-OBJECT and LIBERO-GOAL.** After each incremental training stage, success rates are averaged over the tasks encountered from Task 6 up to that stage. Curves show the mean across three training seeds (41–43), and shaded regions indicate one sample standard deviation. Each task–stage pair is evaluated on 20 episodes. CADL uses the standard training budget; CADL + SHBR w/o Gating retains support-weighted behavioral retention without candidate-update gating. All points are computed from completed evaluations.

## 中文图注

LIBERO-OBJECT 与 LIBERO-GOAL 上的逐阶段闭环成功率。每个增量训练阶段结束后，对 Task6 至当前阶段的已见任务成功率取平均。曲线表示三个训练种子（41–43）的均值，阴影表示一个样本标准差；每个任务–阶段组合评估 20 回合。CADL 使用标准训练预算，CADL + SHBR w/o Gating 保留支持度加权保持并关闭候选更新门控。所有点均由已完成的正式评估记录计算。

## 数据与验收

- 原 MLR + IFA：直接回读 60 个正式 trials 文件并核对已接受来源 SHA，与 BASELINE_BINDING / BASELINE_OUTCOMES 的 1,200 条记录逐项相等。OBJECT seed43 沿用论文已有的冻结 repeat1，不进行成绩选择或替换。
- CADL：采用 20261025 的 GOAL41 已接受 C 和 20261027 的其余五个完整 C；不是 20261030 增加更新预算后的 C。
- Q：20261101 完成并接受的六个完整无门控 run。
- DEBR：20261029 接受的 OBJECT 三种子与 20261030 接受的 GOAL 三种子，保持系数 0.1；不混入敏感性实验或反序实验。
- C/Q/S 的初态与动作噪声配对、S 结果跨比较文件一致性、各方法已有指标均已复核；AUC/FWT/NBT 的均值与最新论文表格在一位小数舍入精度下一致。
- 历史 MLR + IFA 与其他方法核验相同初态，但不声称已核验共同动作噪声。不同方法的物理更新、训练交互及历史硬件边界保持原账。这张图比较已接受方法结果，不是单机制、等算力的因果识别；开发集复用和预训练来源边界沿用原报告。

源文件和 SHA 见 SOURCE_MANIFEST.json。DATA_LAYOUT_AUDIT.json 保存实际数据与版式验收。data 中包含正式成功/失败记录、逐任务矩阵、逐种子阶段均值、阶段均值/标准差和重算指标。原证据只读，原 Word 文档未修改。

## 读图要点

OBJECT 的最终阶段均值依次为 MLR + IFA 65.42%、CADL 82.50%、无门控 SHBR 82.92%、DEBR 85.00%；GOAL 分别为 76.67%、80.00%、80.42%、85.00%。DEBR 在两套件的后两个阶段均值最高，但 Stage7 的 CADL 均值更高，图中保留全部阶段和种子变异。

## 文件与复现

- `figures/DEBR_stage_success_curves.svg`：完整可编辑矢量图，文字保留为 text。
- `figures/DEBR_stage_success_curves.png`：600 dpi，6960 × 3150 像素。
- `figures/DEBR_stage_success_curves.pdf`：矢量投稿版。
- `figures/preview.png`：便于浏览的同版预览。
- `data/stage_success_summary.csv`：各方法/套件/阶段的均值、标准差及三个种子值。

在项目根目录运行 make_figure.py，需要 numpy、matplotlib 和 Windows Times New Roman 字体。当前工作区依赖位于 `tmp/publication_plot_deps`，可通过 PYTHONPATH 指定。脚本只读正式记录，输出本目录。发布只包含显式允许的图、代码、说明、标量和成功/失败记录，不包含模型或缓冲区数据。
