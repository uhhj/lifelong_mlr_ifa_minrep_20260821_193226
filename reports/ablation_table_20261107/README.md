# 五配置消融表

- `ablation.tex`：独立可编译的LaTeX文件，不依赖外部bib或图片。
- `ablation.pdf`：实际编译的单页PDF。
- `table_ablation.tex`：供论文使用的table*片段。
- `ablation_preview.png`：高清表格预览。
- `per_seed_metrics.csv`、`table_values.csv`：逐seed结果及完整精度均值/样本标准差。

表中仅使用正式配置名称：RA-Coreset、RA-Coreset + ALIGN、+ Execution Supervision、+ Support-Weighted Retention、Ours。第三、四行的加号表示在上一行基础上累积增加机制；Ours再加入候选门控。所有配置以原MLR+IFA为底座。执行专家监督包括执行代价和margin路由；保持包括成功历史、固定教师、逐任务支持度加权及保持路由。内部结果源对应RA、RA_ALIGN、普通预算C、完整Q、最新S，内部代号仅在审计数据中保留，不出现在表格。

OBJECT、GOAL各三个seed41/42/43，列顺序为FWT↑、NBT↓、AUC↑，均值±样本标准差（ddof=1），与主表相同保留一位小数。每列以未舍入均值判断最优加粗；Ours浅蓝底。无引用编号、无LIBERO-50、无评估随机流差异表注。实现及协议关系保留于 `../20261107_align_cqs_audit/REPORT.md`，解释按累计配置比较，不改变原实验记录。

## 编译

```bash
pdflatex -interaction=nonstopmode -halt-on-error ablation.tex
```

也可用Tectonic或XeLaTeX。本包用Tectonic 0.17.0实际编译并渲染核验。

## 插入论文

使用 `\input{table_ablation.tex}`，在论文导言区加入：

```latex
\usepackage{amsmath,booktabs,multirow,graphicx}
\usepackage[table]{xcolor}
\definecolor{oursblue}{RGB}{211,235,241}
\newcommand{\result}[2]{\ensuremath{#1\!\pm\!#2}}
\newcommand{\bestresult}[2]{\ensuremath{\mathbf{#1\!\pm\!#2}}}
```

若已插入主表并定义这些宏，无需重复定义。表格采用与主表一致的Times风格、三线表、双套件表头、尺寸自适应和Ours行强调。数据由已接受矩阵复算，没有新训练或环境回合。
