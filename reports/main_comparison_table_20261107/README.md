# 完整方法对比主表（OBJECT / GOAL）

- `main_comparison.tex`：自包含文档，包含主表和所需宏包，无需额外 `.bib` 或 BibTeX。
- `main_comparison.pdf`：实际编译预览；单页主表。
- `table_comparison.tex`：用于论文 `\input{table_comparison.tex}` 的表格片段。
- `main_comparison_preview.png`：主表高清预览。
- `table_values.csv`、`ours_per_seed.csv`：完整表格数值和 Ours 逐seed指标；`mlr_ifa_per_seed.csv` 保存MLR+IFA复现逐seed指标。
- `TABLE_AUDIT.json`：原表提取、Ours 复算和最佳均值标记记录。

仅包含 LIBERO-OBJECT / LIBERO-GOAL，各按 FWT↑、NBT↓、AUC↑ 排列。所有方法统一排版；不在主表中设置来源分组或来源标签。Ours 为既定 λ_R=0.1 的最新完整 S，使用seed41/42/43计算均值与样本标准差，保留1位小数。删除MLR行；MLR+IFA使用本项目seed41/42/43复现指标。按每列最佳均值重新加粗，不直接照搬截图中的粗体。Ours 行浅蓝色强调。

表格含9个方法，不显示引用编号、不附参考文献页。七个外部方法数值来自所附Yu等论文Table 1；MLR+IFA从 `20261101_ablation_summary/DATA.json` 的baseline/MLR_IFA六个任务矩阵重新计算；Ours从已验收配对outcomes重新计算。两种本项目方法均采用跨seed样本标准差（ddof=1）。来源SHA与复算记录保留在TABLE_AUDIT.json中。

MLR+IFA（FWT、NBT、AUC）：OBJECT为66.7±3.1、−2.8±1.9、68.3±3.1；GOAL为76.7±5.6、0.2±6.0、76.4±6.4。GOAL的NBT最优均值属于MLR+IFA，其余五列最优均值属于Ours。

## 编译

```bash
pdflatex -interaction=nonstopmode -halt-on-error main_comparison.tex
pdflatex -interaction=nonstopmode -halt-on-error main_comparison.tex
```

也可用 XeLaTeX / Tectonic。自包含文件使用常规LaTeX宏包，无系统字体依赖、无图片依赖、无shell-escape。

## 放入论文

使用 `table_comparison.tex` 时，将以下设置并入论文导言区：

```latex
\usepackage{amsmath,booktabs,multirow,graphicx}
\usepackage[table]{xcolor}
\definecolor{oursblue}{RGB}{211,235,241}
\newcommand{\result}[2]{\ensuremath{#1\!\pm\!#2}}
\newcommand{\bestresult}[2]{\ensuremath{\mathbf{#1\!\pm\!#2}}}
```

片段使用 `table*`，适合双栏论文通栏；单栏文档也可使用。若不需要底色，删除Ours前的 `\rowcolor{oursblue}` 即可。表格内容与数据脚本独立保存，不修改已有实验接受记录。
