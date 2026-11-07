# 完整方法对比主表（OBJECT / GOAL）

- `main_comparison.tex`：自包含文档，包含主表、所需宏包和参考文献，无需额外 `.bib` 或 BibTeX。
- `main_comparison.pdf`：实际编译预览；第一页为主表，第二页为参考文献。
- `table_comparison.tex`：用于论文 `\input{table_comparison.tex}` 的表格片段。
- `main_comparison_preview.png`：主表高清预览。
- `table_values.csv`、`ours_per_seed.csv`：完整表格数值和 Ours 逐seed指标。
- `TABLE_AUDIT.json`：原表提取、Ours 复算和最佳均值标记记录。

仅包含 LIBERO-OBJECT / LIBERO-GOAL，各按 FWT↑、NBT↓、AUC↑ 排列。所有方法统一排版；不在主表中设置来源分组或来源标签。Ours 为既定 λ_R=0.1 的最新完整 S，使用seed41/42/43计算均值与样本标准差，保留1位小数。MLR与MLR+IFA不再带原论文的“Ours”后缀。按每列最佳均值重新加粗，不直接照搬截图中的粗体。Ours 行浅蓝色强调。

方法旁保留正常文献引用，编号由当前文档自动生成。基线数值从所附 Yu 等论文 Table 1 直接提取，未重跑；Ours从已验收配对outcomes重新计算，并与上一轮绘图CSV逐项核对。来源与协议记录保留在审计文件中，不把排版统一解释为所有实验均由本项目重跑。

## 编译

```bash
pdflatex -interaction=nonstopmode -halt-on-error main_comparison.tex
pdflatex -interaction=nonstopmode -halt-on-error main_comparison.tex
```

也可用 XeLaTeX / Tectonic。自包含文件使用常规LaTeX宏包，无系统字体依赖、无图片依赖、无shell-escape。

## 放入论文

使用 `table_comparison.tex` 时，将以下设置并入论文导言区，并在论文自己的参考文献中加入表格的引用key（完整条目可从自包含文件复制）：

```latex
\usepackage{amsmath,booktabs,multirow,graphicx}
\usepackage[table]{xcolor}
\definecolor{oursblue}{RGB}{211,235,241}
\newcommand{\result}[2]{\ensuremath{#1\!\pm\!#2}}
\newcommand{\bestresult}[2]{\ensuremath{\mathbf{#1\!\pm\!#2}}}
```

片段使用 `table*`，适合双栏论文通栏；单栏文档也可使用。若不需要底色，删除Ours前的 `\rowcolor{oursblue}` 即可。表格内容与数据脚本独立保存，不修改已有实验接受记录。
