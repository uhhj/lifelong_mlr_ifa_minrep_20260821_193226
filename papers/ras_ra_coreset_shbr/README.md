# RA-Coreset + SHBR：RAS论文框架

主入口：main.tex。英文可编辑初稿，包含两模块动机、方法公式、真实结果表、参考文献和投稿声明位置；蓝色 Draft task 为待补内容。

## 模板来源

Elsevier官方LaTeX页面：https://www.elsevier.com/researcher/author/policies-and-guidelines/latex-instructions

官方模板压缩包保存在 official_template，来源与SHA见 TEMPLATE_PROVENANCE.json。本文使用官方 elsarticle 类和数字引用样式；review、12pt 是工作稿选择。RAS作者指南本次访问返回403，未将版式、页数或匿名要求声明为期刊强制规则，正式投稿前须复核期刊指南。

官方压缩包包含3.4版模板源。单独可下载的官方 elsarticle.cls 为3.3版，原样保留；没有手工改写期刊类。elsarticle-num.bst来自官方压缩包。

## 编译

Overleaf：上传交付ZIP的全部平铺文件，主文件选择main.tex，编译器pdfLaTeX。

本机：运行 build.ps1，优先使用已下载的便携Tectonic；常规TeX安装可运行 latexmk -pdf main.tex。没有安装latexmk时依次运行 pdflatex main、bibtex main、pdflatex main两遍。

prepare_tables.py 从工作区最终接受的轻量JSON生成表格与 results_snapshot.json。在独立下载的工程中直接使用已生成表格即可，不需要访问实验服务器。不要把真实检查点、缓存或凭据放入论文工程。

## 文件组织

- sec01–sec07：引言、相关工作、问题定义、方法、实验、讨论、结论。
- table_main、table_controls、table_sensitivity：可编辑LaTeX表格，数值来自接受报告。
- references.bib：已核实的MLR+IFA、M2Distill和LIBERO文献条目。
- declarations：作者贡献、利益冲突、基金、数据可用性与AI使用声明的待填位置。
- highlights.txt：简短研究亮点草稿，是否作为独立文件提交以期刊系统为准。
- WRITING_PLAN_ZH.md：下一步写作与图表安排。

本工程是论文框架，不是可直接投稿的终稿。没有虚构作者、实机实验、外部基线结果或参考文献。图中明确标为占位的内容须替换为真实素材。
