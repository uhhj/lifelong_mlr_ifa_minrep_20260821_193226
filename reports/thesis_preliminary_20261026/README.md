# 论文前期工作：实证素材

本材料按用户提供的参考页面组织，正文聚焦已取得成果，方法、套件、种子与指标口径均明确标注。

## 概览页简化修订

`output/前期工作_概览简化版.pptx`保留六页材料，仅简化第一页五栏概览。每栏改为两行作用说明，正文由23增至26px，移除任务编号、损失系数、mask、分量路由等实现术语。第一页使用TASA、SHBR新名称，对应历史ALIGN、S，后续证据页和原记录保留旧标识便于追溯。首版PPTX和素材包保留。修改后的第一页已渲染检查文字、换行与边界；不涉及任何实验或数据变化。

## 可直接使用的材料

- `output/前期工作_MLR-IFA_RA_ALIGN_S_实证版.pptx`：6页可编辑汇报，包含方法流程、指标表、真实双相机图和训练曲线。前4页为方法与结果，第5页为真实rollout画面，第6页为训练记录。
- `latex_document/main.tex`：自包含LaTeX项目入口。使用XeLaTeX与`ctexart`，所需表格与图在同目录。`table_baselines.tex`与`table_S41.tex`可独立复制进论文，需加载`booktabs`。`metric_definitions.tex`需加载`amsmath`。
- `training_curves_S41.png`和`.pdf`：实际400轮训练数据生成，未平滑，PDF适于论文排版。BC曲线是连续高斯分布的NLL，数值可为负。
- `train_S41_original_excerpt.jsonl`：S41 Task9 epoch97、98、99的三条原始日志行，未改写数值。
- `EPOCHS_S41.json`：最终验收证据中的完整400轮原始标量记录。
- `training_scalars_S41.csv`：便于绘图的训练标量。批均值由记录总量除以对应batch数计算。
- `all_metrics.csv`：GOAL与OBJECT各三方法、三种子的完整已完成指标，以及S41结果。报告正文选用与成果主题相关的行，不更改完整数据。

## 真实图片说明

`assets_private/`的6张无损PNG来自已保存并验收的ALIGN诊断rollout，GOAL Task8“将碗放到盘子上”，seed41、Stage9、初态3，最终成功，第0、65、77步各有外部相机和腕部相机。

这是此前已审阅记录中最小成功初态编号，用于展示真实任务和执行画面。图片为原始128×128 RGB，未重绘或增强。时间点是不连续的关键帧，来源不是S。逐帧信息、源NPZ哈希与导出PNG哈希见`REAL_FRAME_PROVENANCE.json`。原NPZ及动作、状态、模型数据均未下载。

## 数值与使用口径

- RA、ALIGN表格采用seed41/42/43均值与样本标准差，每阶段--任务单元20回合。原MLR+IFA为本项目复现成绩。
- S表格只采用已最终验收的GOAL seed41，200正式回合和额外40次训练采集分开记账。
- AUC、OldFinal、FinalAll等指标由完整成功率矩阵独立复算，见`EVIDENCE_AUDIT.json`。
- S42/43本次不纳入结果，引用的最近进度仍为2026-10-26 14:55:25的PARTIAL记录。

## 文件检查与发布

PPTX按6页和可编辑表格要求检查，图片/曲线逐页检查。LaTeX源码完成结构与数值检查；内置编译调用未取得完成返回，不宣称已编译出整篇PDF。

本次只提取既有图像与标量，新增训练、模型前向、环境回合均为0。Git仅发布代码、表格源码、报告、标量与哈希，真实RGB、PPTX及素材包只用于本地论文交付。
