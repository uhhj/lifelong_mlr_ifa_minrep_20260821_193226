# CAET-v3-TAC OBJECT seed41 final acceptance

Independent exact full episode matrix and mechanism ledger PASS. Single-seed exploratory test.

|Metric|RA %|Original v2 %|New %|New-RA pp|
|---|---:|---:|---:|---:|
|AUC|73.020833|65.000000|74.895833|+1.875000|
|FWT|75.000000|67.500000|72.500000|-2.500000|
|NBT|4.166667|5.833333|-4.166667|-8.333333|
|OldFinal|78.333333|73.333333|61.666667|-16.666667|
|FinalAll|70.000000|58.750000|66.250000|-3.750000|

FWT is trained-diagonal performance; NBT lower is better. TAC/ALIGN change BC targets and valid exposure. EFTR retains original BC. No closed-loop guarantee or automatic seed expansion. All extra computation counted.

## 原始成功率矩阵（%）

|阶段|任务|RA|旧v2|本组|本组−RA pp|
|---|---|---:|---:|---:|---:|
|6|6|60.000000|60.000000|35.000000|-25.000000|
|7|6|85.000000|85.000000|90.000000|5.000000|
|7|7|95.000000|95.000000|75.000000|-20.000000|
|8|6|35.000000|65.000000|45.000000|10.000000|
|8|7|90.000000|75.000000|75.000000|-15.000000|
|8|8|100.000000|100.000000|100.000000|0.000000|
|9|6|45.000000|30.000000|15.000000|-30.000000|
|9|7|95.000000|100.000000|70.000000|-25.000000|
|9|8|95.000000|90.000000|100.000000|5.000000|
|9|9|45.000000|15.000000|80.000000|35.000000|

## 成本与解释边界

主epoch累计 64.739 分钟，正式vector累计 75.968 分钟；额外准备、存储、加载和验收另计。辅助/候选计算详见 QUEUE_TASK_COST_AUDIT.json，不称等算力比较。
TAC的跨任务锚点曝光为重复训练次数，不是独立样本数。TAC/Align改变BC目标及有效监督量；TAC相对RA的差异不能在Align完成前归因于效果模块。单seed探索性，不宣称统计显著。NBT按有符号数值越小越好，须同时看OldFinal与各任务绝对成功率。
