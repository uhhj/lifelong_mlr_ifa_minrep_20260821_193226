# GOAL seed41/42/43：原MLR+IFA、RA、RA+ALIGN

1800条已完成回合逐条复算，90格各20条；本轮0新训练/前向/环境。原MLR+IFA为Random+IFA复现；RA为RA-Coreset+IFA；RA+ALIGN不含C组额外专家监督。

|seed|方法|AUC|FWT|NBT↓|OldFinal|FinalAll|
|---|---|---:|---:|---:|---:|---:|
|41|MLR_IFA|69.0625|71.2500|2.7778|66.6667|66.2500|
|41|RA|78.7500|82.5000|5.5556|73.3333|76.2500|
|41|RA_ALIGN|83.4375|91.2500|16.6667|75.0000|78.7500|
|42|MLR_IFA|80.9375|82.5000|4.4444|76.6667|80.0000|
|42|RA|82.5000|85.0000|5.5556|75.0000|80.0000|
|42|RA_ALIGN|82.7083|85.0000|6.6667|71.6667|77.5000|
|43|MLR_IFA|79.0625|76.2500|-6.6667|88.3333|83.7500|
|43|RA|84.6875|83.7500|-3.3333|85.0000|86.2500|
|43|RA_ALIGN|77.1875|81.2500|9.1667|73.3333|72.5000|

三seed等权均值±样本SD（ddof=1）：

|方法|AUC|FWT|NBT↓|OldFinal|FinalAll|
|---|---:|---:|---:|---:|---:|
|MLR_IFA|76.3542 ± 6.3840|76.6667 ± 5.6366|0.1852 ± 5.9921|77.2222 ± 10.8440|76.6667 ± 9.2139|
|RA|81.9792 ± 3.0028|83.7500 ± 1.2500|2.5926 ± 5.1320|77.7778 ± 6.3099|80.8333 ± 5.0518|
|RA_ALIGN|81.1111 ± 3.4174|85.8333 ± 5.0518|10.8333 ± 5.2042|73.3333 ± 1.6667|76.2500 ± 3.3072|

|seed|方法|最终Task6|Task7|Task8|Task9|
|---|---|---:|---:|---:|---:|
|41|MLR_IFA|40.0|100.0|60.0|65.0|
|41|RA|35.0|100.0|85.0|85.0|
|41|RA_ALIGN|55.0|100.0|70.0|90.0|
|42|MLR_IFA|70.0|100.0|60.0|90.0|
|42|RA|55.0|100.0|70.0|95.0|
|42|RA_ALIGN|60.0|100.0|55.0|95.0|
|43|MLR_IFA|95.0|100.0|70.0|70.0|
|43|RA|75.0|100.0|80.0|90.0|
|43|RA_ALIGN|65.0|90.0|65.0|70.0|

RA相对原基线三seed的AUC都提高，但seed42最终四任务均值持平，seed42/43最终旧任务均值下降。ALIGN相对RA：seed41/42 AUC提高，seed43下降；最终四任务平均在seed41提高2.5pp、seed42下降2.5pp、seed43下降13.75pp，最终Task8三seed均下降15pp。不能从seed41单独的AUC提升推断跨seed保持改善。

AUC先对各任务已见阶段平均再跨任务平均；FWT沿用项目对角线平均定义；NBT是相对刚学完的后续平均下降，越低越好；OldFinal/FinalAll为最后阶段旧3任务/全部4任务平均。NBT不能替代绝对成功率。

同seed三组200条对应初态文件/逐状态SHA一致；未声称动作噪声逐轨迹完全配对。完整矩阵、负单元、源SHA和精确分数见JSON。

Historical stage-global action RNG does not establish identical action noise for trajectories across methods.
Three development seeds, 20 episodes/cell; means are descriptive, no significance claim.
Original replication state=True and schedule completion; official pretraining modality metadata unknown.
Do not combine OBJECT or seed44 or repeat experiments with this fixed GOAL41-43 comparison.
