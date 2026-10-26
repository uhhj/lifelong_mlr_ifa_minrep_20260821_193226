# OBJECT seed41/42/43：原MLR+IFA、RA、RA+ALIGN

主表1800条已完成回合及seed43首次敏感性200条逐条复算；本轮0新训练/前向/环境。原MLR+IFA为Random+IFA复现；RA为RA-Coreset+IFA；RA+ALIGN不含C组额外专家监督。

|seed|方法|AUC|FWT|NBT↓|OldFinal|FinalAll|
|---|---|---:|---:|---:|---:|---:|
|41|MLR_IFA|66.8750|63.7500|-4.1667|58.3333|61.2500|
|41|RA|73.0208|75.0000|4.1667|78.3333|70.0000|
|41|RA_ALIGN|81.4583|80.0000|-1.6667|70.0000|75.0000|
|42|MLR_IFA|66.1458|66.2500|-0.5556|70.0000|63.7500|
|42|RA|71.5625|70.0000|-3.0556|65.0000|63.7500|
|42|RA_ALIGN|83.0208|81.2500|-3.3333|76.6667|80.0000|
|43|MLR_IFA|71.8750|70.0000|-3.6111|75.0000|71.2500|
|43|RA|75.1042|75.0000|-1.3889|81.6667|75.0000|
|43|RA_ALIGN|76.0417|71.2500|-9.7222|76.6667|72.5000|

三seed等权均值±样本SD（ddof=1）：

|方法|AUC|FWT|NBT↓|OldFinal|FinalAll|
|---|---:|---:|---:|---:|---:|
|MLR_IFA|68.2986 ± 3.1186|66.6667 ± 3.1458|-2.7778 ± 1.9444|67.7778 ± 8.5527|65.4167 ± 5.2042|
|RA|73.2292 ± 1.7800|73.3333 ± 2.8868|-0.0926 ± 3.7816|75.0000 ± 8.8192|69.5833 ± 5.6366|
|RA_ALIGN|80.1736 ± 3.6627|77.5000 ± 5.4486|-4.9074 ± 4.2522|74.4444 ± 3.8490|75.8333 ± 3.8188|

|seed|方法|最终Task6|Task7|Task8|Task9|
|---|---|---:|---:|---:|---:|
|41|MLR_IFA|15.0|85.0|75.0|70.0|
|41|RA|45.0|95.0|95.0|45.0|
|41|RA_ALIGN|30.0|85.0|95.0|90.0|
|42|MLR_IFA|10.0|100.0|100.0|45.0|
|42|RA|10.0|90.0|95.0|60.0|
|42|RA_ALIGN|40.0|100.0|90.0|90.0|
|43|MLR_IFA|35.0|100.0|90.0|60.0|
|43|RA|50.0|95.0|100.0|55.0|
|43|RA_ALIGN|35.0|95.0|100.0|60.0|

主表seed43原MLR+IFA采用冻结RA协议指定的repeat1；首次另作敏感性，不按结果选择。ALIGN相对RA三个seed AUC均提高，但最终旧任务平均在seed41/43下降，seed43最终四任务平均也下降。

AUC先对各任务已见阶段平均再跨任务平均；FWT沿用项目对角线平均定义；NBT是相对刚学完的后续平均下降，越低越好；OldFinal/FinalAll为最后阶段旧3任务/全部4任务平均。NBT不能替代绝对成功率。

同seed三组200条对应初态文件/逐状态SHA一致；未声称动作噪声逐轨迹完全配对。完整矩阵、负单元、源SHA和精确分数见JSON。

Historical stage-global action RNG does not establish identical action noise for trajectories across methods.
Three development seeds, 20 episodes/cell; means are descriptive, no significance claim.
Original replication state=True and schedule completion; official pretraining modality metadata unknown.
Main original baseline uses seed43 repeat1 per frozen RA protocol; first seed43 is separate sensitivity, not a fourth seed.

seed43首次原MLR+IFA敏感性：{"first_seed43": {"AUC": 78.75, "FWT": 82.5, "NBT": 6.666666666666667, "OldFinal": 80.0, "FinalAll": 80.0}, "original_mean_using_first": {"AUC": 70.59027777777777, "FWT": 70.83333333333333, "NBT": 0.6481481481481481, "OldFinal": 69.44444444444444, "FinalAll": 68.33333333333333}}

未更改任何实验合同或重新训练/评估。原MLR+IFA是项目复现成绩，不是论文表格原值。
