# 原MLR+IFA基线对比：GOAL seed41

本地复算已验收200条原Random+IFA回合；0新增训练、模型前向或环境。原MLR+IFA指项目按论文描述对齐的Random+IFA复现，不是论文原表成绩；未加Coreset、RA、ALIGN或CCBP辅助项。

|方法|AUC|ΔAUC pp|FWT|NBT↓|OldFinal|FinalAll|ΔFinalAll pp|
|---|---:|---:|---:|---:|---:|---:|---:|
|MLR_IFA|69.0625|0.0000|71.2500|2.7778|66.6667|66.2500|0.0000|
|RA_ALIGN|83.4375|14.3750|91.2500|16.6667|75.0000|78.7500|12.5000|
|C|87.5000|18.4375|92.5000|7.7778|80.0000|82.5000|16.2500|
|D|78.3333|9.2708|86.2500|16.3889|71.6667|72.5000|6.2500|
|O|87.5000|18.4375|92.5000|9.4444|76.6667|80.0000|13.7500|
|S|86.8750|17.8125|91.2500|5.5556|81.6667|85.0000|18.7500|

|方法|最终Task6|Task7|Task8|Task9|
|---|---:|---:|---:|---:|
|MLR_IFA|40.0|100.0|60.0|65.0|
|RA_ALIGN|55.0|100.0|70.0|90.0|
|C|40.0|100.0|100.0|90.0|
|D|55.0|100.0|60.0|75.0|
|O|45.0|100.0|85.0|90.0|
|S|45.0|100.0|100.0|95.0|

完整三角矩阵与所有低于基线的单元见JSON。AUC按每任务已见阶段平均再跨任务平均；NBT按本项目定义越低越好，不能把更高最终成功率等同于更低遗忘。

原基线和DOS的200/600条对应初态文件及逐状态SHA一致；原基线每stage重置全局动作RNG，DOS按轨迹独立噪声，因此不是完全配对的同随机流消融。各组主训练均400epoch/76100步，但DOS额外采集与保持计算另计。

相对原MLR+IFA的总提升包含RA选样、ALIGN监督、C专家/路由辅助项以及D/O/S保持等多项变化，不能全部归因于新保持模块。仍保留原C强对照及未过整体门的结论。单seed开发数据不作显著性或独立确认声称。

Historical baseline uses stage-global action RNG; DOS use per-trajectory paired Normal noise. Same initial hashes are not same action-noise streams.
DOS include RA, ALIGN, C objectives and extra training interaction; total gain cannot be attributed only to O/S retention or success filtering.
Original C comparator and all preregistered stopping decisions remain unchanged.
Single development seed with 20 episodes/cell; no significance or independent-confirmation claim.
Original reproduction uses state=True and completed schedule; official pretraining modality metadata remains unknown.
