# 最终状态：TAC、ALIGN完整验收；EFTR用户提前终止

本地验收时间：2026-10-22T22:06:45.551202+08:00。ALIGN服务器自然完成时间：2026-10-22T21:55:43.238796+08:00。当前授权范围已完成，无运行中或排队训练；不声称三组全部验收。

ALIGN最终包 `final_ALIGN_seed41_20261022_220252`：946轻量文件全部SHA通过，400epoch/105500主步、4个ep99严格100键、200正式记录/20vector/10cell、0技术异常；原RA成员载荷、IFA、初始化、合法mask及完整账目独立PASS。无辅助头、效果或候选更新。续接controller/supervisor/pipeline退出0、无残留，独立receipt与合同绑定PASS；原725+续接5冻结hash不变。EFTR的11个pth原地SHA仍匹配，旧exit1及失败证据保留，不恢复、不补评。

|指标（%）|RA|ALIGN|TAC|TAC−ALIGN pp|
|---|---:|---:|---:|---:|
|AUC|73.020833|81.458333|74.895833|-6.562500|
|FWT|75|80|72.5|-7.5|
|NBT（有符号，低好）|4.166667|-1.666667|-4.166667|-2.5|
|OldFinal|78.333333|70|61.666667|-8.333333|
|FinalAll|70|75|66.25|-8.75|

ALIGN最终Task6/7/8/9成功率30/85/95/90%；TAC为15/70/100/80%。ALIGN比RA的AUC高8.4375pp、FinalAll高5pp，但OldFinal低8.333333pp；不能称旧任务保持全面改善。TAC相对ALIGN的AUC、FWT、OldFinal、FinalAll均更低；NBT更低不抵消最终旧任务成功率下降。仅本seed，未证实TAC辅助效果回归与跨任务对比联合模块的总体收益，不分拆归因、不作显著结论。

完整原episode重算、精确分数矩阵与归因对照：`tac_align_final_comparison_20261022/REPORT.md`及`COMPARISON.json`。TAC/ALIGN均使用相同下一动作BC与mask；因此对比不混入八步输入或BC时间目标差异。两者额外计算不同：ALIGN主epoch47.791分钟、正式vector73.112分钟；TAC主epoch64.739分钟、正式vector75.968分钟，另有105493头更新及184900伙伴前向。准备/存储/验收另计。

EFTR保持EARLY_TERMINATED_BY_USER，246完整epoch/52614主步/0正式，检查点、日志与零训练诊断保留，不作完整负结果比较。既往取消任务不恢复，未解决科学限制保留。本轮发布核验成功后删除同一监控。
