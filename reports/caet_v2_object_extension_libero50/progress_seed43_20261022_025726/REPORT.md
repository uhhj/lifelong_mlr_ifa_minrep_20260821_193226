# CAET-v2 OBJECT43训练进度

状态PARTIAL，观察时间2026-10-22T02:57:47.419581+08:00。

最新2026-10-22 02:58 CST：progress_seed43_20261022_025726，43正常Task8，230epoch47670主步/0正式；Task6实际tensor同RA、505冻结source/input、已完train6/7 retain6/7等exit0与本地逐步审计PASS。Task7全RA回退；Task8预测器留出PASS，MAE.253718对均值.488547/置换.277847，75支持时刻12旧task-demo，10epoch840额外更新。Task8已30epoch，迁移2486步、冲突关闭1594、保护非零4；不能把预测/激活当闭环收益。Task9尚未准入。最近Task8十epoch均13.78秒，剩主训练约47.7分钟（Task9用41实测18.99秒/epoch参考），评估参考79.8分钟另计拟合入库验收，43预计05:10–05:50仅参考。正常无恢复，不重复41/42最终审计。容量watcher仍ARMED，data普通8.205GB/root13.075GB、system普通5.902GB，LIBERO50未GPU准入。package_final.py新增--manifest-only，仅生成白名单manifest后用resume_light_download.py <目录>并行只读下载、全SHA再audit_final/summarize_final，避免下一次顺序SFTP中断；新开关仅语法检查，下载器和最终审计已41实证。未修改服务器源或实验预算。
