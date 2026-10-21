# CAET-v2 OBJECT43训练进度

状态PARTIAL，观察时间2026-10-22T03:27:45.784985+08:00。

最新2026-10-22 03:28 CST：progress_seed43_20261022_032726，43正常Task9，330epoch80160主步/0正式；Task6实际tensor同RA、505冻结source/input、已完阶段exit0及本地逐步审计PASS。Task8完整100epoch，迁移7612/冲突5964/保护2578步；Task9前30epoch迁移2945/冲突1818/保护非零0步，只是当前读数，不能提前说全阶段无保护。Task8/9预测器留出均PASS；9的MAE.255011对均值.468748/置换.286582，64支持时刻18旧task-demo，980额外更新，两预测器合计1820。Task9最近十epoch15.50秒，余70epoch约18.1分钟；完整评估参考79.8分钟加拟合入库验收，43预计05:10–05:50参考。正常无恢复/无重复预检；data普通8.048GB/root12.918GB、system普通5.902GB，容量watcher仍ARMED，L50未GPU准入。下一轮应进正式评估，先真实进程与attempt核对，不把训练结束当COMPLETED。最终用package_final.py 43 --manifest-only后resume_light_download.py <包目录>，全SHA后audit_final.py <包> 43和summarize_final.py <包>；然后三seed完整均值SD，勿提前使用部分指标。
