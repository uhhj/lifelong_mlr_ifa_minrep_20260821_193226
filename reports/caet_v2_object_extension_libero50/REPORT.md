# CAET-v2 OBJECT41/43与RA LIBERO50启动记录

状态：PARTIAL。2026-10-21 23:35 CST观察。服务器controller PID71536，独立五入口有限队列已启动，不调用已取消的COVR总调度。

|实验|当前状态|
|---|---|
|CAET-v2+RA+IFA OBJECT41|第一关通过、8次隔离GPU更新通过；Task6已9完整epoch/1413正式训练更新，0正式闭环|
|CAET-v2+RA+IFA OBJECT43|第一关通过；顺序等待41完成，尚未GPU准入|
|RA+IFA LIBERO50 41|排队；先执行共享seed42复现预训练，再导出/训练/评估41|
|RA+IFA LIBERO50 42、43|依次排队；复用同一新共享预训练，各自新特征和RA记忆|

两OBJECT入口219源码逐文件与已验收seed42完全相同；仅原有seed/路径/父实验绑定不同。41首轮507项冻结源码及输入hash通过、真实GPU恢复与无效化严格一致、历史退化响应和塑性门通过。Task6最终参数同父的比较需本阶段100epoch完成后执行，当前不提前称通过。两组各400epoch/105500主更新/4ep99/200正式20vector，已有RA和CAET-v1基线只读。后续性能不改变固定两seed预算。

|seed|Task8合格当前时刻/留出支持|Task9合格当前时刻/留出支持|
|---|---|---|
|41|178 / 74|206 / 66|
|43|174 / 75|204 / 64|

两seed Task7均零支持，按原方法整阶段回退RA；Task8各6合格组、Task9各7组。真实float32旧sidecar支持图与离线图完全相同。这仅证明可行性，效果预测器的留出验收和训练期实际迁移/保护仍待各阶段执行。

seed42实测主训练4668秒、完整评估4387秒，另含标签、参考评分与验收。按该参考，41预计10月22日02:10–03:00、43随后约05:00–06:30，均非承诺；用各阶段实测及时更新。LIBERO50无实际GPU吞吐，不虚构总完成时间。

LIBERO50 50文件已审计，沿用既定共享复现预训练1941000更新及三seed各859400更新，总4519200更新，新增4500正式轨迹/450vector。共享权重明确reproduced，非官方50权重，不借用OBJECT/GOAL权重。尚未通过GPU准入。三seed新特征各45388窗口，单seed原文件预计约2.82GiB；buffer同卷硬链接不翻倍。已把seed41新输出安排系统盘，其他新输出数据盘，每入口容量保护；总余量紧张，完整累计存储尚未验收。后续以实际产物大小处理新资产存储，不删除旧资产、不降低精度、batch或更新数。详见IMPLEMENTATION_NOTES。

CAET-v2是看到seed42结果后的探索性扩展，不能将预先固定阈值表述为全新独立验证。主BC与因果辅助有一拍目标差异；global轨迹噪声未独立隔离；历史NLL保护不等同闭环保证。COVR继续取消、CAET-v1 GOAL科学阻塞保留，本队列不包含GOAL。

新package_final/audit_final已参数化41/43，仅静态检查，尚未做最终验收。所有最终指标须200正式记录及严格加载/退出证据齐全再独立复算，发布轻量结果。


## 2026-10-22 00:54更新

CAET-v2 OBJECT41已378epoch/97536主步，Task8/9预测器留出均PASS，Task8真实保护已激活；Task9第50epoch首次旧组超限并上调mu。正式仍0。507冻结输入源码与29调度文件核验PASS，无重启。见[本轮报告](progress_seed41_20261022_004440/REPORT.md)。Task8拟合墙钟耗时为负的原字段保留并标不可用。新增容量后备watcher72538等待原队列明确容量退出才接管L50待执行42/43，用/data root可写空闲并保留2GiB，不改冻结源或文件系统设置，不删除任何文件。预计41 02:15–02:50完成供参考。


## 2026-10-22 01:31更新

最新2026-10-22 01:31 CST：progress_seed41_20261022_013003完成400epoch105500主步，train6–9/retain6–9全部exit0，507冻结hash与Task6实际tensor同RA、本地逐epoch/逐步账目PASS。Task8/9迁移实际7602/9103步，冲突关闭6224/6998，保护非零2915/4186步；两预测器留出PASS，拟合额外1840更新。Task7全RA回退。正式60/200条、6完成vector+1进行中、0技术异常；evaluate PID72624及10环境正常。新增外置audit_evaluation_progress.py <目录> <seed>，逐episode初态hash/index/envseed/600步及attempt唯一性/成功向量独立PASS，cell summary留最终包。6批均249.01秒，剩余约58分钟加最终审计，预计02:30–03:00 CST供参考。仍PARTIAL，无部分AUC或性能结论。未重启、未改科学源、未重复预检；43和LIBERO50仍排队未GPU准入。容量watcher仍ARMED未接管，data普通空闲9.205GB/root全空闲14.075GB，system普通5.902GB；无需对活动payload操作。下一轮继续核41正式评估，完成后package_final/audit_final 41并确认43自动准入。


## 2026-10-22 01:58更新

最新2026-10-22 01:58 CST：progress_seed41_20261022_015731，seed41正式130/200条，13完成vector+1进行中，0技术异常，evaluate72624及10环境正常；400epoch105500主步、Task8/9迁移7602/9103与保护2915/4186不变。507冻结hash及逐epoch/逐步/逐episode身份与attempt成功向量独立PASS。13批均243.30秒，剩余评估约28.4分钟加最终验收，预计02:30–02:55参考。正常无重启，无部分AUC。43与LIBERO50仍排队未GPU准入，容量watcher仍ARMED未接管，data普通9.205GB/root全部14.075GB、system普通5.902GB。下一轮若完成先package_final.py 41和audit_final.py <包> 41真实独立验收，并核43自动启动；若仍评估则继续原进程，禁止盲重启。


## 2026-10-22 02:32更新

最新2026-10-22 02:32 CST：seed41已完整独立验收final_seed41_20261022_022751/LOCAL_FINAL_AUDIT.json及TASK_COST_AUDIT/QUEUE_EXIT_LOCAL_AUDIT PASS。400epoch105500主步、4CP严格100键、200正式20vector10cell、原RA成员载荷/IFA/初始化与Task6tensor、全部阶段exit0无残留；950轻量文件全SHA通过。AUC65.0 vs RA73.020833/v1 71.25；FWT67.5 vs75/73.75；NBT5.833333 vs4.166667/3.888889（低好）；OldFinal73.333333 vs78.333333/66.666667；FinalAll58.75 vs70/66.25。相对RA FWT-7.5pp全来自Task9 45→15；最终Task6/7/8分别30/100/90，差-15/+5/-5pp。负结果保留，不重训重评。效果额外1840更新、主epoch76.93分钟、正式vector79.81分钟，Task8负拟合耗时无效原样保留；最终旧组超限4/18与12/24，双对偶不是保证。
服务器02:26自动转43，8隔离GPU更新0环境真实PASS，progress_seed43_20261022_022754启动505source/input hash及本地PASS。queue_20261022_023117观察02:31已Task6 42epoch6594主步0正式；Task6最终tensor尚待100epoch，不提前PASS。按41分阶段主训练77分钟+评估80分钟另计拟合入库审计，43预计05:10–05:50参考；不能因41负结果取消固定43。LIBERO50仍排队未GPU准入，data普通8.504GB/root全部13.374GB、system普通5.902GB，容量watcher仍ARMED未动。一次本地顺序SFTP下载因小文件往返慢主动中断exit1，DOWNLOAD_ATTEMPT01保留，resume_light_download.py并行只读补655缺项后950全SHA通过；无服务器写入或实验重启。以后43可先生成manifest，使用同白名单并行包装方式，不必重下已核验文件。
下一轮只核活跃43，不重复41/42审计。完成后audit_final.py/summarize_final.py已在41实证可用，再汇总三seed原episode和均值样本SD。本轮总队列仍PARTIAL，CAET-v1 GOAL科学阻塞/COVR取消不变。


## 2026-10-22 02:58更新

最新2026-10-22 02:58 CST：progress_seed43_20261022_025726，43正常Task8，230epoch47670主步/0正式；Task6实际tensor同RA、505冻结source/input、已完train6/7 retain6/7等exit0与本地逐步审计PASS。Task7全RA回退；Task8预测器留出PASS，MAE.253718对均值.488547/置换.277847，75支持时刻12旧task-demo，10epoch840额外更新。Task8已30epoch，迁移2486步、冲突关闭1594、保护非零4；不能把预测/激活当闭环收益。Task9尚未准入。最近Task8十epoch均13.78秒，剩主训练约47.7分钟（Task9用41实测18.99秒/epoch参考），评估参考79.8分钟另计拟合入库验收，43预计05:10–05:50仅参考。正常无恢复，不重复41/42最终审计。容量watcher仍ARMED，data普通8.205GB/root13.075GB、system普通5.902GB，LIBERO50未GPU准入。package_final.py新增--manifest-only，仅生成白名单manifest后用resume_light_download.py <目录>并行只读下载、全SHA再audit_final/summarize_final，避免下一次顺序SFTP中断；新开关仅语法检查，下载器和最终审计已41实证。未修改服务器源或实验预算。


## 2026-10-22 03:28更新

最新2026-10-22 03:28 CST：progress_seed43_20261022_032726，43正常Task9，330epoch80160主步/0正式；Task6实际tensor同RA、505冻结source/input、已完阶段exit0及本地逐步审计PASS。Task8完整100epoch，迁移7612/冲突5964/保护2578步；Task9前30epoch迁移2945/冲突1818/保护非零0步，只是当前读数，不能提前说全阶段无保护。Task8/9预测器留出均PASS；9的MAE.255011对均值.468748/置换.286582，64支持时刻18旧task-demo，980额外更新，两预测器合计1820。Task9最近十epoch15.50秒，余70epoch约18.1分钟；完整评估参考79.8分钟加拟合入库验收，43预计05:10–05:50参考。正常无恢复/无重复预检；data普通8.048GB/root12.918GB、system普通5.902GB，容量watcher仍ARMED，L50未GPU准入。下一轮应进正式评估，先真实进程与attempt核对，不把训练结束当COMPLETED。最终用package_final.py 43 --manifest-only后resume_light_download.py <包目录>，全SHA后audit_final.py <包> 43和summarize_final.py <包>；然后三seed完整均值SD，勿提前使用部分指标。
