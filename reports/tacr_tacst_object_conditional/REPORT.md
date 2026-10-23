# PARTIAL：TAC-R / TAC-R＋TAC-ST 已排队

观察2026-10-23 00:15:45 CST。snapshot_readonly_20261023_001540/LOCAL_AUDIT PASS：新controller95924实际存活，1462冻结hash一致，六个输出均不存在、尚未GPU准入，无失败。唯一活跃GPU为旧ALIGN42评估94045，未中断。新队列等待旧ALIGN OBJECT42/43及其条件GOAL分支自然有效结束，故没有新实验训练吞吐或完成ETA，不能拿CPU测试伪造GPU进度。

两组为ALIGN＋TAC-R与独立官方初始化的ALIGN＋TAC-R＋TAC-ST。各自完整seed41 AUC严格>原RA41 701/960（73.020833333333%）才自动扩展自身42/43。不是三seed RA均值，不增加其他性能门槛；R未过也照常运行RST41。旧ALIGN→GOAL原三seed均值门槛保留。

三seed新图Task8/9独立支持为156/207、152/205、180/200；零训练身份、动作前观测/下一动作/下一效果、RA成员SHA及新增夹爪sign条件检查PASS。实际Temporal/GMM CPU梯度与部署融合PASS；不代表GPU准入通过或闭环收益。每run预算仍400epoch105500主步，新增105500表征optimizer调用独立计，4主CP100键＋4部署14键、200正式20vector最终验收待后续。

六run及旧待运行四run最坏起步容量界通过，普通system留1GiB、data root留2GiB；不删旧资产、不下载payload。每run启动前重新核容量。当前没有新性能结果，全部科学结论待完整验收，最终保留所有负结果及探索性选择限制。

2026-10-23 00:21补充：六入口15模块实际独立环境导入PASS、无输出/模型前向/优化器/环境，1462冻结hash不变；同一监控保存的30分钟调度与完整新旧队列提示已复核。最终包装仍待完成实验真实验收。


2026-10-23 00:51 CST：snapshot_readonly_20261023_005155/LOCAL_AUDIT与CAPACITY_LOCAL_AUDIT PASS。controller95924仍正常等待旧队列，1462冻结hash不变、六输出不存在、未GPU准入；GPU仍为旧evaluate94045，无新失败/恢复/重启。data普通10.129GB/root14.999GB、system普通7.979GB；旧四待运行与新六run的保守剩余上界仍通过，data保留2GiB后余0.826GB、system保留1GiB后余1.752GB；这不是最终容量验收。旧ALIGN42正式计数见同轮A全快照，不沿用旧10/200。新队列无可用吞吐与ETA，正常等待不通知。


2026-10-23 01:25 CST：snapshot_readonly_20261023_012140/LOCAL_AUDIT PASS，新95924仍WAITING、1462hash不变、六输出不存在，无GPU准入。旧ALIGN42已在A/final_OBJECT_42_20261023_012252完整本地独立PASS，AUC83.020833%对同seedRA71.5625%；旧43已自动GPU准入并训练中，本N不越过旧GOAL条件流程。capacity_readonly_20261023_012323按剩余新增分配计当前43已写稳定私有1.009GB的信用，data root13.990GB，对最坏剩余11.017GB＋2GiB后仍余0.826GB；system普通7.979GB仍余1.752GB。不能拿部署时全额四run界重复扣当前已耗的空间而误报满盘；此仍是规划界，不是最终容量验收。

外置resume_light_download.py发现原复制工具仅允许/data元数据路径，已加明确的/root/mlr_ifa/experiments/前缀以匹配冻结TACR41/43与TACRST42输出；仍只.json/.jsonl/.exit、包目录内路径、<8MiB、manifest全SHA，不下载pth或payload。仅外置包装修改/语法验证，无服务器源码或合同修改；实际新组最终下载仍待真实验收。N无新性能结果或可用吞吐ETA。


2026-10-23 01:51–01:52 CST：snapshot_readonly_20261023_015144/LOCAL_AUDIT PASS，95924仍正常WAITING、1462hash不变、六输出未创建、无GPU准入/新性能结果。旧A的43正常389完整epoch101518步/0正式，Task9已89轮；GOAL门槛仍待完整43。capacity_readonly_20261023_015201按43稳定私有已分配1.184GB抵扣后，data root13.8146GB，对最坏剩余新增10.8415GB＋2GiB保留后余0.8256GB；system普通7.9794GB保留及新三run界后余1.7517GB。无旧资产删除、活动payload操作、付费扩容或预算改变；仍非全程最终容量验收。新组继续等A全部所选分支有效退出及互斥锁，无可用新GPU吞吐/总ETA。正常保存发布后结束，不守候。


2026-10-23 02:21–02:22 CST：snapshot_readonly_20261023_022140/LOCAL_AUDIT PASS，95924仍WAITING，1462hash不变，六输出不存在且尚未GPU准入。capacity_readonly_20261023_022156剩余新增界PASS：43当前稳定私有分配1.3206GB，data root13.6784GB，对最坏剩余新增10.7053GB＋2GiB保留后余0.8256GB；system普通7.9794GB仍余1.7517GB。未删除旧资产或操作活动payload。原A的43阶段/正式进度以同轮A完整快照为准，N不越过旧队列/GOAL门槛，不新增训练/环境/付费资源；仍PARTIAL，新N无实际GPU吞吐/总ETA。


同轮A的43已完成主预算，正式60/200，0技术异常；N仍无GPU准入，严格等待旧队列全部所选分支自然有效结束。


2026-10-23 02:51–02:52 CST：snapshot_readonly_20261023_025144/LOCAL_AUDIT PASS，95924正常WAITING，1462hash一致、六输出不存在，尚未GPU准入，无新失败或恢复。旧A的43仍原evaluate98808正常运行，正式计数以同轮A独立快照为准。capacity_readonly_20261023_025200保守剩余新增界PASS：data root13.6781GB，对最坏剩余新增10.7050GB＋2GiB保留后余0.8256GB；system普通7.9794GB余1.7517GB。不动活动payload或删旧资产，不增付费资源/降预算，无新N实际GPU吞吐或总ETA；仍PARTIAL，正常保存发布结束本轮，不守候。

## 2026-10-23 03:28 CST — 上游OBJECT门通过，继续等完整GOAL队列

`snapshot_readonly_20261023_032141/LOCAL_AUDIT.json`PASS；1462冻结hash未变，95924仍WAITING_PRIOR_QUEUE_NATURAL_EXIT，六输出不存在。A OBJECT43已完整本地验收，三seed等权AUC80.173611%对RA73.229167%，精确门通过并已自然进入GOAL41。A整体仍PARTIAL，不能绕过它启动N；TACR/RST各自seed41仍只与同seedRA41的701/960比较。

`capacity_readonly_20261023_032343`PASS_NOT_FINAL：data root12.612702GB扣剩余新增9.479571GB和2GiB后余0.985648GB；system普通7.979397GB扣3新run及1GiB后余1.751695GB。已接受43不再全额预留，GOAL41稳定私有分配0.828350GB抵扣剩余界。0模型前向/0优化器/0环境/0服务器写入，不改合同、不删资产。新模块GPU及最终包装仍待真实运行，未给新模块总ETA。整体PARTIAL。

## 2026-10-23 03:51 CST — 正常等待上游GOAL，未GPU准入

`snapshot_readonly_20261023_035136/LOCAL_AUDIT.json` PASS_PROGRESS_NOT_FINAL，1462冻结hash一致，controller95924保持WAITING_PRIOR_QUEUE_NATURAL_EXIT，无新失败/恢复，六输出未创建。不调用worker或绕过A，所有取消任务不恢复。A当前GPU进程102448，详细阶段以同轮A独立快照为准，不提前推算任何收益。

`capacity_readonly_20261023_035149`剩余新增界PASS_NOT_FINAL，data root12.578677GB，对余新增9.209231GB+2GiB保留后余1.221962GB；system普通7.979393GB，扣三run及1GiB后余1.751690GB。GOAL41稳定私有已分配1.098691GB抵扣新增界，已接受OBJECT不再额外预留。不删旧资产、不改源、不增加付费资源；0模型前向/0优化器/0环境/0服务器写入。N实际GPU吞吐及总ETA仍未知，整体PARTIAL，正常保存发布后结束不守候。

## 2026-10-23 04:21 CST — 正常等待上游GOAL，六run仍未准入

`snapshot_readonly_20261023_042133/LOCAL_AUDIT.json` PASS_PROGRESS_NOT_FINAL，1462冻结源调度hash一致，无失败/恢复，95924保持WAITING_PRIOR_QUEUE_NATURAL_EXIT，六输出未创建。上游A GPU进程102662，正式阶段以本轮A独立快照为准；不越过A完整队列自然有效exit0/receipt/hash/互斥锁，其他已取消任务不恢复。

`capacity_readonly_20261023_042146`剩余新增界PASS_NOT_FINAL：data root12.415734GB，扣余新增9.046321GB+2GiB后余1.221929GB；system普通7.979377GB，扣三run+1GiB后余1.751674GB。GOAL41稳定私有已分配1.261601GB抵扣新增界，不重复全额预留已消耗或已接受产物。0模型前向/0优化器/0环境/0服务器写入，不删除旧资产、改参数、下载payload或增加付费资源。N无真实GPU吞吐或总ETA，仍PARTIAL，正常保存发布后结束，不守候。

## 2026-10-23 04:51 CST — 正常等待，1462冻结hash不变

`snapshot_readonly_20261023_045134/LOCAL_AUDIT.json` PASS_PROGRESS_NOT_FINAL；95924仍WAITING_PRIOR_QUEUE_NATURAL_EXIT，六输出未创建，未GPU准入，无失败/恢复。上游GOAL41仍evaluate102662占GPU，正式记录以同轮A完整审计为准。N不绕过上游A/互斥锁/自然有效退出，不启动任何新worker，不复活取消任务。

`capacity_readonly_20261023_045147`剩余新增界PASS_NOT_FINAL：data root12.415291GB，扣余新增9.045878GB和2GiB后余1.221929GB；system普通7.979356GB，扣三run及1GiB后余1.751654GB。GOAL41稳定私有已分配1.262043GB抵扣新增界。不删旧资产、不下载payload、不改预算或付费扩容，0模型前向/0优化器/0环境/0服务器写入。新模块仍无真实GPU吞吐或总ETA，整体PARTIAL，正常保存发布后结束，不守候。

## 2026-10-23 05:25 CST — 上游GOAL41完整本地验收，继续等待GOAL42/43

`snapshot_readonly_20261023_052136/LOCAL_AUDIT.json` PASS_PROGRESS_NOT_FINAL，1462冻结源调度hash未变，95924正常WAITING，六输出不存在。A GOAL41已final_GOAL_41_20261023_052225完整本地独立PASS，AUC83.4375 vs同seedRA78.75（+4.6875pp），但NBT+11.111111pp低好、最终Task8-15pp保留，不外推N收益。A自然进入GOAL42，247完整epoch32595主步0正式、自己5GPU主调用0辅助0环境及1209hash/270输入hash/计步PASS，详细见A本轮报告。N仍等A整个队列自然有效exit0/完整receipt/GPU空闲/互斥锁释放，不绕过；各自OBJECT41门仍701/960。

`capacity_readonly_20261023_052148`剩余新增界PASS_NOT_FINAL：data root11.110191GB扣余新增7.978082GB+2GiB后余0.984625GB；system普通7.979352GB扣三run+1GiB后余1.751649GB。已接受GOAL41不再额外预留，GOAL42稳定私有0.611852GB抵扣；1不稳定文件未计信用。0模型前向/0优化器/0环境/0服务器写入，不删资产/改参/付费扩容。N真实GPU吞吐及总ETA未知；两队列整体PARTIAL，正常保存发布后结束不守候。

## 2026-10-23 05:51 CST — 正常等待上游GOAL42，未GPU准入

`snapshot_readonly_20261023_055136/LOCAL_AUDIT.json` PASS_PROGRESS_NOT_FINAL；1462冻结hash一致，无新失败或恢复，95924仍WAITING_PRIOR_QUEUE_NATURAL_EXIT。六输出未创建，不能把CPU验证当GPU准入。A当前GPU进程106640，具体正式进度以同轮A完整审计为准；N不绕过A/互斥锁，不kill或重复启动正常训练评估，取消任务不恢复。

`capacity_readonly_20261023_055148`剩余新增界PASS_NOT_FINAL：data root11.153408GB，扣余新增7.328592GB+2GiB后余1.677332GB；system普通7.979123GB，扣三run+1GiB后余1.751420GB。GOAL42稳定私有已分配1.261343GB抵扣新增界；已接受OBJECT和GOAL41不再额外预留。不删旧资产、下载payload、改精度batch更新或增加付费资源，0模型前向/0优化器/0环境/0服务器写入。N实际吞吐和整体ETA仍未知，总PARTIAL，保存发布后结束本轮，不守候。

## 2026-10-23 06:22 CST — 只读SSH连接重试已恢复，队列正常等待

初次snapshot_readonly_20261023_062138在认证前SSH protocol banner读取EOF失败，未启动远端脚本；LOCAL_CONNECTION_ATTEMPT.json原地留证。仅重试监控连接一次，snapshot_readonly_20261023_062227/LOCAL_AUDIT PASS_PROGRESS_NOT_FINAL，RESOLUTION.json绑定该审计SHA，原失败不覆盖。0服务器写入/0训练或评估重启，不是训练故障恢复，无需用户操作。

1462冻结hash一致，95924仍WAITING_PRIOR_QUEUE_NATURAL_EXIT，六输出不存在，未GPU准入；A的GOAL42正式100/200、10vector+1进行中、0技术异常，完整主预算和逐步/逐episode审计PASS。N不绕过A或改变独立OBJECT41的701/960门槛。capacity_readonly_20261023_062149 PASS_NOT_FINAL：data root11.153052GB扣余新增7.328236GB+2GiB后余1.677332GB，system普通7.978947GB扣三run+1GiB后余1.751244GB，GOAL42私有已分配1.261699GB抵扣新增界。不删旧资产、不下载payload、不改科学协议或新付费资源。总PARTIAL，正常保存发布后结束不守候。

## 2026-10-23 06:51 CST — 正常等待上游GOAL42

`snapshot_readonly_20261023_065149/LOCAL_AUDIT.json` PASS_PROGRESS_NOT_FINAL；1462冻结源调度hash未变，95924仍WAITING_PRIOR_QUEUE_NATURAL_EXIT，六输出不存在、尚未GPU准入，无新失败/恢复。此次只读SSH连接成功，先前认证前banner失败及重连证据原样保留，不当作训练故障。上游A仍evaluate106640，正式记录以同轮A独立快照为准，N不绕过完整上游有效退出/receipt/GPU空闲/锁释放。

`capacity_readonly_20261023_065203`剩余新增界PASS_NOT_FINAL：data root11.152695GB，扣余新增7.327879GB+2GiB后余1.677332GB；system普通7.978930GB，扣三run+1GiB后余1.751228GB。GOAL42稳定私有已分配1.262055GB抵扣新增界；已接受产物不额外全额预留。0模型前向/0优化器/0环境/0服务器写入，不删旧资产、改预算、下载payload或增加付费资源。N无真实吞吐/总ETA，整体PARTIAL，正常保存发布后结束本轮不守候。

## 2026-10-23 07:25 CST — 等待最后一组GOAL43

`snapshot_readonly_20261023_072149/LOCAL_AUDIT.json` PASS_PROGRESS_NOT_FINAL，1462冻结hash一致，95924仍WAITING，六输出不存在。A GOAL42已final_GOAL_42_20261023_072244完整本地验收PASS，AUC82.708333 vs RA82.5（+0.208333pp），但NBT+1.111111pp、OldFinal-3.333333、FinalAll-2.5和Task8最终-15pp保留。A约07:00自然进入最后GOAL43，07:21快照245完整epoch32225主步0正式，自身5GPU主调用0辅助0环境及1209/283输入hash、初始化和mask账目PASS，无科学改动/重启。详细本轮A报告为准；未完整43不作三seed汇总。N仍等A完整自然有效exit0/全部receipt/GPU空闲/锁释放，不因已完成42绕过；N的OBJECT41门仍701/960。

`capacity_readonly_20261023_072202`剩余新增界PASS_NOT_FINAL：data root10.194199GB扣余新增5.913918GB+2GiB后余2.132797GB，system普通7.978926GB扣三run+1GiB后余1.751224GB；已验收42不再预留，43私有0.958030GB抵扣。0模型前向/0优化器/0环境/0服务器写入，不删资产、改预算或新付费。新N真实GPU吞吐/总ETA仍未知，总PARTIAL。A参考08:50–09:20完成只是基于现有实测，不承诺N开始时间。正常保存发布后结束不守候。

## 2026-10-23 07:51 CST — 正常等待最后GOAL43

`snapshot_readonly_20261023_075148/LOCAL_AUDIT.json` PASS_PROGRESS_NOT_FINAL，1462冻结源调度hash一致，95924仍WAITING_PRIOR_QUEUE_NATURAL_EXIT，六输出不存在，自己GPU准入尚待，不套用A报告。A当前GPU进程110830，正式进度以同轮A完整独立快照为准。不绕过A自然有效exit0/receipt/GPU空闲/互斥锁，不kill正常工作，所有取消任务不恢复。

`capacity_readonly_20261023_075204`剩余新增界PASS_NOT_FINAL：data root9.890849GB，扣余新增5.610646GB+2GiB后余2.132719GB；system普通7.978910GB，扣三run+1GiB后余1.751207GB。GOAL43稳定私有1.261302GB抵扣新增界，已验收产物不额外预留。0模型前向/0优化器/0环境/0服务器写入，不删旧资产、下载payload、调精度batch更新或新付费。N真实吞吐和总ETA未知；整体PARTIAL，保存发布后结束，不守候。

## 2026-10-23 08:21 CST — 正常等待上游GOAL43

`snapshot_readonly_20261023_082156/LOCAL_AUDIT.json` PASS_PROGRESS_NOT_FINAL，1462冻结源调度hash不变，95924仍WAITING_PRIOR_QUEUE_NATURAL_EXIT，六输出未创建、尚未GPU准入，无失败/恢复。本轮只读连接正常。A当前evaluate110830，具体正式进度以同轮A完整独立快照为准；不绕过A有效退出/receipt/GPU空闲/锁释放，不重复启动或kill正常运行。

`capacity_readonly_20261023_082210`剩余新增界PASS_NOT_FINAL：data root9.890497GB，扣余新增5.610294GB+2GiB后余2.132719GB；system普通7.978893GB，扣三run+1GiB后余1.751191GB。GOAL43稳定私有已分配1.261654GB抵扣新增界，已验收产物不额外预留。0模型前向/0优化器/0环境/0服务器写入，不删旧资产、下载payload、改精度batch更新或新付费。新N真实吞吐与整体ETA未知，总PARTIAL；保存发布后结束不守候。
