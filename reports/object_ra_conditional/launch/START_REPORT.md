# OBJECT RA-Coreset＋IFA三seed与后续条件实验

状态 **PARTIAL**。2026-10-19 22:44 CST启动有限队列，最新快照时间2026-10-19T22:48:00.862239+08:00，seed41已开始正式Task6；随后轻量元数据收集已有5epoch/785步，0正式闭环。42、43排队未执行预检/训练。未声称全部工程验收或性能通过。

seed41已通过8软件测试、5921真实窗口载荷、原Random/Coreset有序名单、所有交换删除代价独立验收及4次隔离真实GPU更新恢复（0环境）。实际离线交换Task6–9为199/191/196/167，配额781/740/780/660，锚点390/370/390/330。Task6有4条动作窗口存在时间身份歧义，如实记unknown，未用于任何选样规则或假定时间；其他任务该歧义0。新source冻结，170文件字节沿用，7处suite/seed/计数/来源/诊断与门槛适配，原选择器、模型、loss/优化器计算、loader和评估核心规则不变；source.patch与SOURCE_DIFF记录全部实际修改。

用户已固定路径：

1. 完整执行OBJECT seeds41/42/43 RA+IFA，比较各自Random+IFA。三个seed平均AUC严格提高，至少2/3个seed AUC提高，且全部工程验收通过，才通过第一关。OldFinal/FWT/NBT只报告。
2. 第一关通过：在本轮RA四阶段模型及RA记忆上做OBJECT RA+ZR-v2+IFA三个seed，公共模型无需再训练。第二关与本轮RA比较，采用同样AUC规则。
3. 第二关通过：GOAL RA+ZR-v2+IFA seeds41/42/43；第二关未通过停止。
4. 第一关性能未通过：改为GOAL Coreset+ZR-v2+IFA seeds41/42/43/44。工程失败须处理或BLOCKED，不能冒充性能失败进入替代分支。

三seed Random参考AUC分别为41=66.875%、42=66.145833%、43第二次repeat1=71.875%；平均68.298611111%（精确1967/2880）。已按原1200条Random/Coreset历史episode只读复算，全部与原SUMMARY吻合，输入哈希见BASELINE_EPISODE_AUDIT。未重训或重评任何基线。此处是state=True本研究基线，不使用早期state=False E0数字。

第一阶段三seed共1200epoch/316500步/12ep99CP/600正式记录/60vector。OBJECT每任务100完整epoch、batch10、BC+.1平均IFA/gamma.3、warmup500逐step、sampler0、原concat排序shuffle。原0–19初态、20/cell、10lane/env[s,s+1]/600步/5zero、阶段global动作RNG及argmax分量Normal不变。第二阶段每OBJECT seed额外40残差校准epoch/10550步/200闭环，GOAL每seed额外40/7610/200；额外T0与校准预算单独报告，不宣称等算力。

后台当前只部署第一阶段，后续按已授权门槛由自动化继续真实适配、准入与执行，不能把已写的条件分支当成已经训练的结果。监控ra-object-zr-v2每2小时检查，可恢复工程堵塞按完整epoch和冻结合同处理，临近完成可缩短；全部授权分支验收和发布后自动取消。

资源：RTX3090空闲时启动；数据盘初始约29.3GiB可用，新产物放/data专用目录，旧/root及/data资产只读。第一阶段估计6–8小时，按此前同GPU训练/评估吞吐推算而非完成保证，后续实测更新。无需让本地持续守候，服务器有限队列在没有阻塞时自动完成三个seed；后续条件分支由定时检查在验收后推进。

真实命令：probe.py退出0、build_stage1.py退出0、deploy_and_launch.py退出0（远端安装+单次启动PID18767）、check_progress.py及--package退出0、verify_local_launch.py退出0、baseline_episode_audit.py远端退出0。软件/GPU预检原命令及exit日志在progress证据；下载33个seed41轻量文件逐manifest哈希通过。第一轮SOURCE查询两个不存在的本地旧命名文件仅为只读路径查找，随后定位现有实现，不涉及任何训练恢复或冻结文件修改。

输出根：`/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed{41,42,43}_20261019`；源码队列`/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019`。全流程保护旧资产，不删除有效数据、不下载上传权重或载荷、不申请新付费资源。当前状态不是COMPLETED，Task6实际tensor与父终点一致性仍须阶段结束验证；41/43历史4090对当前3090限制保留。全局动作随机流未逐轨迹隔离、排序/IFA间接变化、pretrain state未知及调度补全、条件分支探索性均需最终披露。
