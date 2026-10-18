# SC-v2 GOAL seed42：正式运行与条件扩展

先完整阅读本文件、PROTOCOL.md、RUN_STATUS.json、LOCAL_LAUNCH_ACCEPTANCE.json、launch_evidence/CONTRACT.json及恢复补充。用户先要求离线验收，之后明确授权验收通过即先跑seed42，双门槛通过才补43/44；评估回答“按上次实验的计划来”，本轮解释已明确告知并落实为原SC-v1的200条stage global RNG协议，复用旧基线。600条逐轨迹Generator共同重评没有选定，不启动。

SSH root@jq1.9gpu.com:11920已授权，密码仅内存/getpass，不写日志或自动化。W=/root/mlr_ifa/experiment_workspaces/SCv2_Coreset_GOAL_seed42_20261018，O=/root/mlr_ifa/experiments/SCv2_Coreset_GOAL_seed42_20261018，P=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed42_20261017。source W/env.sh，Python=/data/mlr_ifa/env/bin/python。系统盘产物已在GOAL授权内；旧SC-v1/Random/Coreset/其他实验全部只读。

离线证据execution/20261018_scv2_offline。四任务全部4759窗口身份与原S0完整顺序重现、4781输入哈希不变；5/2/4/6次交换，独立时间495/426/449/981均持平，示范/原终点支持保留，R和mean均不超S0。独立检查重现全部17次贪心选择及终止无合法交换。Task6–8共11条干预；Task9存储发生训练后，不是Task9性能直接来源。首轮离线dtype校验失败exit1保留，float32绑定修复后二次CPUexit0，非训练失败。

新源只修改sc_exp选择/存储/准备/验收，训练器、模型、loss和evaluate/eval_metric字节与父SC-v1相同。5软件测试及4次隔离GPU原生优化器恢复PASS，零环境。四任务实际storage集成成员与离线逐项一致、全RNG不变。不要重复预检或增加rollout。正式进程从官方GOAL权重空buffer重新seed42，原stateTrue缓存、同模态合同、100完整epoch×4、batch10、76100step、BC+.1平均IFA/gamma.7/warmup500逐step线性/concatshuffle/sampler0不变。内部arm=sc_coreset，但独立SCv2目录。当前任务SC-v2入库配额499/445/452/985，原标定42/FP64距离不变。每次实际成员必须等于离线有序清单；不重选旧任务。IFA按实际buffer原算法重算并报告。Task6首次入库前实际tensor/初始化RNG必须等于父。

先运行W/monitor_snapshot.py并查看source/config/input/modal/feature冻结哈希、真实进程、GPU磁盘、RUN_STATUS、epochs/recovery/storage、日志时间、train/evaluate/audit/pipeline/supervisor exit、episode及attempt。防止重复启动。20:20 CST supervisor PID43881、pipeline43886、正式train已启动。原launch_evidence采样于初始化阶段0epoch，不是失败。后台supervise.sh自动依次train/evaluate/audit、pipeline退出后final_process_audit，成功后supervisor.exit0。

评估仅四ep99最终模型原0–19初态/20每cell/10lane/env42,43/600步/5零动作/globalRNG每stage42/argmax分量Normal。总200条20vector，不加680/9000/新RNG，不重训重评基线。部分评估拒绝盲重启，正常失败不重试。

可恢复工程堵塞须确认进程已退出、合同一致、留存失败；从本run完整epoch恢复model/optimizer/scheduler/grads/modes/allRNG/cursor；存储按预提交manifest幂等复制，不重抽/重复步骤或样本。不能盲重启pipeline、改冻结代码或重贴合同。评估技术恢复必须保持原全局RNG顺序和全部attempt；无法保证则BLOCKED。只读汇总错误可留证外置修复，不重跑环境。

工程完成需400epoch76100step、4CP严格加载、200真实20vector、所有SC约束/载荷/输入保护/IFA/Task6tensor一致、完整矩阵/精确AUC FWT NBT OldFinal FinalAll、pipeline与supervisor退出0、无残留且PROCESS_EXIT_ACCEPTANCE PASS。训练完不是完成。

双门槛固定：AUC>63/80且OldFinal>=41/60，对照为原seed42 Coreset；Random只参考。未全过则报告停止，不扩展。全部通过才准备seed43/44独立配置/合同/基线输入；须分别先执行同一SC-v2离线规则与真实窗口核验，验证有Task6–8实际干预才训练，保持全部距离/容差/预算不变，不重训基线。当前代码只42，不能直接换环境变量冒充完整参数化；需要审计seed路径/基线门槛/初始化比较/评估记录与指标。不得按结果调参。

每次正常检查保存轻量快照和实测ETA后结束，不持续守候。上轮同GPU训练约41分钟、评估约73分钟，本轮启动预计约22:15–22:40完成，实际吞吐再更新。通常2小时，接近结束30分钟；状态不变安静，仅有实质进展/失败/恢复/完成/需用户行动通知。

W/package_light.py只收轻量资料并去除binding last_action；禁止权重、latent、buffer、演示下载发布。校验PACKAGE_MANIFEST后本地独立复算。每轮按AGENTS向publish_mlr_branch的MLR推送本轮代码报告轻量证据，审查secret/体积/文件类型，核对远端SHA，不force push，无变化不空提交。完整交付且无获准扩展工作后删除本监控。披露探索性单seed/后续条件扩展、很小成员变化、RNG未逐轨迹隔离、pretrain state unknown、IFA间接变化/调度补全、离线覆盖不能保证控制收益。

最新检查2026-10-18 21:33 CST：400epoch/76100step全部完成，train.exit=0；100/200条、10/20向量完成，当前stage8/task8/group0，0技术异常。实际Task6–9成员顺序/2381文件SHA与离线一致，替换5/2/4/6；Task6–8独立监督495/426/449均保持，两项覆盖守线。163源文件/196合同文件匹配，IFA任务对仍与父一致但相似度不同。四CP仅保存/哈希验证，严格加载等最终验收尚未完成。新证据progress_20261018_2128，按实测ETA22:10–22:30，不需要重启或追加测试。

最新检查2026-10-18 21:56:27 CST：170/200条、17/20向量，当前stage9/task8/group1，无技术异常；入库/原输入/冻结哈希再次PASS。最终阶段Task6=6/20、Task7=20/20，Task8仅8/10尚未完整；不提前算最终指标。ETA评估约22:10，完整验收22:10–22:30。证据progress_20261018_2155。

最终状态2026-10-18 22:11 CST：服务器退出验收PASS，400epoch76100step4CP严格加载200条20vector，所有退出0且无残留；本地完整复算PASS。AUC69.0625%低于Coreset78.75%，OldFinal70%达到68.3333%门槛，双门槛未同时通过，因此NO_EXTENSION，不启动43/44。对比SC-v1见FINAL_REPORT.md和LOCAL_FINAL_ACCEPTANCE.json。定时任务sc-v2-goal-seed42已删除；所有工作结束，不重启实验。
