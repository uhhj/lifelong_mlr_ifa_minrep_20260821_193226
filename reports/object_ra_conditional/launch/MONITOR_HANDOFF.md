# OBJECT RA → ZR-v2 条件实验交接

先完整读PROTOCOL.md、DECISION_GRAPH.json、PARENTS.json、各seed ACTUAL_SPEC、SOURCE_DIFF、RUN_STATUS、最新进度和恢复补充。用户最新确认第一关比较Random+IFA，不能用原RA GOAL默认Coreset门槛。两关仅三seed平均AUC严格提高且至少2/3个seed提高、全部工程PASS；OldFinal/FWT/NBT仅报告。不得事后删seed或改门槛。seed43 OBJECT始终repeat1，首次高分不参与。

凭据沿本对话SSH root@jq1.9gpu.com:11920，仅内存/getpass；不写任何文件日志或自动化。W/Q=/data/mlr_ifa/experiment_workspaces/RA_OBJECT_conditional_20261019，O=/data/mlr_ifa/experiments/RA_Coreset_OBJECT_seed{s}_20261019，s41/42/43。本轮数据盘约29GiB，旧数据/root资产只读。无需另问本轮数据盘存储。

本地 check_progress.py 可读取Q/monitor_snapshot.py；--package额外verify冻结hash并生成/下载安全轻量包。每轮检查实际GPU/进程/磁盘、queue/wrapper与阶段exit/log、epoch/recovery/storage/正式records/attempt。禁止重复deploy、prepare、freeze、预检或启动整个queue。有限队列已安排41→42→43各admission/supervise→aggregate_stage1→final_queue_audit，无性能中途停止。admission为8软件+5921真实载荷/原两名单/全删除代价独立验收+4GPU隔离原生更新恢复，0环境。正式每seed400epoch/105500step/4ep99CP200records20vector；三seed1200/316500/12CP600records60vector。

RA算法距离/标定42/sampler0、半配额Random硬锚点、最少交换、loss/AdamW/原路径排序loader/eval GMM都冻结。唯一套件改为OBJECT/gamma.3和对应权重/数量。窗口1562/1480/1560/1319，配额781/740/780/660，anchors390/370/390/330。三入口stateTrue。官方预训练state配置缺失保持unknown/DISCLOSED_GAP，不再暂停。Task6 actual tensor与init RNG要求同父；若不通过保留失败，不偷偷宽阈值。41/43父历史4090、本轮3090需披露；不能因此静默重训基线或换首次43。

OBJECT恒定动作窗口的真实时间身份可能多解，仅诊断记unknown和候选真实身份；载荷SHA及末端动作严格一致。原RA不使用时间/事件分组，不为诊断歧义改选择决策。复制父43保留feature_origin_contract_sha256来源，不给旧缓存贴新合同。

工程恢复：确认相关进程停止、原失败证据和合同/输入一致后，仅本seed本arm完整epoch恢复模型optimizer/scheduler/grads/modes/全部RNG/cursor；按预提交名单幂等补齐，不能重抽或重复计步。部分原globalRNG评估禁盲重启，正常失败不重试，技术恢复不能保证原RNG环境顺序则BLOCKED。冻结源不改/不重贴合同；只读汇总故障可留证外置修复。工程BLOCKED不能当作指标未通过转fallback。

第一阶段结束要queue/wrapper0、每seed严格4CP实际加载、Task6/init/全哈希/配额有序成员与offline等、600records60attempt、无残留、FINAL_QUEUE_ACCEPTANCE PASS。下载只含元数据/episode/log的PACKAGE_MANIFEST包并逐文件验证，在本地按真实记录独立复算三seed完整矩阵、精确Fraction指标、均值样本SD与RA-R/RA-C差和成本。先验收发布，再进入已授权条件分支；STAGE1_RESULTS的next_authorized_branch只作声明，不是后续已启动证据。

## 条件分支的后续实际执行责任

用户已授权整个分支，不要只停在第一关或再次询问是否运行。当前服务器仅部署第一阶段；后续需在门槛确定后按冻结v2版本实现真实seed/suite/parent适配、测试和独立合同，禁止直接运行旧hardcoded seed43入口。无需重新设计算法：参考本地 execution/20261018_zrv2_seed43 的attachment完整PROTOCOL/spec、source、admission.py、EXECUTION_DECISIONS、completion_20261018及其外置退出验收路径修复。远端原W=/root/mlr_ifa/experiment_workspaces/ZRv2_OBJECT_seed43_20261018，其source和原repeat1保持只读。

若第一关PASS：新增OBJECT RA+ZR-v2+IFA seeds41/42/43，**父为本轮RA**的对应四个ep99和对应RA buffer有序成员。不是旧Coreset buffer，也不是ZR-v1共享共同训练。每seed40校准epoch/10550步/4cal_epoch9快照/200闭环。三seed结束工程完整通过，再相对本轮RA同样计算第二关。第二关PASS后，新增GOAL RA+ZR-v2+IFA seeds41/42/43，父为已完成GOAL RA各seed；GOAL各40校准epoch/7610步/200闭环。第二关不通过则报告停止。

若第一关FAIL（完整工程PASS但指标条件未过）：新增GOAL Coreset+ZR-v2+IFA seeds41/42/43/44，父为已有对应Coreset，40校准epoch/7610步/200闭环每seed。不得在此路径继续OBJECT组合。

GOAL父RA41=/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019；42=/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed42_20261018；43=/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed43_20261019。Coreset父41=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019；42/43/44=/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed{s}_20261017。先真实验证模型/manifest/旧episode，不凭路径假定可用。

ZR-v2机制全部不变：每阶段正确父Ck和固定官方T0冻结eval，FP64按已见规范语言构建双向门（真实batch1/10/32及所有合法缓存语言准入），rank8UV局部seed=s+task、七维0.1*tanh均值前激活修正、共享所有GMM分量、scale/logits不改；旧UV仅自身buffer样本重新校准。任务标签只限制反向归属，推理全语言门，无taskID路由。旧支路无本任务监督grad=None及Adam状态不更新，公共无梯度。每阶段10完整epoch、batch10、末端NLL、AdamW1e-4原参数、warmup500逐step、sampler0，末轮ep9，0早停。新增预算/固定T0开销要计量。GOAL真实门未验收前不能承诺通过，失败BLOCKED不硬归零/放宽阈值。每新seed最多4隔离GPU优化器预检、0环境，不追加闭环smoke。原0–19/20cell/10lane/env[s,s+1]/600/5zero/stage global RNG/argmaxNormal，独立进程200正式；不恢复9000或新增680/独立Generator。

每次检查/审计和实验轮次按AGENTS用publish_mlr_branch提交本轮review过的代码、报告和轻量证据至MLR，精确stage、审查类型大小secret/hash、不forcepush并核对远端SHA存receipt。不要发布source zip、checkpoint、RNG二进制、latent、buffer、原actions、语言门数组。保持旧结果和用户修改。正常运行保存轻量快照与实测ETA即结束，不守候；通常2小时，临近分支/最终验收可30分钟同自动化，不建重复。仅实质变化/失败/恢复/完成/需用户行动通知；未完成不假报COMPLETED。全部授权分支真实验收和发布后删除自动化并停止，不擅自新增seed/套件/调参或付费资源。


## 2026-10-19 23:05最新覆盖
用户要求后续分支服务器自动部署；部署前发现第一阶段Task6实际张量断言已失败。见BLOCKED.md及TASK6_READONLY_DIAGNOSIS.json。依“无法安全恢复则停止”已停止部署及自动推进。不得自动resume、不能触发fallback；需先解决当前冻结合同的真实张量差异。原初始和更新代码/配置一致，根因未证实。原exit1保留。
