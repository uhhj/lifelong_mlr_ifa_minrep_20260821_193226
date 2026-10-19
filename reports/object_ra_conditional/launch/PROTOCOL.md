# RA OBJECT 与 ZR-v2 条件队列

用户2026-10-19授权并明确确认分支门槛，保持已完成实验协议；不重训/重评已有父基线。先完整运行OBJECT seeds41/42/43的RA-Coreset+IFA，再按三个seed最终AUC的精确分数决定分支。不能按途中结果跳过seed。

第一关比较同seed Random+IFA：三个seed平均AUC严格提高、至少2/3个seed的AUC严格提高、全部工程验收PASS。OldFinal/FWT/NBT报告但不是门槛。seed43使用第二次完整repeat1的Random和Coreset，绑定其实际四模型/特征/入库及原episode，而非仅替换报表数字。

第一关通过：在本轮RA三个seed的四阶段模型及实际RA记忆上分别追加冻结版本ZR-v2校准，公共模型不重训。第二关相对本轮RA+IFA，同样平均AUC严格提高且至少2/3个seed提高、全部工程PASS。第二关通过才在GOAL seeds41/42/43已有对应RA父模型与RA记忆上校准ZR-v2；第二关未通过停止。

第一关未通过：不执行OBJECT组合，转为GOAL seeds41/42/43/44已有Coreset+IFA父模型与Coreset记忆上的ZR-v2校准。不能混用被取消的9000新初态协议或ZR-v1共享训练。后续分支已获授权，无需再问是否运行；必须先完成对应真实资源、源/config/输入与GPU工程准入。工程失败属于BLOCKED，不等同于性能未过，不能借工程失败切换fallback。

第一阶段每seed：官方OBJECT六任务预训练权重、state=True同seed既有特征、独立空buffer，任务6–9各100完整epoch，batch10，400epoch/105500step/4ep99CP/200正式episode/20vector。三seed1200epoch/316500step/12CP/600正式episode。当前窗口1562/1480/1560/1319，配额781/740/780/660，硬Random锚点390/370/390/330。

RA算法与GOAL冻结selector_reference/storage不变：原两视觉token及末端动作距离、全任务标定seed42、原KCenter有序S0、Random(seed)完整q无放回名单前floor(q/2)硬锚点、最少交换与全局最大/平均最近距离/固定ID删除tie。不加SC/Event/损失选样/覆盖硬底线/stable-slot。真实8×7动作窗口及SHA绑定；OBJECT恒定动作窗口可能无法唯一识别真实监督时间，此时如实记unknown和候选身份，不猜时间，不影响任何选样决策；时间诊断只报已知子集与未知数量。

同seed训练/数据/锚点/阶段评估global RNG，DistributedSampler seed0，BC+0.1平均IFA、gamma0.3、warmup500逐optimizer-step线性、原模型/AdamW/路径排序concat shuffle/原数据读取与BC末端目标不变。原IFA按实际新记忆计算，完整报告，不强行与父配对一致。Task6首入库前实际模型张量和初始化RNG必须与父相同；若失败停止，不以文件哈希或重新标合同绕过。历史41/43为4090，本轮3090，须保留硬件限制与实测诊断。

每seed先8参考软件测试、5921真实载荷和原Random/Coreset有序名单、全部交换删除代价独立验证及全局RNG保护，冻结后最多4隔离GPU原生optimizer更新恢复、0环境；通过后正式重新初始化。队列41→42→43，各准入/训练/原评估/严格4CP加载/退出验收；完成后只读三seed汇总精确门槛及无残留退出验收。已过步骤不重跑。

原闭环：各阶段已见任务10单元，每单元20条原0–19初态，10lane、env[s,s+1]、600策略步、5零动作、每stage重置global动作RNG、argmax GMM分量后Normal。每seed200条及20vector尝试，不加680/9000、不隔离轨迹Generator、不重试正常失败。公共前端和对应阶段下游权重必须正确完整加载。

ZR-v2继承execution/20261018_zrv2_seed43已完成真实实现：每阶段冻结对应公共Ck及官方固定T0，FP64双向门，rank8七维有界均值修正0.1*tanh，scale/logits不改，无taskID推理路由；当前全数据和合法旧buffer校准10完整epoch，旧UV暖启动并仅自身样本反向，无公共IFA反向。新增校准不是相同算力预算。OBJECT每seed40校准epoch/10550步/200闭环；GOAL每seed40epoch/7610步/200闭环。阈值、初始化局部seed=s+task、AdamW、500warmup、阶段epoch9终点、结构和恢复规则均沿用v2，不根据结果调参。GOAL语言门需要真实准入，未通过不能硬归零或放宽阈值。

数据盘现可用约29.3GiB，本轮新产物在/data专用目录，旧资产只读。先盘点并保留已有修改，无reset、删除有效资产、新付费GPU或权重上传。正式源和编排合同冻结；任何工程故障先确认进程退出、保存失败、核对合同和输入，只能本seed本arm完整epoch恢复全部模型/optimizer/scheduler/grads/modes/RNG/cursor，storage按预提交名单幂等补齐。禁止盲重启queue/partial评估、重抽、重复计步；不能保持原评估RNG和环境顺序则BLOCKED。只读汇总错误可外置留证修复。

先服务器验收所有预算、严格加载、Task6/源输入/配额锚点名单载荷与尝试账目，再轻量下载按manifest验证、本地独立逐episode复算全部矩阵、AUC/FWT/有符号NBT/OldFinal/FinalAll、逐seed差与均值样本SD(ddof1)、成本。只有完整退出0和无残留才能COMPLETED。三seed门槛用Fraction，所有负结果同等报告。探索性条件分支并非独立确认性试验；阶段global动作RNG未隔离、排序/IFA间接作用、pretrain state元数据unknown、原调度补全及额外T0/校准成本限制保持披露。

每轮实验/检查/审计按AGENTS发布代码、报告和轻量证据到publish_mlr_branch/MLR，审查类型体积secret和hash、精确stage、不forcepush、核对远端SHA。不得发布权重、latent、buffer、原actions、RNG二进制、语言数组或凭据。每2小时检查，临近结束可30分钟；可恢复工程阻塞处理后续行，完成全部条件分支验收发布后删除自动化。
