# OBJECT完整S去掉支持度校准：三个种子

用户2026-11-08明确授权OBJECT seed41/42/43：完整S只把q_j固定为1，其余协议不变；已有完整S直接只读复用，不重训不重评。本目录只新增一个Q1变体，不恢复已取消的旧分叉。Q1是完整S双候选门控方法，不是此前单候选Q。

唯一科学变化：对每个非空成功缓存，将有效保持权重固定为1；空缓存仍贡献0。候选保持损失、门控旧任务评分、方向诊断共同使用此权重。原始成功轨迹数和经验q=成功数/10继续保存为诊断，不参与加权。保留lambda_R=0.1、保持路由0.1、原固定4槽分母、原探针归一化、10步同state C/Q候选、两个epsilon0及严格保持改善门控。增加总体保持强度属于去掉校准的自然作用，不另行归一化或调整lambda。

每seed官方OBJECT初态与同seed已有S的tensor/RNG一致；Task6→Task9各100轮，RA数据/manifest/mask、模型和其余损失、AdamW、warmup500及每阶段原接受更新终点不变。各阶段批次数157/227/309/362，每run105500接受更新、195300生产物理optimizer调用、9100门控块。三run1200轮316500接受、585900生产、27300门控。自身隔离273optimizer/123准入autograd、最多108方向autograd单列。旧工程诊断不计入本轮预算，不为发布追加测试或实验。

每阶段自己的10次训练采集（初态20–29），每任务缓存quota128、不补采；只从自己的成功轨迹形成缓存，不能复制已有S的缓存。三run总120采集。训练结束后每阶段已学任务正式评估初态0–19，600steps、5zero、原配对Normal；每run200，共600正式。旧S的600正式只读逐项配对一次。

沿用已授权工程方式：最多两个独立world_size=1训练，显式c10d localhost:0独立UUID；41/42先启动，空槽接43，三者自然完成后41/42/43串行评估。全部12次采集和3次评估共享environment_locked全局flock，不修改site-packages。失败保存证据，不盲恢复部分轮、补采、重评或重启controller。

复用既有设备快照与精确检查实现，每个新stage必须实际COPY_EQ、BRANCH_COPY_EQ、CHECK_EQ和自己的GPU准入。新的Q1准入额外断言实际缓存下候选更新等于原均匀保持实现；伪造raw q=.1时有效Q1保持必须与均匀保持逐位相等。不把旧工程验证标成当前新run已通过，也不把未来阶段套PASS。

当前硬件RTX3090；历史主S来自早前硬件，沿用用户已有硬件迁移授权。初始化和软件精度开关核相等，Task6全tensor比较记录真实结果；不得声称严格硬件受控单因素因果。数据开发集反复使用、预训练来源未闭合限制保持。不得根据结果选变体、扩seed或追加实验。

唯一候选root /data/mlr_ifa/experiment_workspaces/S_Q1_OBJECT41_42_43_20261108_v1，私有输出 /data/mlr_ifa/experiments/S_Q1_OBJECT41_42_43_20261108_v1。部署前核根不存在、GPU空闲、旧进程结束、完整基线/父源hash、实际容量、源码及冻结合同。空间不足不删除旧模型、不付费扩容。容量按现有实际对象大小、两并行atomic保存峰值及额外裕量核算，预留root1GiB/data2GiB。

仅自然controller.exit0且无自身进程后进行独立CPU12CP/3完整恢复/12缓存teacher/600初态noise/27300门控及环境互斥、预算验收；随后audit_final_local→compare_primary_s。最终包装仅静态准备，实际结果才能接受。任何包装失败保留证据后外置版本化修复，不重训重评；不预先继承OBJECT旧scale例外。

报告全部三seed的AUC/FWT/NBT/OldFinal/FinalAll、新旧任务、SF/FS、门控取舍和原始支持度分布，包括所有负结果。按AGENTS审核轻量代码/报告/hash/scalars/outcomes发布MLR，禁止模型、CP、state、缓存payload、RNG和凭据。报告PARTIAL不等于接受。
