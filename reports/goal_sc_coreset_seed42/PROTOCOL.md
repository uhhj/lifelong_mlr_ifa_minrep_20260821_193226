# SC-Coreset GOAL seed42 preregistered execution

2026-10-18。用户指定新设计“监督构成保持核心集”。先seed42单组；用户已确认只有AUC严格高于同seed原Coreset、OldFinal不降低且工程验收全过，才扩展43/44。Random同时报告、不作为推进门槛。seed42原Coreset AUC=63/80，OldFinal=41/60；原Random AUC=259/320。不得看到结果后改变门槛。

唯一干预：当前任务完成后入库成员。原公开基点a721807f2f7fa7f9833dcc85c592d78e9accd00b的已验收GOAL共同实现为源；不重训/重评Random或Coreset。复用同seed原stateTrue特征，modal/feature合同保留原字节，新训练合同独立绑定，绝不重新给旧缓存贴标签。官方GOAL ep96权重起步，空buffer，四任务6—9各100完整epoch，batch10；BC+0.1平均IFA、gamma.7、AdamW1e-4、warmup500逐步线性、任务边界重建优化器而网络连续、原concat/shuffle/sampler0。实际样本997/889/904/1969，配额499/445/452/985，batch100/139/185/337，每组76100步400epoch。四ep99最终CP；200原协议评估、20向量组，初态0—19、10lane、env42/43、600步上限、5零动作、每阶段global RNG42、原argmax分量Normal采样，无独立轨迹Generator、680或9000口径。旧任务仅已存buffer，元数据另放，不变latent/监督。

身份：每条当前原训练窗口完整8×7 float32动作与真实HDF5示范所有含padding窗口逐字节匹配。必须唯一确定(demo_id,末端target,真实length,末端action)；文件号绝不当时间。Task9两条完整action匹配起点有两种可能，但两者同demo、同target151、同length152，故监督身份唯一，保留全部候选起点且各样本独立不去重。若任何任务不能唯一确定所需身份则BLOCKED，不猜、不删除样本。

分组：target/(T−1)四等时间区间，真实终点独立第五区间；整数4*target//(T−1)避免边界误差。夹爪按真实命令<0、==0、>0，不能解释为成功失败。组ID=time_bin*3+sign_index，sign_index=0负、1零、2正，仅有样本的联合组参加配额，空组0。保留padding原总体，不把窗口起点/8步内事件当末端监督。

每组floor(qNg/N)到ceil(qNg/N)。枚举合法整数分配，首先最小L1原组计数偏差即最小交换M，继而最小总绝对理想配额误差（整数分子精确比较），最后较小组ID优先获得ceil。此为用户并列规则的确定实现。S0在全部区间内则M0、成员与顺序完全不动。

S0必须由原greedy_kcenter_select实算，与原同seed入库有序ID一致。原两视觉token×8步、末端7维动作、0.5/0.5、全任务原始动作robust尺度、4096标定对、256参考、seed42、FP64、1e-12字典序平局不变。全候选集标定，不分组重新归一化。

每步在欠配额未选样本中取离当前集合最远者x；加入后从超配额组选择删除后全候选覆盖半径最小者y。使用原Distance全矩阵及精确nearest/second-nearest删除半径，平局按文件名固定ID/1e-12。执行恰好M次。用x替换y原选择列表槽位，其余元素相对顺序不动；无交换完全保留原列表。不得声称跨方法batch位置相同。所有非超额组的原成员保留，无重复/合成载荷，不选择未来任务/未入库旧数据。覆盖半径可能变差，须报告。

完整记录联合表、M、每次交换、原/新半径均值P95、示范数量分布、时间、载荷hash、固定quota和缓存。IFA公式与选对算法完全沿用，但每阶段按本轮实际buffer重算，报告与原Coreset差异，不冻结原配对。

测试：新增软件覆盖边界、零命令、罕见组、穷举取整最优、无变化原样保留、字典序平局与独立暴力交换对照；目标端全四任务真实payload/原S0对齐/配额/最小交换/删除半径检查；最多4次隔离真实GPU native BC+IFA优化器调用及边界恢复、0环境/0策略rollout。父空间模态语义与原评估代码字节未改，引用已验收父GPU证据，不冒称本轮重复验证。正式单独进程重新seed42，不接续smoke。准备期工程错误保存证据，可在冻结前改正；冻结后不得悄改源码或重贴合同。

从完整epoch恢复需模型optimizer/scheduler/grads/module modes/全部RNG/cursor；epoch99入库预提交manifest后幂等复制，禁止重抽/重复入库。部分global RNG评估拒绝盲重启，正常失败不重试，无法恢复精确顺序则BLOCKED。最后只有400epoch/76100step/4可加载CP/200记录20attempt/输入保护/精确矩阵指标和pipeline退出0无残留后独立验收PASS才工程完成。性能不通过正常报告停止。

用户已允许GOAL新产物系统盘，沿用/root/mlr_ifa专用目录；父基线/特征只读。无付费资源启动。每轮发布代码、报告、轻量证据至MLR，不上传权重、演示、latent、buffer载荷、绑定的原动作向量或凭据。正式训练启动后正常监控约2小时，临近结束可缩短；有效后才另建43/44合同与输入绑定。

解释边界：原预训练use_state元数据unknown，本轮state=True；论文未公开的调度、ep99细节沿用披露补全。按原全局动作RNG且未逐轨迹隔离，仅单seed探索性；约束保证离散联合监督表保持，不保证连续动作/状态分布、覆盖最优或闭环性能。新记忆可间接改变IFA。
