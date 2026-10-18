# 指令零响应残差 seed40/41 扩展合同

2026-10-17用户明确要求补跑seed40、41，保持刚完成的Coreset＋IFA＋指令零响应残差方案不变，并已明确授权两组新产物放系统盘。此授权覆盖原seed42仅单seed和43/44两seed范围，不重训或重评任何基线。此前定时监控授权沿用于本扩展：正常每两小时检查，临近完成可缩短，全部验收后删除。

每seed官方OBJECT六任务预训练起点，独立空buffer，复用对应seed已验明的state=True特征与Coreset成员/顺序/载荷/配额。每任务6–9各100完整epoch、batch10、105500步/seed、4个ep99 checkpoint、原协议200正式episode/20vector attempts。新增合计800epoch/211000step/8CP/400episodes；加已完成42–44为2000epoch/527500step/20CP/1000episodes。

基线40：/data/mlr_ifa/experiments/Seeds40_41_three_arm_20261016/seed40/rc；基线41：/root/mlr_ifa_new_experiments/Seeds40_41_three_arm_20261016/seed41/rc。只读coreset组，AUC分别81.14583333333333、70.10416666666667；实际指标从各200条记录精确复算。父源码/data/mlr_ifa/experiment_workspaces/Seeds40_41_three_arm_20261016/seed{s}/rc/source，以其CONTRACT逐文件哈希为准；原始上游1ed6e0dd25cb5748ee5e887444619483bd9bdc31不等于声明本轮变更已提交Git。

新队列Q=/root/mlr_ifa/experiment_workspaces/ZR_coreset_seeds40_41_20261017，源码Q/seed{s}/source；新输出/root/mlr_ifa/experiments/ZR_coreset_OBJECT_seed{s}_20261017。旧实验所有模型、缓存、记录保持原位只读。不得给旧特征重贴合同、删除有效旧数据或CPU回退。

BC+.1平均IFA、gamma.3、warmup500逐step线性、原拼接shuffle、sampler默认0、Coreset标定42不变。训练/数据打乱/阶段全局评估RNG用对应40或41，env[s,s+1]；残差局部初始化固定42+task。共享T/head训练，当前rank8UV训练，旧UV冻结且保留z梯度；FP64门、所有已生通道前向，无TaskID路由、不夹断门、不改阈值。IFA对与reference必须与对应父组一致。官方起点公共tensor及RNG核对真实父INITIAL_STATE，不强求训练后的Task6等于无残差基线。

每seed按顺序：准入（真实权重/观测，零optimizer零环境）、prepare、5软件指标测试、最多4隔离optimizer调用含一次恢复重演、原生get_action零环境检查、freeze、训练、独立进程评估、finalize/严格加载4CP、独立进程退出验收。已通过步骤不重复；预检更新绝不进入正式训练，正式进程重新初始化RNG。不重跑200步原型或已完成seed42/43/44预检，不增加闭环smoke。

每10epoch和各阶段最终全旧buffer结构检查，44次/seed；旧门<=1e-6、当前门误差<=1e-5、旧raw累计贡献<=1e-6、GMM容差及选中分量一致按原spec。失败停止，不修门或放宽阈值。原评估初态0–19、20/cell、10lane、600步、原argmax GMM分量后Normal采样、benchmark=True、阶段全局动作RNG；不引入独立Generator/新初态/680口径。

恢复：先核实进程退出、合同一致，保留失败；仅从本seed完整epoch恢复模型/optimizer/scheduler/grads/modes/allRNG/branches/cursor；按预提交清单幂等入库。不盲重启队列，不覆盖exit/log，不重复训练步/入库。部分评估须保留全部尝试及原global RNG顺序，正常失败不重试，无法保证则BLOCKED。冻结科学源不可悄改或重贴合同。只读汇总故障可外置留证修复，不重跑环境。

全部完成需两个seed各400epoch105500step4严格加载CP200正式记录20attempt44结构检查、buffer/IFA身份和input/source/config hash通过、pipeline/supervisor退出0无残留，queue退出0后运行final_queue_audit.py，再下载轻量证据本地独立复算。报告五seed全部任务矩阵及AUC/FWT/NBT/OldFinal/FinalAll、逐seed配对差、均值和样本SD(ddof1)、成本。主五seed表中seed43用第二次完整repeat1基线，首次基线仍作完整辅助敏感性表；原已发布五seed结果不覆盖。这是事后调整，不能隐去不利比较。

预定筛选条件不变：AUC严格上升、FWT不降、NBT严格下降、OldFinal不降；负结果不影响工程验收，不能据结果追加seed或调参。新增40/41是在看到前三seed结果后授权的探索性扩展。父40/41基线历史4090、当前实测3090需披露；官方pretrain state元数据unknown，调度补全，额外参数，共享遗忘及未隔离动作RNG限制保留。只记录观察到的效果，不声明普遍因果或统计显著。

不push，不上传/下载权重数据，不启动新付费资源。正式启动后正常检查保存快照即结束，不持续守候。全部授权工作验收后停止并删除定时监控，不启动下一轮。
