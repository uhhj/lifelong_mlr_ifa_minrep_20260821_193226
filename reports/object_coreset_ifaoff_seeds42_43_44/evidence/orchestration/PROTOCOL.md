# IFA关闭：原协议三seed对照

2026-10-14用户取消新增初态9000条独立评估及其旧队列，随后明确授权：
“可以用原实验协议再把IFA关闭跑一次3seed实验”。本合同替代新评估方案，
不改变已经准备但尚未开始训练的IFA-off源码、训练配置及训练合同。

## 唯一训练干预

LIBERO-OBJECT，seed42/43/44，每seed Random MLR与Coreset MLR各一组。
各组均从同一官方六任务预训练权重和独立空buffer开始，任务6–9各100次
完整遍历、batch10、105500 optimizer steps。共2400epoch/633000step。
网络跨任务继承、任务边界重建原optimizer/scheduler，warmup500逐step线性，
状态模态True、gamma0.3、原拼接shuffle和ceil(0.5*N)存储预算不变。
特征复用对应seed原始state=True导出文件，原modal/feature合同原样保留，
训练源码变化另记新合同。Coreset标定/参考seed42与DistributedSampler默认seed0
保持原设定。禁止借助Task6只读分析改变选样。

原损失BC+0.1*IFA，本轮loss=BC，配置ifa_weight=0.0。保留IFA配对计算及
前向诊断流程以避免其他训练逻辑变化，但它不进入backward。4次隔离真实GPU
更新/seed验证不同配对不影响实际更新以及完整恢复一致性。正式训练重新初始化
模型和RNG。任务6首次入库前实际模型张量应跨arm及对应原模型一致。
每arm入库成员/顺序/原载荷应与对应原+IFA arm一致；不重抽来凑一致。

## 原评估协议

直接复用每seed原coreset_exp.evaluate和eval_metric，字节哈希核对。
仅四个ep99最终模型、各阶段已见增量任务；每cell原初态0–19各一次。
原10lane、两批env seed=[s,s+1]、最多600步、五步零动作、原全局动作RNG，
原argmax GMM分量后Normal采样；cudnn.benchmark=True沿用原评估。
每组200条，每seed400条，新三seed共1200条。**不使用独立轨迹Generator、
新初态20–49、5次重复或新增9000条评估。** 不追加680条次口径。
原+IFA三seed共1200条作为已有对照，不重新评估或训练。

## 工程与验收

新队列独立于已取消的run_all.sh，旧队列绝不恢复。原训练合同保持不变，
新增评估配置和原评估源码通过EVALUATION_CONTRACT.json单独绑定。
每seed先18项软件测试、4次真实GPU隔离更新和纯spawn导入检查，再正式训练。
复用已经验证的原评估器，不新增真实A/A或其他rollout预算。
训练从本seed本arm完整epoch checkpoint恢复，保留原失败、全部RNG、模式、
梯度、optimizer/scheduler/cursor与存储预提交清单，不重复追加或计步。
部分原全局RNG评估不得盲重启；正常失败不重试，技术恢复需核实原RNG顺序。

验收2400epoch/633000step/24可加载最终CP、1200条正式闭环、原配置/初态/seed
匹配、完整尝试账目、全部六张矩阵及精确AUC/FWT/NBT、三seed均值与样本SD、
各seed C−R与( C−R )IFA-off−( C−R )IFA-on。新队列exit0且无残留进程后独立
复算再标COMPLETED。旧9000评估仅11条已提交+20条A/A，标CANCELLED_BY_USER，
不得混入本轮或作为正式新对照。旧取消时shell trap写入exit0不代表完成。

仅3个探索性seed，不宣称显著性或普遍因果；原全局RNG会有模型轨迹依赖，
本轮未消除该评估噪声。保留硬件、预训练use_state元数据unknown、调度补全
等历史边界。旧成果只读，不推送、不上传下载权重、不申请新付费资源。
