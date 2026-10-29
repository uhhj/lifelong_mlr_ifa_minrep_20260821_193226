# 最新完整 S GOAL seed41/42/43 冻结协议

用户 2026-10-29 授权在已完成 OBJECT 后，再运行 GOAL 三 seed。仅最新完整 S，主对照为已有完整 C；不重训 C、不恢复已取消机制实验、不覆盖 OBJECT。

三个 seed 各从官方 GOAL 初始化开始，Task6→9 各100epoch。保持原 GOAL manifest、RA、mask、BC/expert/route 损失、顺序与学习率。Task6 单候选无保持；Task7–9 使用各 seed 自己的训练成功缓存 q=独立成功轨迹数/10。保持系数0.1、固定4槽分母、不重归一化。每10更新（轮尾短块）从同一完整状态计算 C 与 Q，当前训练64合法窗口 BC 与执行误差均不劣于 C、旧任务各8缓存窗口的支持加权保持误差严格优于 C 才选 Q，否则 C；两 epsilon 均0。完整模型/优化器/scheduler/grad/modes/requires/RNG/step 恢复，不能调整门控或用测试挑选。

每seed批次数100/139/185/337，接受更新76100、额外 C 候选66100、生产物理142200、门控6700块。三seed1200epoch、228300接受、426600生产物理、20100块。各阶段自身准入：Task6 4隔离optimizer/8额外autograd，Task7–9各29/11；三seed总273/123。训练方向诊断最多108额外autograd，全部另记。

每阶段10训练采集，固定初态20–29，成功轨迹缓存quota128，不补采；三seed共120训练采集。正式各stage评所有已见任务，初态0–19，原600步/5zero和配对Normal噪声；每seed200，三seed600正式。顺序train41→42→43→evaluate41→42→43。正常负结果不重试，不扩seed、扫参、追加实验。

复用已通过完整训练路径逐位验证的CPU复制优化：仅CUDA转CPU后去除重复clone，CPU源仍clone。原OBJECT诊断3914optimizer/18autograd仅为既有工程证据，不混入新GOAL成本，也不声称已经验证GOAL全部轨迹。新GOAL每阶段实际初始完整state复制与原函数逐位比对，0额外optimizer/environment，原自身GPU准入全部保留。

独立新root SUPPORT_GATE_FULL_S_GOAL41_42_43_20261029_v1，独立合同/锁/输出。容量上界8GiB，root普通保留1GiB、data保留2GiB。单次部署，不覆盖或重复训练已完成部分。故障先查进程/hash/完整恢复和已耗成本，不盲重启部分门控块或环境attempt。

最终只在controller.exit0且无自身进程后独立验收：12CP、3完整恢复state、12成功缓存teacherCPU重放、600初态/noise、全部门控标量及闭环。主报告与同seed已有完整C配对，核对初态/噪声/任务顺序。原S仅辅助；新S与C不等采集/不等算力，不将门控接受或训练代理当闭环收益，不把全方法结果归因于单一机制。

代码/轻量报告/哈希/标量/episode outcomes按根AGENTS审查后nonforce发布MLR，禁止CP、模型、状态、RNG、缓存和凭据。三个seed全部验收、解释及发布后才关闭本轮监控。
