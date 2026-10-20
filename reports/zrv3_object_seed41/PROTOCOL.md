# ZR-v3低成本验证，OBJECT seed41

用户附件为本轮科学方案。先完整ZR-v3，复用20261019已验收RA＋IFA父阶段6–9模型、官方T0与同一批state=True特征/RA记忆；旧实验只读。seed41为看到ZR-v2下降后选定的诊断seed，非确认性独立检验。

仅新增执行分量Gaussian NLL和每状态执行分布KL投影。父网络/官方T0/语言门冻结；rank8 UV、FP64双向门、原0.1幅度界、梯度归属屏蔽及grad=None语义不变。新轨迹从task6零输出UV起步，不继承ZR-v2 UV。各阶段新分支局部初始化seed41+task；旧UV仅继承本方法上一阶段。尺度、logit与argmax分量保持同输入下的父值，其他四分量均值不变。

epsilon固定0.5 nat。训练与原生get_action都调用execution_core.calibrate_distribution。以FP64计算闭式KL与投影系数，最终均值按父dtype送Normal；若舍入造成实际KL超过预算，只沿父均值方向做最多16次nextafter ULP修正，保留解析投影梯度，预算不放宽。最终实际分布KL须≤0.5+1e-12。超界停止，不夹断动作或调整预算。v3-A仅关闭投影，仍采用执行Gaussian NLL和选中分量修正；其他一致。

每阶段10完整epoch、batch10、sampler0，1562/1480/1560/1319当前窗口、RA旧记忆配额781/740/780/660。每组40epoch/10550optimizer调用/105240样本曝光；AdamW只更新UV，lr1e-4、weight_decay0.1、warmup500及原逐步调度不变。缓存成员、顺序、文件载荷、原排序/concat/shuffle不变。父公共训练BC+IFA已完成，不重跑；新增校准NLL不另加IFA。

只评校准ep9，原0–19初态、每单元20条、10lane/env41,42、600步、5零动作、每stage重置global RNG41、原argmax分量Normal分布。每组200正式/20vector attempts；不重评基线或ZR-v2，不隔离轨迹RNG，不增加680或9000预算。

新预检：继承软件性质测试＋9项核心数值测试（CPU和CUDA）；每个新变体最多4次隔离实际optimizer调用及恢复重放、真实图像history1/4/8原生接口比较，0环境执行。正式RNG独立重置，smoke UV不继承。通过预检后冻结代码/输入合同。结构检查stage_start、epoch0、epoch9、stage_end，全扫描各阶段实际dataset；累计16次。检查零残差精确还原、非选中均值不变、选择身份与训练/推理一致、原门约束、实际KL上界和父/T0不变。每次在线前向也核验KL，保存每cell最大值和投影计数。

完整400公共epoch来自已有父资产，不冒充新训练。本轮新增40校准epoch/10550step、4严格加载CP、200真实记录、20完整attempt、源输入保护/全部数量与结构、pipeline/退出验收通过，才完成一个变体。预先冻结正结果条件：相比RA父AUC严格提高、FWT与OldFinal均不降低、全部工程PASS。绝对门槛为AUC>701/960、FWT≥3/4、OldFinal≥47/60。NBT仅报告。

完整v3达门槛才从新零UV运行v3-A，最多新增同预算一组；若v3不达门槛则停止。v3-A用于区分目标对齐与信任域贡献，不按其结果调参。最多两组、21100正式optimizer调用、400闭环；不自动扩其他seed或GOAL。

资源均在已有服务器数据盘，GPU必须真实可用，不申请新付费资源。旧资产只读。工程失败保留日志/exit；训练只显式从完整epoch恢复全部状态，幂等入库；部分global-RNG评估禁止盲重启，不能保持顺序与预算则BLOCKED。冻结源码不悄改或重贴合同。用户已授权的只读审计修复可外置实施，不能改变科学协议。

每轮按AGENTS推送代码、报告和审核后的轻量证据至MLR，不发布模型/RNG二进制、latent、buffer或动作载荷。正常后台运行后结束检查不持续守候；最终仅基于真实记录解释，单seed通过不宣称普遍有效或信任域已证明有效。
