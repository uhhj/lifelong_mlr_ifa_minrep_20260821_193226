# 单因素容量诊断：Temporal FFN 512→128

登记于2026-10-23，预算对照完成且独立验收后、新拟合之前。预算结果已发布并核验远端7c280d6c85ccbbeacf327b54cc7f30dd67a3d6ba。状态REGISTERED_NOT_IMPLEMENTED_NOT_DEPLOYED；不得解释为已经启动。沿用用户对小读出网络、一次一个因素、工程自修和发布的授权。

## 问题与唯一干预

200epoch已经让G的目标训练误差按折均值降至4.88/3.73mm，S降至6.92/5.21mm，但验证目标误差仍约19–23mm。这支持研究容量与跨演示泛化，而不是继续无限延长预算；不能证明唯一瓶颈。选择仅将两个Temporal TransformerEncoderLayer的dim_feedforward从512改为128，保留d_model128、heads4、层数2、激活、dropout0、norm与残差顺序及其余全部代码逻辑。

512维视觉/模态输入→128维投影不变，空间394token、3个查询及融合不变，语言/状态/模态/时间/固定空间位置编码不变，输出Head不变。P0的同帧frame Transformer FFN仍512；仅其后两个Temporal层FFN改128。G/S没有frame层，保持原结构。这样减少Temporal非线性中间容量，避免把输入投影或查询语义同时改变。不能把此对照解释为纯参数数量的普遍因果效应，改变的是指定模块的FFN宽度。

预期参数P0 483335、G/S各400775，均比原版减少197376；实施时由实际模型核数，若不一致先查工程原因，不能悄悄更改协议。模型初始化仍按原种子和构造规则；尺寸改变自然改变随机数消耗，不能宣称跨尺寸所有共同张量逐元素同初值。用于工程准入的512配置须与原模型构造顺序、state_dict和同输入输出精确一致，才能确认参数化改造没有附带变更；这仅是未训练模型隔离检查，不重新训练基线。

## 固定样本、预算与基线

P0/G/S × Task6/7 × 原五fold =30条新的独立轨迹，每条200epoch，总6000epoch、49800正式探针更新。保持原labels/native/私有Patch SHA、7训练/1验证/2测试演示、mask和样本身份、seed41+task*100+fold、FP32、有效batch128、P0micro128/G/Smicro16、AdamW3e-4 wd.01、scaledSmoothL1及所有前向干预。只改变Temporal FFN宽度，不同时改优化器、正则、dropout、训练预算、演示集合或表示。

50/100/200保存原验证联合位置分数的earliest strict best。全部最终诊断同时报告，不根据测试选择预算。200epoch范围内选优模型是唯一准入对象。原512宽度30条已验收预算轨迹作为只读基线，不重复训练或重评，绑定全部选择/指标/源/输入/最终验收SHA。新小模型没有相同结构50前缀复现要求，不得套用原预算前缀相等断言；仍严格绑定同原划分及训练样本人口。新实现自己的隔离GPU恢复、最新视觉依赖及长度1–8准入必须实际通过，P0/G/S共9隔离优化器调用另列；额外未训练基线一致性前向另列，0环境。

## 判定与限制

全部30轨迹独立验收后，固定G再S优先级，对同小模型P0-200应用原始A门：两任务的object_eef/object_goal均median≤10mm、P90≤25mm、原demo bootstrap改善CI下界>0、原动态前一目标优势>0和最新视觉替换优势>0。不得放宽、舍入过关或按中途测试选择模型/预算。512→128容量对照另报告训练/验证/测试的全部目标和配对demo探索性CI，不把训练损失或间隙缩小当闭环收益。

这是当前诊断序列最后一个自动容量取值。若仍失败，完整验收发布结论，不再自动轮试64/256、更多epoch或正则参数，B保持未启动；需要新的明确研究决策才能开新的科学扫描。正常科学负结果不重试。所有旧失败与已终止实验保留。多轮重复使用这些演示只能提供探索性证据，不能作为未触碰测试集的确认性发现。

只有严格A独立通过，才另版本实现物理效果验证/B0/B1/B2，绝不运行占位B。原用户B扩展门仍是B2最终两任务mean严格高于B0/B1且旧Task6不低于任一，然后全部组/干预扩seed42/43。

## 实施边界与监控

下一轮创建新local execution/20261023_readout_capacity和remote /data/mlr_ifa/experiment_workspaces/READOUT_CAPACITY_OBJECT41_20261023_v1。当前预算root及v2root永远只读。新contract绑定完整已验收预算闭合/源码/输入及新源，启动前核旧exit0/无自身残留/GPU空闲/互斥锁，原Patch原地只读复用，不复制或重导出，不下载权重/特征/rawstate/action。

完整epoch恢复需模型/optimizer/best/选择/cursor/全部RNG/grads/modes/requires_grad/父数据SHA，故障先核停与hash、保留证据，再版本化工程修复；不盲重启或覆盖冻结源。预计新增≤0.4GiB，data root f_bfree≥2GiB、system普通f_bavail≥1GiB；不删旧资产、不付费、不降低精度或预算。

当前预算实际fit总42.65min，仅作为新阶段量级参考；更小FFN的真实速度未测，不承诺线性加速。准备、准入、最终审计另计。先5分钟实施/准入续接，启动后按首个P0/G/S完整fit重估，训练检查5–15分钟，接近结束再缩短。更新同一个heartbeat patch，不建重复监控，不空轮询已闭合预算队列。

新路径须独立CPU strict检查所选CP/恢复绑定/完整epoch/验证best/分区身份和SHA，再本地误差标量复算、CI/门及全部负结果。不能把旧预算审计PASS移植为新运行PASS。每轮review轻量文件/secret/type/size/hash，经新目录publish.py定向publish_mlr_branch的MLR，非force push，核remoteSHA和本地receipt。
