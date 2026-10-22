# TAC-AUX8 OBJECT seed41 — 用户修订后的独立实验

2026-10-22。旧七步 TAC 已按用户要求中断；新实验从原官方 OBJECT 权重和空本 run buffer 开始，不接续旧 TAC 的297个完整epoch，不与其结果合并。

## 主路径不变

主策略输入始终为8步 `[8,4,512]`，原 Temporal Transformer 与部署 GMM Head 架构、原 BC 最后动作 a[t]、IFA、GMM argmax分量Normal采样及闭环推理保持原实现。主策略独立 state_dict 仍严格100键；H_aux不注册在主policy下，不出现在部署模型或推理代码中。架构不变不等于权重冻结：辅助梯度可更新共享 Temporal Transformer。

每阶段100epoch，四阶段400epoch/105500主优化器更新，4个ep99主模型、10cell/200正式episode/20vector。原RA seed41配额781/740/780/660、有序成员及载荷，BC+.1meanIFA、gamma.3、warmup500逐步、sampler0/concatshuffle、全局阶段RNG41、env41/42、初态0–19/600步/5zero不变。父RA与原v2不重训重评。不得调参追加seed。

## 因果下一动作辅助路径

x[t]=feature(obs[t-7:t])，H_aux(x[t])预测a[t+1]，F_phi(x[t],a_hat[t+1])预测物体从states[t+1]到states[t+2]的平移、SO3旋转及目标进展。原生数据obs[t]已等于动作a[t]之后的物理状态，因此不能用于预测a[t]的因果效果；用于下一动作合法。禁止末帧裁剪、零填充或重新编码图像。

H_aux为主GMM头的独立深拷贝（相同分量结构，独立参数），仅首阶段从共同官方初始化复制；以后加载本run上一阶段H_aux及其优化器状态。它不是部署头。为明确训练预测目标，在每个原生batch主更新之后，合法且非效果留出demo的下一动作NLL只训练H_aux，输入为当前共享Temporal的eval/no_grad输出，完全detach；独立AdamW lr1e-4、weight_decay.01，原grad_clip，无额外scheduler，至多每主batch一步。没有合法样本则不更新。此新超参数在正式运行前冻结，属于探索性设计，不称事前独立验证。教师损失不更新Temporal或部署GMM。所有头更新及样本曝光另记账，不冒充105500主步。

效果迁移使用H_aux最高权重分量的可重参数Normal采样；H_aux和F参数在该分支冻结，但保留对输入的梯度，故迁移只向共享Temporal传播。原.05效果权重及当前BC负点积关闭规则保留。F继续原v2训练期均值聚合、10辅助epoch、批至多64；整demo留出优于均值与动作置换、动作梯度有限非零、当前—旧效果图均通过才启用。Task7零支持时主更新严格回退RA；独立H_aux教师仍可学习，但不改变主策略。

原v2保护也在部署GMM上使用下一动作NLL。为避免保留同一矛盾，本版将该NLL的出口同步改成H_aux；只更新共享Temporal，不直接更新部署GMM。仍用同一RA效果组、至少3demo/12独立时刻、最佳已完成阶段参考、epsilon.05、dualrate.1、mu[0,1]，每完整epoch更新。参考必须绑定该阶段主policy及该阶段H_aux双checkpoint，禁止未来模型。此保护是辅助下一动作功能保护，不是部署执行分布保护，不宣称闭环遗忘保证。H_aux教师与保护仍可能竞争，必须报告实际超限和mu。

## 随机数、恢复、独立验收

H_aux教师使用独立种子940000000+seed*1000000+main_step，并保存恢复全部原随机状态和主模块train/eval状态。主更新只有一次；额外头更新不推进主scheduler。新真实GPU准入须验证：关闭迁移时主模型/optimizer/scheduler/grads/modes/RNG与RA精确相同；辅助作用的部署GMM梯度不存在、共享Temporal梯度非零；完整主+辅助恢复可重复；人为退化保护响应；BC冲突门；主导出100键且无辅助头。预检0环境，不计正式更新。

完整epoch恢复包含主状态、H_aux模型/AdamW/梯度/模式/requires_grad/步数、mu/bounds/效果与支持hash、全部RNG和游标。每阶段ep99独立保存训练用辅助头状态，正式评估仅加载原主policy。旧失败/中断只读保留，不修改旧合同。

Gate1复用独立B/v2合法下一动作科学图（仅语义相同的数据证据，不复用GPU通过或效果模型）：5561合法窗口，Task7=0，Task8/9=178/206支持；每阶段实际sidecar与原图逐项比较。先前已看过v1/GOAL/v2及支持数据，披露探索性。RSET/EFTR保留各自原冻结独立版本，排在新A之后，不混入新RSET零训练原型。
