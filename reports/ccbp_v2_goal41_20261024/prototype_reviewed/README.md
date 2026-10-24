# CCBP-v2 候选损失原型

## 状态

仅完成本地 CPU 合成数据检查，没有加载真实机器人检查点、训练或环境评估。
不修改 GitHub 仓库、原始实验或服务器；CCBP-v1 的失败结果仍应保留。

依据：已验收 seed41 四组实验，A/B/C/D 的 Task8/Task9 为 65/80、80/80、80/85、70/80%。
这些单种子结果支持尝试结构修改，不证明本方案会优于 B 或 C。

## 修改

1. 删除 teacher_cost + 0.05 预算、合格集合、空集合回退和固定 margin。
2. 始终保留专家动作误差监督；按教师与训练示范的一致程度调整行为锚权重。
3. 回归和路由共用 J=(c+rho*w*d)/(1+rho*w)，其中 d 是到教师实际执行高斯的分组 W2 平方。
4. 使用按分量代价差加权的 softplus 路由；不使用分量固定编号或夹爪翻转风险标记。
5. 推理完全保持原 argmax-logit Normal 采样。推理时不能访问专家动作来筛选分量。

w=tau/(tau+c_teacher)。tau 为阶段开始时同任务全部合法训练回放教师 cost 的中位数，一次计算后固定。
该权重是“示范一致度”的代理，不是教师成功率或闭环安全概率。

## 文件

- losses_ccbp_v2.py：损失定义、训练回放中位数计算、quality/uniform/none 三种锚定模式。
- test_ccbp_v2.py：纯合成数值/梯度测试。
- synthetic_test_results.json：本次真实运行的测试结果。

运行测试：

```sh
python test_ccbp_v2.py
```

## 接入

在一个新的实验工作区增加模块；不要直接覆盖冻结的 v1 losses.py 或复用其验收合同。
原来的成本定义、训练尺度、标签 mask、Task8 合法窗口、样本流、BC/IFA 和部署路径保持不变。
从现有冻结教师缓存和全部合法训练回放标签计算 tau；不是从一个 batch 计算，也不使用测试轨迹。
在原 train.py 的辅助前向位置调用：

```python
from losses_ccbp_v2 import auxiliary_v2
extra, stats = auxiliary_v2(
    q, eligible_next_actions, teacher, scales,
    teacher_cost_median=frozen_tau,
    anchor_strength=1.0,
    routing_weight=0.1,
    anchor_mode="quality",
)
loss = bc + 0.1 * ifa + 0.1 * extra
```

q 和 teacher 均为 (means, stds, logits)，前三者形状 [N,K,7]、[N,K,7]、[N,K]。
scales 沿用 v1 的三组动作尺度。老师缓存必须与窗口身份、标签、阶段检查点绑定。
辅助学生前向为 eval 模式但保留梯度；照旧保存/恢复 mode 与 RNG。教师始终冻结。
函数返回的新统计键需要在新的报告脚本中处理；旧 v1 route/aux 准入测试不能原样作为 v2 验收。

## 首轮固定设置与比较

rho=1、routing_weight=0.1、外部辅助权重=0.1 为未优化初始设置。
不扫描 epsilon/margin；本方案已经删除二者。相同数值权重不代表相同梯度强度，应记录分项梯度/耗时。

主要比较：B、C、v2-none（只更换路由）、v2-quality（完整）。
若要独立声称教师质量权重有效，应另加 v2-uniform，而不能只从 B/C 对比得出该结论。
保持同起点、同任务数据及更新数、配对初态/噪声；新版本用新的协议，不拼接旧 AUC。
seed41 仅作探索，确认实验使用未用于本版选型的种子/任务顺序/初态。
以旧任务保持和新任务学习共同判断，离线 cost、路由 regret 或尾部降低均不能替代闭环。

## 限制

所有新增项都是软正则，不保证 argmax 始终选取最低 J，更不保证任务成功。
训练示范可能存在多种正确动作，质量权重仍可能判断失准或阻碍有益偏离。
一期不引入风险网络、额外闭环训练数据、硬部署过滤器或多候选优化器。
本方法不解决原读出可见性问题，也不证明训练回放之外的状态覆盖已经充分。
