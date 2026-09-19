# 单seed完整逐样本QP实验

固定开发seed46，任务6→9，每任务100 epochs，共105,500次更新。
只有 SAMPLE_STEP_QP 一个训练分支。历史MLR结果只作跨环境的非配对背景，不能作为方法优越性的确认。

## 算法与数据

使用原混合GMM BC梯度、裁剪、AdamW及batch-step scheduler。AdamW仍按历史实现于任务边界重建，任务内部不重置。
QP作用于原生AdamW产生的实际参数位移；保留完整原生moments，只修正写入参数的位移。

每个阶段从当时原生buffer保留的每个旧任务中固定抽取32个窗口。
采样使用由实验名、seed、当前任务和旧任务确定的私有随机种子，不消耗训练RNG。
每步重新计算所有参考窗口的梯度，联合处理0/32/64/96条约束；不逐条投影，不改变损失权重。

原生buffer是数据独立的：使用未修改的构建函数提前重建各阶段名单，并逐项核对历史名单顺序及文件哈希。
训练只读取对应阶段的冻结名单，不访问未来任务。

## 执行与证据

服务器仓库：`/data/lifelong_mlr_ifa_minrep_20260912`

- 代码：`sample_qp_lifelong_v1/`
- 冻结协议、预检、队列和最终报告：`reports/sample_qp_lifelong_v1/`
- 训练事件、完成标记和评估：`runs/sample_qp_lifelong_v1/seed46/sample_qp/`
- 模型与完整恢复状态：`/root/sample_qp_lifelong_v1/`

CPU数学、抽样、被动记录和评估恢复测试全部通过后，队列才会核对资产、冻结协议并执行GPU预检。
GPU预检覆盖四种参考数量、完整状态一致性、epoch边界恢复、记录不干扰状态，以及真实历史64条约束实现回归。
全部通过后才会启动训练。出现数值或资源故障时保留证据并停止，不静默替换seed、放宽容差或改变算法。

每个epoch结束后保存模型、AdamW、scheduler、Python/NumPy/Torch CPU/CUDA RNG、训练模式、任务和epoch游标及已提交事件文件名单。
两个恢复槽轮换；各epoch日志不可覆盖。只能从完整提交的epoch边界显式恢复，未提交的尝试日志保留但不纳入主分析。

评估固定ep99，保留原生test.py调用顺序、GMM动作采样和success判定，共10个矩阵单元、每单元20 trials。
被动记录两轮共20条trial的成功标记和动作/终止信号哈希，不增加或选择trial。
评估重启时复用已完成单元及其RNG边界，每次调用保留独立原生日志；完整10单元核对后再发布一次主日志。

## 查询

登录服务器，进入仓库后：

```bash
cat reports/sample_qp_lifelong_v1/QUEUE_STATUS.json
cat reports/sample_qp_lifelong_v1/STATUS.json
tail -n 10 reports/sample_qp_lifelong_v1/train.log
```

初次启动入口是 `python -m sample_qp_lifelong_v1.queue`，拒绝重复队列。
训练中断时，先核对冻结输入、GPU身份、恢复文件哈希及中断原因，再使用 `python -m sample_qp_lifelong_v1.train --resume`。
该命令只续训，不自动启动评估；后续评估命令记录在 `QUEUE_PLAN.json`。

最终输出 `FINAL_REPORT.md`、`RESULTS.json`、`trials.csv`、`epochs.csv`、`per_task.csv`、`INDEPENDENT_AUDIT.json` 和 `DECISION.json`。
完成结论最多为 `SINGLE_SEED_PILOT_COMPLETE`，不能据此声称跨seed稳定性或相对MLR的受控提升。
