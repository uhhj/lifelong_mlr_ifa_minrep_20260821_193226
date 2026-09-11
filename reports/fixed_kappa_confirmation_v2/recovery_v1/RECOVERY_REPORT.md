# 通信隔离修复与忠实恢复核验

核验时间：2026-09-11 21:03（北京时间）。状态：恢复闸门通过，正式剩余评估已实际启动；全实验尚未完成。

## 故障与修复

原队列于 20:18:20 因通信基础设施故障停止。PyTorch 1.13 的两个并存 `--standalone` 启动器使用相同的 `localhost:29400` C10d store；MLR 启动器退出后，仍等待的 soft 启动器失去连接。原始失败记录及日志保留。

恢复入口改用独立 `localhost:0` 临时端口和不同 `rdzv_id`，保持单 GPU、world_size=1。原有科学实现、κ=0.5、数据、缓存、任务顺序、训练及评估参数均保持冻结，原文件哈希已复核。恢复代码单列于 `confirmation_experiment/recovery/`。

## 已通过的验证

| 验证 | 证据与结果 |
| --- | --- |
| 已完成资产保留 | 3,822 个原文件、3,317,335,436 字节已登记哈希；启动前及重建后、恢复评估前均通过复核。已有训练及评估证据不被重放结果替换。 |
| 真实通信隔离 | 两个真实 NCCL 进程分用端口 48047 / 56207；首进程退出后，另一进程继续执行 collective 成功。 |
| 评估恢复边界 | 使用原 `test.py` 和真实 MLR checkpoint，在零机器人 trial 的完整性测试中，对比连续调用与复用首格路径：其余九个调用前的模型、Python/NumPy/CPU/CUDA RNG、配置、模块模式、后端设置及任务顺序一致。 |
| 新增单元测试 | 5 项通过，覆盖启动器边界、原训练配方、已完成评估行保留、首格复用及不完整/不一致日志拒绝写入。 |
| soft task6 状态重建 | 原 15,700 步 loss 序列哈希、模型、AdamW、scheduler、RNG、step 计数全部精确一致；781 个回放文件逐文件一致。 |
| 原任务数据与缓存 | 官方 checkpoint、原数据及三个 seed 的 feature cache 按原冻结清单复核通过。 |

## 保留和恢复范围

- **seed45 MLR**：task6–9 的 400 epochs、105,500 次更新和四个 ep99 checkpoint 全部保留，不重新训练。
- **seed45 两个 arm 的 task6 评估**：各自已完成的 20-trial 真实返回值继续使用，不重新测量、不择优。
- **seed45 soft**：原 task6 checkpoint 和 15,700 条事件保留。由于异常退出丢失活状态，在独立审计目录忠实重放 task6，仅用于恢复状态；通过全部相等闸门后保存 `soft45_boundary.pt`，由同一活进程等待并继续 task7–9。重建计算不作为额外 seed、arm 或正式成绩。
- **异常终止的未完成评估单元**：不能排除 task7 仿真已开始；该未完成单元按原协议忠实重放。失败尝试日志保留，正式主矩阵仍每格 20 trials，不根据结果增加 trial。
- **seed46/47**：继续原配对流程，task6 参数、训练轨迹和真实评估闸门仍须通过。

恢复评估先写独立 segment，完整后核验已有首格并只追加缺失单元。恢复后的 soft 日志只追加后续任务，旧字节前缀受保护。最终汇总前还会再次执行完整结果保留检查。

## 已实际启动

- 恢复队列 PID：`530796`。
- 当前阶段：seed45 MLR 的 stage7 起剩余九格评估。
- 评估进程 PID：`533202`；启动时间 21:02:10。
- soft 状态重建 worker PID：`532088`，处于已验证的 task6 结束边界，等待 MLR 剩余评估结束后自动继续 task7。
- 原 task6 已完成结果已写入独立恢复 segment，作为复用记录；没有为它新增机器人 trials。
- 当前没有新的 `recovery_v1/FAILED.json`。

后续顺序固定为：seed45 MLR 剩余评估 → soft 剩余训练与评估 → seed46 配对流程 → seed47 配对流程 → 汇总与发布。尚未完成的 task7 采样闸门、后续训练、评估和最终保存检查由队列自动执行，不能将本次恢复通过等同于它们已经完成。

## 版本与位置

- Branch：`research/fixed-kappa-confirmation-v2`
- Starting SHA：`b7a14ff715f6a3e745a43f3aea887a1a5f6ec50b`
- 原方法实现 SHA：`3284c4213c7a7694ee906aed5505aa06bb1f3c44`
- 通信修复实现 SHA：`951be98953f0d5addaf1f4de7fc6a62ae08d04bf`
- 恢复启动前审计提交：`907b33d03efae23405c54263b3b09bbb75e501a4`

服务器根目录：`/media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep`。

- 正式结果：`runs/fixed_kappa_confirmation_v2/formal/`
- 恢复状态与启动日志：`runs/fixed_kappa_confirmation_v2/recovery_v1/`
- 恢复审计：`reports/fixed_kappa_confirmation_v2/recovery_v1/`
- 本地验证文件：本报告旁的 `verified_audit/` 及 `verified_recovery_audit.zip`。

查询当前恢复进度：

```bash
cd /media/dm/Elements/wzp2026/lifelong_mlr_ifa_minrep
/home/dm/miniconda3/envs/wzp_mlr_tmp/bin/python -m confirmation_experiment.recovery.progress
```

原根目录 `FAILED.json` 作为历史证据继续存在，应使用上述恢复进度入口。服务器训练不依赖本地窗口；最终自动 push 由本地发布中继执行，需要本机在线。
