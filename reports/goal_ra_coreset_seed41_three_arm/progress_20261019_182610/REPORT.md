# GOAL seed41 三组：进度检查

状态 **PARTIAL**，证据时间 2026-10-19T18:27:32.734566+08:00，不是最终验收。本次只读检查，没有新增优化器更新、环境执行、重启或协议修改。

| 组别 | 训练epoch | optimizer step | 最终ep99文件 | 正式闭环 | 状态 |
|---|---:|---:|---:|---:|---|
| Random | 400/400 | 76100/76100 | 4/4 | 80/200 | 原协议评估进行中 |
| Coreset | 400/400 | 76100/76100 | 4/4 | 0/200 | 等待Random评估结束 |
| RA-Coreset | 0/400 | 0/76100 | 0/4 | 0/200 | 等待基线完整验收后准入与训练 |

合计800/1200epoch、152200/228300step、80/600正式记录。初次进程快照18:26为70条；随后18:27只读证据采集时为80条，两份时间点均保留。8个vector完成、1个进行中，未见technical_exception。不得用未完成矩阵计算完整AUC/FWT/NBT。

Random与Coreset训练分别16:35–17:15、17:15–17:56完成，真实命令exit0；实测训练39.97及40.70分钟。Random评估17:56开始，GPU RTX3090利用率79%、显存9965MiB；系统盘约13.97GiB可用。queue/wrapper及baseline pipeline仍运行，无失败退出，RA未启动属于正常顺序依赖。

冻结源码、配置、输入、模态、4759特征及orchestration哈希重新核查PASS。8个ep99文件存在且哈希与训练完成凭证一致；最终完整加载验收尚未执行，不标已完成。Task6实际tensor一致，初始化模型/RNG哈希一致；四任务入库配额499/445/452/985、唯一目标文件、原特征载荷哈希和buffer最终manifest本地核验通过。本地逐episode与attempt一一核对PASS，无正常失败重试。

当前Random完整单元（只是进度，不是方法对比）：stage6/task6=14/20；stage7/task6=12/20、task7=20/20；stage8/task6=11/20。其余未完成，不补值，不据此改变预算或选checkpoint。

耗时：已完成8个vector平均224.81秒/10轨迹，剩余52个vector约195分钟，加RA训练约41分钟、准入/加载/汇总约15分钟，中心估计剩余4.18小时，约22:38结束。任务耗时不同，暂给 **22:00–23:30** 区间，不是确定承诺。继续每2小时检查，无需重启或用户操作。

真实本地命令及退出码：check_progress.py(0)、collect_readonly_progress.py progress_20261019_182610(0)、audit_progress.py progress_20261019_182610(0)。完整远程命令在COMMAND.json及READONLY_COMMAND.json。代码、报告与轻量证据按约定发布MLR；不含权重、latent、buffer载荷、原actions或凭据。仍须三组600条、12CP严格加载、退出与最终独立复算全部通过才COMPLETED。
