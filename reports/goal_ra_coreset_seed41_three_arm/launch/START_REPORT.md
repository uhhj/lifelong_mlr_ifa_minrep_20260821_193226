# GOAL seed41 三组启动报告

状态：PARTIAL，证据时间 2026-10-19T16:29:20.874141+08:00。已启动后台有限队列，当前为新seed41特征导出；训练与闭环尚无完成结果，不是实验最终验收。

用户明确选择补齐 Random、Coreset、RA 三组。服务器没有可复用的 GOAL41 基线，已有OBJECT41不混用。沿用已完成GOAL42–44科学协议，先运行同seed配对Random/Coreset，再由RA绑定本次完成的Coreset父模型、特征与名单；旧实验只读。

三组共1200完整epoch、228300optimizer step、12个ep99最终checkpoint、600条正式闭环/60vector attempts。state=True、seed41、BC+0.1平均IFA、gamma0.7、batch10、500step warmup与逐step线性、原concat/shuffle及sampler0均保持。RA仅改变当前任务入库成员，固定半配额Random锚点与原距离最小损害交换。完整协议见PROTOCOL.md，源码diff与逐文件哈希见SOURCE_DIFF.json/source.patch。

| 检查 | 实际结果 |
|---|---|
| 基线软件测试 | 18项通过，exit0 |
| 真实资源与四份resolved config | PASS，4759窗口、三模态布尔True、seed41、GOAL gamma0.7 |
| 真实GPU预检 | torch1.13+CUDA11.6、RTX3090；状态token语义/GMM参数一致/4次隔离更新及恢复PASS |
| 环境接口 | 1个单环境零动作接口检查；0策略rollout，不混入正式结果 |
| RA软件、真实离线名单与GPU预检 | 尚未执行，基线完成后自动顺序执行 |
| 本地启动独立验证 | PASS_LAUNCH_ONLY_PARTIAL |

真实部署命令：`/data/mlr_ifa/env/bin/python /tmp/ra_goal41_install.py`，exit0；后台入口`bash /root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019/supervise_queue.sh`，启动PID9005。实际子命令和退出码保存在progress目录queue/commands.jsonl；当前queue/wrapper尚未退出。检查命令为`python execution/20261019_ra_coreset_goal_seed41/check_progress.py`（SSH凭据仅内存）。本地`verify_local_launch.py`退出0，核验代码/orchestration/config/modal/预检哈希。

预计从16:26启动起约5.5–7小时，约今晚22:00–23:30完成；依据前轮同RTX3090训练约41分钟/arm、评估约69分钟/arm，加导出与准入。当前尚无本轮训练吞吐，因此不是精确完成承诺。服务器后台将自动完成三组、汇总和退出验收；每2小时自动化goal-seed41检查并处理可恢复工程故障，临近完成可缩短。完整交付后取消。

产物：
- 基线：`/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019`
- RA：`/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019`
- 队列：`/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed41_three_arm_20261019`

保留最初本地草稿生成GBK解码失败exit1及UTF8修复说明，发生在部署前、0更新0环境，没有修改冻结执行合同或重复正式步骤。运行工程恢复遵守完整epoch/全部RNG及幂等入库；部分global RNG评估不盲重启。

本seed在看到42–44结果后授权，是探索性扩展。全局动作随机流未逐轨迹隔离、原路径排序会改变实际batch、IFA实际参考随记忆变化、预训练模态配置unknown及调度补全仍需披露。负结果不改变预算，不自动新增seed。

按约定只发布代码、报告及轻量证据，不上传权重、原动作、latent、buffer或凭据。发布进度不等于实验COMPLETED。
