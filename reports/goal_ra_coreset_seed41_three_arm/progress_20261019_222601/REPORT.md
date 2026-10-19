# GOAL seed41 三组最终验收

**COMPLETED**。服务器于 2026-10-19T22:17:52.608108+08:00 通过最终队列验收；本地重新读取600条正式episode、60组vector尝试，并用Fraction独立复算全部指标，LOCAL_FINAL_ACCEPTANCE为PASS。三组各400完整epoch、76100步，共1200epoch/228300步；12个最终ep99 checkpoint严格加载凭证均为100个下游键。队列、wrapper和所有pipeline/supervisor退出0，无残留科学进程。无需继续等待，没有追加训练或评估。

以下均为百分数，NBT越低越好，其他列越高越好。精确分数保存在LOCAL_FINAL_ACCEPTANCE.json。

| 方法 | AUC | FWT | NBT | 最终旧任务均值 | 最终全部任务均值 |
|---|---:|---:|---:|---:|---:|
| Random + IFA | 69.062500 | 71.250000 | 2.777778 | 66.666667 | 66.250000 |
| Coreset + IFA | 71.875000 | 78.750000 | 12.222222 | 66.666667 | 67.500000 |
| RA-Coreset + IFA | 78.750000 | 82.500000 | 5.555556 | 73.333333 | 76.250000 |

RA相对Coreset：AUC +6.875pp，FWT +3.75pp，NBT -6.666667pp，最终旧任务均值 +6.666667pp，最终全部任务均值 +8.75pp。相对Random：AUC +9.6875pp，FWT +11.25pp，NBT +2.777778pp（遗忘指标更差），最终旧任务均值 +6.666667pp，最终全部任务均值 +10pp。不将AUC改善等同于所有任务和所有遗忘指标均改善。

每格20个共同编号初态，全部有效单元如下；未来任务未评估。

| 训练阶段 | 评估任务 | Random | Coreset | RA-Coreset |
|---:|---:|---:|---:|---:|
| 6 | 6 | 70 | 70 | 70 |
| 7 | 6 | 60 | 40 | 60 |
| 7 | 7 | 100 | 100 | 100 |
| 8 | 6 | 55 | 35 | 35 |
| 8 | 7 | 100 | 100 | 100 |
| 8 | 8 | 50 | 75 | 75 |
| 9 | 6 | 40 | 25 | 35 |
| 9 | 7 | 100 | 100 | 100 |
| 9 | 8 | 60 | 75 | 85 |
| 9 | 9 | 65 | 70 | 85 |

RA的最终Task6/7/8/9为35/100/85/85%。相比Coreset，分别+10/0/+10/+15pp；相比Random，分别-5/0/+25/+20pp。Task6遗忘仍未解决：从刚学完70%降到35%；阶段8的Task6也只有35%，低于Random的55%。Task7三组均为100%，不能从这个饱和值判断谁更好。

验收与可追溯证据：

- 新seed41官方GOAL权重起步、state=True、每任务100完整epoch/batch10、BC+0.1平均IFA/gamma0.7/warmup500逐step线性及原排序shuffle。原0–19初态、10lane、env41/42、600步与原GMM采样，阶段global RNG重置41不变。
- Task6三组实际模型张量相同，初始化模型/RNG一致；不是用含路径元数据的整checkpoint文件哈希冒充模型等价。
- RA任务6–9配额499/445/452/985、硬锚点249/222/226/492、交换119/107/112/244。实际有序成员与离线结果完全一致，源与buffer载荷哈希通过。
- 源码/配置/模态/父输入冻结合同通过；下载两份PACKAGE_MANIFEST逐文件验证。全部600条正式记录和60个vector start/complete一一匹配，无技术异常、无重试。预检与正式结果分开。
- 18项基线软件和8项RA软件已通过；各4次隔离GPU更新恢复此前已通过，本次不重跑。最终检查点实际加载由原流水线完成，本地下载仅元数据、日志和结果，不含权重、latent、buffer载荷、原actions或RNG二进制。

| 方法 | 正式训练分钟 | vector执行分钟之和 |
|---|---:|---:|
| Random + IFA | 39.97 | 70.32 |
| Coreset + IFA | 40.70 | 71.22 |
| RA-Coreset + IFA | 41.20 | 66.95 |

vector时长不含全部环境初始化、进程启动和审计开销，不能充当完整总耗时。完整命令/时序在各包commands与timings记录中。

本次真实命令（均退出0）：

1. `python execution/20261019_ra_coreset_goal_seed41/check_progress.py`
2. `python execution/20261019_ra_coreset_goal_seed41/collect_ra_progress.py progress_20261019_222601`
3. `python execution/20261019_ra_coreset_goal_seed41/collect_completion.py progress_20261019_222601`，仅调用远端元数据打包器和SFTP；每条远端命令及退出码见PACKAGE_DOWNLOAD_ACCEPTANCE.json。
4. `python execution/20261019_ra_coreset_goal_seed41/audit_completion.py progress_20261019_222601`，只读复算，无模型/环境调用。该入口复用旧进度审计时生成的PROGRESS_ACCEPTANCE仍写PARTIAL，最终状态以其后的LOCAL_FINAL_ACCEPTANCE和RUN_STATUS为准。

服务器产物分别保留在 `/root/mlr_ifa/experiments/Coreset_paper_GOAL_seed41_20261019` 和 `/root/mlr_ifa/experiments/RA_Coreset_GOAL_seed41_20261019`。本地轻量证据位于本报告同目录completion_baseline和completion_ra，完整发布名单与哈希见PUBLICATION_MANIFEST；推送SHA在工作区PUBLICATION_RECEIPT中记录。旧seed42–44结果未替换。

此seed是看到42–44结果后的探索性扩展；只有一个新增训练seed、每单元20次闭环，动作随机流没有逐轨迹隔离。成员改变会改变文件排序下的batch内容及IFA实际输入，不能把差异归因于单独的覆盖半径或纯锚点作用。官方pretrain state元数据仍unknown，调度/最终ep99规则包含工程补全，不能声称精确恢复作者轨迹。RA达到本轮报告门槛，不自动扩展其他seed/套件或启动下一轮；交付发布后删除goal-seed41定时任务。
