# RA-Coreset GOAL 43/44：进度与seed43独立复算

快照时间：2026-10-19T14:18:37.233852+08:00。整体仍为PARTIAL；本次只读核查，没有启动训练、追加评估或重复预检。各package采集稍晚于快照，训练计数以下注明来源。

seed43已完成400epoch、76100step、4个严格加载的ep99模型、200条正式记录及20个完整vector attempts。训练/evaluate/audit/pipeline/supervisor均退出0，服务器PROCESS_EXIT_ACCEPTANCE及本地独立复算PASS；0技术异常。seed44已自动启动，准入与4次隔离GPU更新恢复PASS，0环境预检；正式评估尚未开始。

seed44包内epoch证据为65/400、6500/76100step；初始快照为46epoch。Task6尚未完成时，最终模型tensor对齐检查为PENDING，不是失败；初始化和最终tensor将由冻结训练入口强制核对。当前无队列失败、GPU正常工作。

## seed43指标（百分数，精确值见PROGRESS_ACCEPTANCE.json）

| 方法 | AUC | FWT | NBT | OldFinal | FinalAll |
|---|---:|---:|---:|---:|---:|
| random | 79.062500 | 76.250000 | -6.666667 | 88.333333 | 83.750000 |
| coreset | 78.437500 | 83.750000 | 11.666667 | 73.333333 | 76.250000 |
| ra_coreset | 84.687500 | 83.750000 | -3.333333 | 85.000000 | 86.250000 |

RA相对Coreset：AUC +6.25pp，FWT不变，NBT降低15pp，OldFinal +11.666667pp，FinalAll +10pp。相对Random：AUC +5.625pp，但OldFinal低3.333333pp。不得将这些结果写成每项指标均优于Random。

## seed43全部有效矩阵单元（每格20条）

| stage | task | Random | Coreset | RA |
|---:|---:|---:|---:|---:|
| 6 | 6 | 80 | 80 | 80 |
| 7 | 6 | 80 | 55 | 80 |
| 7 | 7 | 100 | 100 | 100 |
| 8 | 6 | 80 | 70 | 70 |
| 8 | 7 | 100 | 100 | 100 |
| 8 | 8 | 55 | 70 | 65 |
| 9 | 6 | 95 | 70 | 75 |
| 9 | 7 | 100 | 100 | 100 |
| 9 | 8 | 70 | 50 | 80 |
| 9 | 9 | 70 | 85 | 90 |

Task8刚学完RA65%，低于Coreset70%；最终则80%高于50%。Task6最终75%高于Coreset70%，但低于Random95%。这些负向或未超越结果全部保留。Task7各已见阶段保持100%，Task9 RA90%/Coreset85%/Random70%。

## 只读一致性与时间估计

seed43实际有序入库与冻结offline完全一致，四任务交换121/106/100/249，配额499/445/452/985，锚点全部满足，累积2381载荷哈希链一致；Task6实际模型tensor和初始化全RNG同父Coreset。两seed冻结源码/config/modal/input/feature/orchestration校验PASS，全部下载文件通过PACKAGE_MANIFEST。父结果逐文件对照新合同哈希；逐episode、vector账目、CP身份、初态身份及精确指标独立复算PASS。

seed43实际训练40.06分钟，评估69.30分钟。按44最近20个epoch与43同位置实测时间比0.9997，估计44剩余训练约30.7分钟；评估参考约69.3分钟。含验收余量，快照时剩余约90–122分钟，预计北京时间15:50–16:25完成。可能随真实吞吐变化，未提前最终验收。

本次真实读取命令为check_progress.py --package（exit0）；其远端monitor_snapshot、verify及package_light命令/exit0详见COMMANDS.json。本地audit_progress.py progress_20261019_141825（exit0），仅复算元数据与结果；没有模型加载或环境调用。保持两小时检查，不持续守候、不重复启动。

等待44完整结束、queue/wrapper退出0及三seed最终验收后再汇总三seed均值/样本SD。当前不可根据两个已完成seed作三seed结论。该扩展在看到42后授权；global动作RNG未隔离、batch排序、IFA间接变化、pretrain state unknown及调度补全边界不变。
