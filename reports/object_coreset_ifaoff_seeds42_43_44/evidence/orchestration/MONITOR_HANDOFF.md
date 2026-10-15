# 当前有效任务：IFA关闭，原协议三seed（2026-10-14）

先读本文件、PROTOCOL.md、RUN_STATUS.json及launch_evidence，不恢复上一轮任务。
用户先取消新增初态9000复评和旧队列，随后明确授权使用原实验协议跑三seed
IFA关闭对照。取消证据保留在execution/20261014_coreset_validation/CANCELLATION.json。
旧9000复评只产生11条正式记录+20条A/A，禁止混入本轮。旧workflow.exit虽为0，
是SIGTERM时旧shell trap行为，绝不代表完成。旧自动化coreset-ifa已删除。

## 位置和正在执行的队列

SSH root@jq1.9gpu.com:11920，已有凭据仅内存/getpass，不写文件日志。
W=/data/mlr_ifa/experiment_workspaces/Coreset_validation_20261014
N=W/original_protocol
Q=/data/mlr_ifa/analyses/Coreset_IFAoff_original_protocol_20261014
每seed源码W/noifa/seed{s}/source，产物
/data/mlr_ifa/experiments/Coreset_IFAoff_OBJECT_seed{s}_20261014。
旧对照/data/mlr_ifa/experiments/Coreset_paper_OBJECT_seed{s}_20261013完全只读。

真实启动：`/data/mlr_ifa/env/bin/python N/launch.py`，exit0。
2026-10-14 12:53:40 CST后台queue PID3113，按42、43、44顺序运行。
queue.sh执行各seed_pipeline.sh，再各seed独立acceptance，最后compare.py。
新队列原始代码和共享preflight_noifa.py/accept_noifa_training.py冻结于
Q/orchestration_hashes.json，不能悄改。训练源码早已冻结，训练合同不重贴。
新评估合同EVALUATION_CONTRACT.json绑定新增resolved_eval和原评估源码哈希。

## 固定协议和预算

每seed Random/Coreset两组均官方权重、空buffer起步，任务6—9各100完整epoch、
batch10、105500步/arm，共2400epoch/633000步/24个epoch99最终CP。
仅loss从BC+0.1*IFA改为BC，ifa_weight0。保留IFA诊断计算但不参与梯度。
state=True/gamma0.3/warmup500逐step线性、原拼接shuffle/DistributedSampler、
每任务ceil0.5存储、原Coreset距离/标定/参考seed42全部不变。三个seed各自用原
导出state=True特征；modal/FEATURES字节不变，新的训练合同记录原来源及新源码。
首次任务6入库前，实际模型张量和初始化RNG跨arm及对应原+IFA应一致。
相应arm入库成员/顺序/原载荷与原+IFA逐项核对，不重抽凑一致。

原评估器coreset_exp.evaluate/eval_metric/model和各seed原源码哈希完全一致。
每阶段ep99对已见任务，原初态0–19，每cell20条，10lane两批env seed=[s,s+1]，
原全局torch动作RNG、原argmax分量→Normal、600步、5零动作、benchmark=True。
每arm200条，新三seed共1200；旧+IFA三seed1200作为已有对照，不重跑。
禁止使用独立Generator/new初态20–49/5重复/9000方案，不增加680次口径。

## 前置检查、恢复和完成

每seed先guard.py（训练与评估来源核验）、18软件测试、纯spawn导入检查，
再4次隔离真实GPU optimizer调用。4次验证IFA配对变化不影响更新和恢复一致；
不追加真实环境smoke。正式模型/RNG重新初始化，绝不接续预检状态。
seed42的18软件测试、纯spawn和4次隔离GPU更新均已PASS；12:54:33开始正式
Random训练，12:56快照任务6已完成17epoch、2669步。后续状态须读取真实日志，
不能把seed42检查声称为43/44已经通过。监控automation_id=ifa-seed，每两小时。

运行 `/data/mlr_ifa/env/bin/python N/monitor_snapshot.py` 保存轻量快照。
核对queue.pid/exit、实际进程/GPU/磁盘、各seed commands.log/pipeline.exit/
RUN_STATUS、epochs.json/recovery/latest.pth/storage、trials和attempts以及冻结哈希。
每次检查结束即返回，不一直守候。优先用最近epoch实际吞吐分别估计训练/评估。
历史+IFA六arm训练epoch合计4.58h、评估合计7.18h，只能作启动时参考；
新流程暂估12–16h，含预检/存储/加载/审计开销，后续用新实测修正。

遇工程堵塞，先确认相关进程退出和哈希一致，保留原失败exit/log，不盲重启
queue/seed_pipeline。训练只从本seed/arm完整epoch CP显式--resume恢复模型、
optimizer/scheduler/grads/modes/所有RNG/cursor；边界存储按预提交名单幂等复制。
不能重复计步、重复入库、混合arm或seed。之后只执行未完成队列段，留独立恢复
脚本、命令及退出证据。部分原评估拒绝盲重启；保留已完成条目及所有技术尝试，
正常失败不重试，技术恢复必须保持原全局RNG顺序和固定预算。协议无法保持则
BLOCKED，不悄改冻结源码/重贴合同。只读统计/验收工程错误可留证外置修复，不
重跑训练或环境。科学超参、模态、采样与精度不变，不新增seed/套件/付费资源。

每seed pipeline.exit0后原acceptance核查8CP真实加载、400条、无残留进程。
训练额外accept_noifa_training.py核对原模型和buffer配对；compare.py独立从
新旧原始记录复算全部12张矩阵与精确指标、三seed均值/样本SD、C−R和off−on交互。
**queue.exit0且无残留后**运行N/final_audit.py，完成冻结hash/旧输入保护/2400epoch/
633000step/24CP/1200新正式记录独立终验，才可总体COMPLETED。训练完不等于完成。
恢复时原exit保持原样，应外置适配验收记录真实恢复退出，不把失败改写成0。
下载轻量证据/报告，不下载checkpoint；不推送Git或上传权重。

报告全部seed/task，不选择性省略负结果。只有3个探索性训练seed，原评估全局
动作RNG未隔离；不得将收益变化称为普遍或显著因果。原预训练模态元数据unknown、
RTX3090与论文A100差异、调度补全等仍披露。Task6只读分析不反馈训练。
每两小时检查，阶段转换/实质进展/恢复/失败/完成才通知，无变化保持安静。
所有授权工作验收后删除本任务监控并停止，不恢复取消的9000方案或启动下一轮。
