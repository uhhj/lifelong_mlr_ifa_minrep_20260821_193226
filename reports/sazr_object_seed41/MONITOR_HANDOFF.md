# SAZR OBJECT seed41 Task6

最新15:38 CST：正式anchor进程41072正在0–600步的首个10lane批次，尚未提交10条结果；GPU3090/8485MiB/83%，哈希PASS。supervisor41065、pipeline41071均在运行。见progress_20261020_153745/LOCAL_ACCEPTANCE.json（PASS_PARTIAL，19个本地部署文件与冻结源一致）。新自动化`sazr-object-seed41-task6`每15分钟检查，预计15:37启动后约25–35分钟完成（约16:02–16:12，非保证）。无堵塞或追加预算。

先完整读USER_PROTOCOL.txt、PROTOCOL.md、RUN_STATUS.json和最新progress。仅首轮Task6原型，不扩后续任务/seed。父RA＋IFA seed41 stage6冻结，只读已有资产；50条上限=锚定20–29十条父＋验证30–39父/候选各十条＋原测试0–19二十条最终部署策略。开关FS>=2且SF=0，未通过精确回退父head；正式测试不能选择开关。零锚定成功则保留10条结果BLOCKED，不追加预算。

本地根execution/20261020_sazr_object_seed41。远端W=/data/mlr_ifa/experiment_workspaces/SAZR_OBJECT_RA_seed41_task6_20261020；O=/data/mlr_ifa/experiments/SAZR_OBJECT_RA_seed41_task6_20261020。已有SSH授权root@jq1.9gpu.com:11920，密码仅getpass内存不写文件日志。数据盘25GiB可用，新资产在数据盘。旧ZR-v3等已完成不恢复。

已通过CPU/CUDA各8软件性质、40初态真实载荷互异、全部Task6缓存及实际规范语言路由、恰好4次隔离真实optimizer恢复、真实图片history1/4/8原生/latent参数与采样、拒绝分支bitwise回退；独立噪声适配器history1/4/8真实GPU零更新检查PASS，0环境。禁止重复预检或增加rollout。CONTRACT冻结9109源/父输入文件，后续ANCHOR_CONTRACT和DECISION各以派生hash链冻结。

科学方案：当前父T6特征，不用T0；成功轨迹均匀选最多32历史、FP64 SVD span、z−QQᵀz，无bias rank8UV；新U0/V局部47，双向语言门；仅所选Gaussian均值加.1tanh残差，直接末端7维平均L1，无v3KL。1562窗口/10epoch/batch10/1570step/15620曝光、AdamW及warmup500逐step与原校准相同。父和Q冻结。训练当前任务全数据，无旧buffer变动。每epoch锚点/非选中/父hash检查；阈值相对投影1e-10、实际均值/残差1e-6固定不可放宽。

验证两策略使用初态30–39、10lane/env41、600step5zero、各初态CPU torch.Generator(41000000+index)产生600x7固定Normal表，按轨迹与step索引共享值，不共享消耗状态。正式原global RNG41/env41,42/argmax Normal，0–19二十条。未知/基任务/歧义语言与关闭分支直接绕过残差，不依靠近零响应冒称bitwise。原60%历史父结果仅报告参照。

supervisor已独立后台启动，初始PID41065（仅历史必须查实际），有限序列anchor→anchors→train→validation_parent→validation_candidate→decide→formal→audit。各phase仅启动一次，无盲重试。pipeline/supervisor返回0后PROCESS_EXIT_ACCEPTANCE与COMPLETED。预检phase日志/exit在O；首次CPU软件日志OK但外层shell输出包含cat，避免把外层exit冒充单独记录的CPU测试exit，日志8项OK为通过证据；其他真实退出码均已保存。

运行本地check_progress.py只读hash/进程GPU/账目并下载allowlist元数据，凭据getpass；再audit_local.py progress目录本地核验。RUN_STATUS需结合实际process和phase文件（环境phase没有连续状态更新）。正常运行保存时间与PARTIAL，基于实际vector耗时ETA后结束不持续守候。启动估计25–35分钟（5vector每约5min），无成功锚定会提前BLOCKED。自动检查30分钟合适，不能重启已运行的队列。

允许恢复工程堵塞：先确认进程退出，保留原失败日志/exit和原合同，验证hash；完整epoch可显式train --resume恢复bank/optimizer/scheduler/grads/modes/allRNG/cursor，不能重复已计步。部分环境不能盲重启或重试正常失败，无法保证原预算/RNG/初态顺序则BLOCKED。冻结科学源不可悄改或重贴合同；只读统计错误可外置留证修复。入口原型未写自动恢复，需审计后外置恢复流程，不重启pipeline。

最终验收：50真实记录/5vector/0未解释技术尝试，1570steps/10epoch/1ep9CP严格加载，11结构扫描，所有真实success锚点/基rank<=32与非零修正证据、验证同初态noise hash和SS/SF/FS/FF，开关与正式策略一致，基/未知/拒绝实际父head分布采样bitwise回退，冻结输入保护，所有exit0且无残留；本地独立audit_local PASS_FINAL后再标完成。正式正负均完成，不自动扩展。若零成功锚点，不冒充完成50条，报告BLOCKED并停止。

每轮使用publish.py推publish_mlr_branch远端MLR，精确stage审查哈希/大小/secret，不上传模型、Q、原成功特征/动作、噪声表、RNG二进制。下载轻量证据与完整报告（验证与正式分开）、成本及单seed/验证10条阈值非显著/仅锚点子空间保护/历史global RNG等边界。所有授权工作验收交付发布后删除本自动化，不开启后续任务。
