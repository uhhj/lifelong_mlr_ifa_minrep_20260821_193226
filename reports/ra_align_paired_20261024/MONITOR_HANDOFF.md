# 当前有限配对诊断：只跟踪，不重新部署

2026-10-24 18:58 CST。目录D:/论文课题研究/MLR+IFA/execution/20261024_ra_align_paired。用户新授权：RA补同窗口漂移，再固定初态/逐轨迹噪声做Stage8/9同历史双检查点闭环；只有可重复有害改变得到支持才另登记三组保护分叉。目前0训练，不自动启动保护。

RA离线实际完成：ra_drift_readonly_20261024_184617/LOCAL_AUDIT与OFFLINE_COMPARISON实际PASS，3seed共1356共同窗口/1283合法a[u+1]标签、6CP/174批/2712 CPU模型窗口/0环境更新，ALIGN已有结果只读复用。RA分量切换28/32/28，合法gripper flips6/6/3中改善3/6/1、恶化3/0/2；RA均Huber变化-.00007683/-.00709339/+.00105116，混合下一步NLL均变差（RA本来监督a[u]）。不是所有离线指标一起改善，更不能把模式切换/夹爪纠错当ALIGN特有退化。

唯一活跃root=/data/mlr_ifa/experiment_workspaces/RA_ALIGN_PAIRED_GOAL_20261024_v1_engineering02，supervisor7051、worker7052，合同12fc5c69ebacbbb0362d089122c5e07f230c8ed6d17294e34f3073bbe8765e67。private=/root/mlr_ifa/experiments/RA_ALIGN_PAIRED_GOAL_20261024_v1。科学code/source/INPUT_SPEC/PAIRED_PROTOCOL冻结；禁止重跑deploy.py/prepare_engineering02.py或正常worker。engineer01缺LIBERO_CONFIG_PATH在import阶段EOF，0模型前向/0更新/0环境，无private产物；原失败、源码、contract保留。02只补原环境路径，自己的真实GPU准入12CP严格100键、长度1–8/GMM最大分量Normal/前端cache parity PASS：208下游、16前端调用、0env/update。cache最大feature误差9.30e-6，GMM参数1.91e-6，不声称旧CUDA逐位相同。

snapshot_20261024_185657实际18:57:21，LOCAL_AUDIT PASS_PROGRESS_NOT_FINAL，20/240新诊断回合、2/24vector完成，RA_s41_driver9_g0正常进行，源合同不变，GPU7052健康。前2完整vector均76.386551sec，扣当前进行38sec后剩线性1642.486sec=27.375min；阶段加载/保存/最终本地审计另计。按用户按预计整体完成时间规则，下一30分钟后检查（来自实际剩余+少量余量，不是固定30min轮询）。预计rollout19:25–19:35，最终审计另计，早期两batch仍不确定。健康不变化安静，重要阶段/失败/完成才通知。

有限范围GOAL Task8，seeds41/42/43×RA/ALIGN×driverStage8/9×20初态=240回合，24vector，600步+5zero。逐轨迹独立RandomState(20261024+seed*1000+800+init)FP32正态噪声，跨方法与stage共享，环境seed=seed+group。每个driver实际轨迹上两个checkpoint使用同一FP32缓存8帧历史并哈希；不比较不同状态的同时间步动作。全程真实Temporal输入/低维观测/双GMM参数/执行动作私有保存；原始双相机仅三首次事件前8帧及终止前16帧+终止后帧，非全程RGB。首次mean abs>1e-3仅定位不是危险门，模式切换/flip不等于坏；timeout600不说明不可逆失败的时刻。所有正常失败保留不重试，不扩大seed/预算或自动保护训练。

运行检查：python -X utf8 check_progress.py（getpass仅内存，root@jq1.9gpu.com:11920已授权）→audit_progress.py <snapshot>。无权重/原动作状态/features/RNG/private载荷下载。若健康未完按实际剩余重新调同一监控，不hover；不能运行其他旧ALIGN/CAET/TACR/readout队列。

自然worker/supervisor exit0、无自身进程后：collect_final.py调用外置final_readonly.py在CPU独立重构全部private history/hash/噪声/执行动作/事件/CP/source，无前向或环境、无服务器写入；只下载白名单light scalar/jsonl。然后audit_final_local.py <final>独立240回合/120pairs/24vector/SS-SF-FS-FF/共同初态噪声/同history与关键片段复算。新final路径目前仅AST未实际完整PASS；包装错误保留attempt可外置修，不改server科学脚本、不重训重评。完整light逐步标量留本地，publish.py排除light目录但发布SHA清单、summary/关键片段，避免百MB日志。禁止下载private/原图/模型/噪声文件。

继续解读同历史决策分歧、成功→失败配对及失败前窗口，给出可支持与不能支持的机制结论。不能把单次SF或切换率当保护训练已获因果支持；发现可重复有害模式后先冻结同Stage8三分支协议、Task8保持/Task9学习及预算，不进行无界扫描。若诊断不足如实交付，完成本轮验收分析发布后删除本监控，不留空轮询。

每轮按AGENTS publish.py，经publish_mlr_branch精确review类型大小secret/hash、串行stage非force推MLR并核remoteSHA/receipt。TLS必要OpenSSL/HTTP1.1保持证书验证。system普通保留1GiB/data根2GiB，private新增最坏3GiB按稳定已分配抵扣；18:56已分配50.65MB、system普通4.598GB，扣余新增后余1.427GB>1GiB。不删旧资产、不减预算、不付费。当前总PARTIAL，离线已完整接受不重复。
