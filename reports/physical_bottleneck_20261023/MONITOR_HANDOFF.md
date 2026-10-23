# 活跃：新瓶颈定位有限实验

最新用户授权：同单位诊断→读出→空间patch，单因素；自动处理工程问题并push；新A通过后才B。先完整读PROTOCOL.md、REPORT.md、RUN_STATUS.json、ACTIVE_ROOT.json、新launch及最新snapshot。父experiment execution/20261023_physical_effect_probe已经完整STOP_STATE_GATE，只读不复活；所有旧TAC/EFTR/TACRST等取消。此次是新增授权，原A失败不得改写。

新Q=/data/mlr_ifa/experiment_workspaces/PHYSICAL_BOTTLENECK_OBJECT41_20261023_v1，controller120819，合同c26f09ff41b912935a85e6eb92e2eeda62ff76f8fc1e0ccae8c1d5cd3d75c8dc；24本合同文件加原207源合同。SSH root@jq1.9gpu.com:11920已授权，getpass内存，凭据不落任何文件日志。正常check_progress.py→audit_progress.py <snapshot>，核源/inputSHA/进程GPU磁盘/退出/完整epoch与优化步/已选CP和测试case。正常不重启，检查后保存报告并publish一轮结束，不守候。只有有意义阶段切换/失败/完成/需用户行动通知。

D0 same_unit_readonly_20261023_150704完成50原nativeCP/690批前向0优化0环境；audit_same_unit.py本地身份/计数/原测试复现PASS。P3物体－目标训练median约6.9mm P90约12mm、测试median21–24/P9046–54（各折指标平均），有泛化差距。不要重复D0。

队列controller: readout_admission→readout→patch_export→spatial_admission→spatial→audit。P0/M/Q已自身readout admission3隔离AdamW各一次初始+参考续步+恢复续步，共9调用，不能套G/S；G/S待自身准入。正式每模型×2task×5fold×200epoch，P0/M同时记录50/100/200预算选择点，Q/G/S200；共50fits10000epoch。M原P3精确重现；Q仅均值换query（额外128参数）；G/S同架构仅全局重复或patch值不同，两者有效batch128 micro16 FP32。D1不根据测试选择D2，列表全部执行。native输入、标签mask、原7/1/2demo划分、lr3e-4 wd.01、归一化SmoothL1、种子不变。原M/P0的50预算预测与父实际CP严格allclose验证，失败须查工程不调科学参数。

patch来自原官方CLIP全部已加载权重，postLayerNorm→官方image_proj→语言FiLM，逐相机独立；不能绕过预处理、LoRA或FiLM。新导出全局路径必须与旧native全部窗口一致。数据root至少2GiB/system普通至少1GiB，patch约2.45GB/总新增≤3.3GiB实际监测。不删旧资产/降低FP32或科学batch/新付费。科学源冻结不覆盖；故障先确认停止/hash一致/保留exit和日志，独立版本化工程修复，仅完整epoch model/optim/best/allRNG/grads/modes/requires_grad/cursor恢复，不能盲重跑已完成测试或正常科学失败。

A要完整90个预算case、50fits、10000epochs、完整独立标量复算及源/输入/CP绑定再接受。门原10/25mm、相对P0-200 demo-bootstrap下界>0、动态当前优于旧目标且去最新视觉更差，两个任务两个位置都过。固定优先M200→Q200→G200→S200，所有候选都披露。若都失败STOP_STATE_GATE，不B不扩seed，发布后删监控。若过：controller止于A_PASSED_B_IMPLEMENTATION_ADMISSION_PENDING是有意准入边界，不是失败；按用户授权自主继续完整实施和审核B，不能停在承诺或宣称B已完成。B沿父PROTOCOL的Q/P效果验证关、三组原GMM+IFA/RA Task6→7及180主+120干预共300episodes，物理选中架构新初始化，不用A训练权重；Q/P无专家动作/未来真值输入。B再按用户确认严格门扩ALL3arms及干预42/43，不是RA-AUC门。新架构B实现尚未有，不可调用不存在的脚本或假定已验收。

十demo已用于上一轮测试分析，明确探索性，不称独立确认或总体显著。追加实验需保持有限预注册范围，不反复调参直到过关。原A结果不覆盖。每轮本目录publish.py→publish_mlr_branch精确review/stage/push MLR核remoteSHA/receipt，代码报告标量允许，权重/predvector/rawstate/action/latent/patchcache不发布。所有所选分支本地验收和发布完成后删除监控。
