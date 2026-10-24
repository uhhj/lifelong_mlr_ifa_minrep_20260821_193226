# 实施说明

当前运行见ACTIVE_ROOT.json。原ALIGN所有训练/评估已接受；本次离线RA使用共同保留窗口，不重跑ALIGN。共同窗口比较实际完成并通过audit_ra_local/compare_offline；2712 CPU模型窗口、174批前向、0更新/环境。

新闭环为有限240回合，12旧检查点，三seed×RA/ALIGN×两个驱动stage×20初态，Task8。工程01缺少LIBERO已有配置环境而在import阶段EOF，0模型前向/更新/环境；退出1、GPU空闲、原source/contract不变，证据与源码保存在engineering01_preserved及snapshot_20261024_185233。工程02仅补原环境路径，自己的GPU准入实际通过，模型12×100键、1–8长度、GMM选最大分量Normal、缓存对完整历史空间编码近似一致。准入参数路径208次、前端16次，无环境/优化器。最大缓存特征差9.298324584960938e-6、GMM参数差1.9073486328125e-6，来自不同前端batch形状，未声称逐位等同旧CUDA评估。正式两检查点在完全相同缓存FP32历史上比较，逐步SHA与private重建绑定。

实际运行采用共同逐轨迹噪声，与原global-RNG rollout不同，成功率仅本新协议内部配对。私有原始RGB只保留三首次事件前窗口及终止前16帧+终止后帧；全程精确Temporal输入和低维观测/双GMM参数/执行动作保存。没有全程RGB视频。存储预算3GiB且system普通余1GiB/data root余2GiB，运行逐vector抵扣已分配量检查，不删旧资产。

进度：check_progress.py→audit_progress.py <snapshot>。自然worker/supervisor exit0、无自身进程后才collect_final.py（调用final_readonly.py，私有历史重构/动作/noise/12CP hash/全部源验证、只下载白名单标量）→audit_final_local.py <final目录>。这两个新最终路径尚未实际完整验收，当前只AST，不得套此前PASS。最终分析还需解读SF/FS和关键片段，无反事实同状态动作干预不能断言单步因果。用户的保护三组训练是发现可重复有害改变后的条件任务，当前未注册训练预算或启动任何优化器。

发布仅轻量代码/报告/hash/结果标量。完整逐步标量light目录本地留存且SHA清单发布，不将数百MB逐步记录都推仓库；汇总与关键事件片段发布。绝不下载或发布服务器private帧、动作、模型或噪声载荷。共享checkout串行stage，证书验证下OpenSSL/HTTP1.1必要重试；最终确认remoteSHA与receipt。

## 2026-10-24 最终覆盖

上述“仅AST”仅描述运行前状态：final_20261024_193008的远端私有重构及本地完整审计均已实际PASS，无包装失败或重跑；新增analyze_mechanisms.py及close_diagnostic.py已对全量标量实际运行。最终状态与限制见REPORT.md，禁止将描述性关联当保护训练准入。
