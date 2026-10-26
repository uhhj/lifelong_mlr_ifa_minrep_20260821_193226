# S OBJECT seed41/42/43监控交接

继续用户授权的S OBJECT seed41、42、43完整有限实验。先完整读本目录PROTOCOL.md、IMPLEMENTATION_NOTES.md、REPORT.md、RUN_STATUS.json、ACTIVE_ROOT.json及最新证据。用户授权自主完成工程障碍、实验、独立验收与发布；不因此增加科学预算或重试正常负结果。

唯一root /data/mlr_ifa/experiment_workspaces/CCBP_V4_S_OBJECT41_42_43_20261026_v1，controller15770，合同8319d05c3795f500e1ad546dca1aa1de14b0b05f5f57dd6f6ce32f8c5260e311，48冻结项。严格train41→42→43→eval41→42→43；每seed400epoch105500主步40训练采集200正式，总1200/316500/120/600。禁止重新deploy/build/controller或重训已完成阶段，旧GOAL和所有旧实验不恢复。

最新check_readonly_20261026_165809实际16:58:09CST本地PASS_PROGRESS_NOT_FINAL：seed41 Task6完整3epoch471步，自己4隔离GPU更新和短历史1–8、完整恢复下一步及原ALIGN更新准入PASS，GPU15776健康；42/43尚未准入，环境0，48冻结项及父源不变。capacity_readonly_20261026_170019实际PASS_NOT_FINAL，system留1GiB之外余372576256B、data留2GiB且扣剩余6GiB界之外余41742307328B，无清理/付费。首次只读预检小写benchmark断言错误已留证改规范LIBERO_OBJECT，0写入/更新/环境；部署与科学源码未重试。

当前凭据worker17968，先读.session_control/status.json；queue_remote.py check_readonly.py→读对应result输出的实际目录→audit_progress.py SNAPSHOT（自动包含已存在评估审计）。不要猜请求秒数为目录。capacity_readonly.py按稳定唯一inode抵扣总6GiB，保留root普通1/data root2GiB；不下载payload。活跃worker不重启，若确已关闭可隐藏启动credential_session.py自动读用户授权Windows Credential Manager；禁止密码明文文件/参数/日志/聊天输出，.session_control不发布。

按整seed含采集或全部评估预计结束点更新同一监控。首次下一140分钟基于同卡GOAL S纯epoch82.245707min按OBJECT剩余主步外推113.510964min，加25min未实测余量，约19:15–19:30检查seed41整段结束。不是自己的辅助吞吐或上界，不是三seed全队列ETA。后续用自身实测修正，不内部Task短轮询；健康无变化安静。一轮检查、审计、记录、发布后结束不守候。

S损失/缓存/成功筛选/固定获得时教师/RA和ALIGN规则严格同已验收实现，各seed用自己的OBJECT官方初始化与人口mask，不从GOAL复制标签。采集初态20–29，每任务10轨迹、成功才保留固定quota，最大128/task512/seed，不补采不重分配；正式0–19每cell20，600步5zero，原argmax Normal配对逐轨迹噪声。66自己的隔离更新、216正式额外autograd、78准入autograd另列；不能套GOAL或其他seed准入。3090按用户授权只记录硬件差异。失败先核停止/hash/完整恢复和attempt原证据，再版本化工程修，不覆盖冻结源或盲重启部分评估。

全部自然controller.exit0且无自身进程后queue_remote.py final_readonly.py→audit_final_local.py FINAL→describe_final.py FINAL。外置最终路径目前仅AST，须实际12CPU strict CP/3完整恢复/12源教师缓存重放与抽样/120训练600正式初态字节noise/全部标量接受；外置错误留证修，不重训重评。三seed完整矩阵/AUC/FWT/NBT低好/OldFinal/FinalAll、所有负单元、S内跨阶段SF/FS及离线与闭环解释。OBJECT历史基线仅只读外部参考，非同噪声/等交互，无C/O OBJECT三seed，不造因果或显著。完整验收机制解释及发布后删除此监控，不自动扩seed或调参。

每轮本目录publish.py按根AGENTS精确审核轻量类型/大小/secret/hash，串行stage非force推https://github.com/uhhj/lifelong_mlr_ifa_minrep_20260821_193226分支MLR，核remoteSHA/localreceipt。必要OpenSSL+HTTP1.1仍验证证书。只发布code/report/hash/scalars/outcomes，禁CP/weights/features/rawactions/states/demo/buffer/RNG/凭据及未审bundle。发布进度不等于科学接受。
