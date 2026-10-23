# 用户停止TAC-R/ST训练：终止记录

用户明确要求“停止训练，TAC-R设计有问题”。该最新指令覆盖此前继续队列、条件扩展与自动处理工程问题的授权。未经新的用户明确授权，不得重启/恢复/补评、替换token读出或另启修订版训练。

- 2026-10-23 12:17:56 CST，先暂停已验证的队列控制器阻止新入口，然后向当前队列5个实际进程发送SIGTERM并恢复其信号处理；PID/start_ticks/命令/cwd均记录。控制器95924、supervisor118243、pipeline118359、torchrun118364、train118372全部停止。
- 12:18:02记录无残留进程、GPU空闲，1462冻结文件停止前后hash一致；没有删除或下载任何checkpoint/演示/缓存payload，没有重新训练或环境调用。原队列RUN_STATUS、退出文件及历史失败均保留，不伪改为正常exit0；服务端新USER_TERMINATION_BY_USER.json明确标记人为终止。
- TACRST41：91个完整epoch（Task6截至epoch90），完整epoch计14287主步，正式0。未完成epoch不计完成。恢复检查点latest.pth SHA d017fc03c35416136e945c6a6078e51ddf84bfa519269880386a4d1035e073bd5及preflight恢复/标签等原地保留；没有Task6 ep99最终模型，不伪称完整负结果或已验证ST效果。
- TACR41在停止请求前已经完成400epoch、105500主步与105500表征步、200正式20vector10cell，原结果/4主100键/4部署14键/4学生20键/输入与成员IFA/完整账目/有效退出凭据保留。此前用户“检查进度”已开始下载的932个轻量文件全部SHA核验后，本地audit_final只读复算实际通过；没有新训练、补评或额外闭环。AUC619/960=64.479167%，低于同seed RA701/960=73.020833%，未过扩展门。此后只保留结果，不继续三seed或追加实验。
- TACR42/43原来已因门未过而未启动；所有TACRST后续seed及任何未来扩展现在均由用户取消。
- 监控align-object-seed-goal先暂停，停止确认后已通过工具删除，并核本地automation.toml不存在。没有保留自动恢复任务。

## 保留与核验

stop_all_by_user_20261023_121741/EVIDENCE.json与LOCAL_AUDIT.json保存实际信号、零残留/GPU、完整epoch计数、35个pth工件的路径/大小/SHA及原退出文件。这里35项也包含标签和RNG恢复工件，只发布元数据，不发布二进制。

verify_user_stop_readonly另行复核所有已记录pth大小/SHA、当前队列无残留、GPU及日志清单；不修改服务器。所有日志、恢复状态与既有Temporal token诊断均保留。A ALIGN已完成结果及所有旧取消实验未改动。

实验终态为CANCELLED_BY_USER，不是全部科学实验完成或三seed成功。TACRST41为EARLY_TERMINATED_BY_USER，不纳入完整性能比较。token可见性问题仍只证明结构性限制，没有据此把TACR性能归因或自动启动新设计。

Independent post-stop verification at 2026-10-23 12:19:58 CST: zero residual queue processes, GPU idle, all 35 retained pth artifact sizes/SHA match, 25 log metadata records retained, frozen 1462 files unchanged. Local audit: verify_user_stop_readonly_20261023_121930/LOCAL_AUDIT.json. No server writes, model forwards, optimizer updates or environment steps.
