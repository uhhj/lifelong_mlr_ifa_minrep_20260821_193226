# 本机关闭后的执行范围

2026-10-22只读核验。已部署controller由远端subprocess.Popen(start_new_session=True, stdin=DEVNULL, stdout=远端日志, stderr=STDOUT)启动；队列、种子supervisor、训练/评估与独立审计均在GPU服务器运行，不依赖保持本机SSH连接。源码核验：deploy.py启动段、controller.py的run/aggregate/main、seed_supervisor.py。

正常情况下，本机关机或断网后仍可完成OBJECT42→43、精确三seed平均AUC相对RA的严格提升判定，以及判定通过后的GOAL41→42→43训练、评估与服务器独立验收。判定未过则GOAL不启动。服务器保留检查点、日志、正式记录与receipt；原OBJECT41只读复用。未做断开连接破坏测试，未重启或改变科学任务。

当前30分钟监控属于本机对话heartbeat，并未部署成服务器上的Codex代理。本机关闭后不能依赖它即时通知、诊断并修复新的工程故障。服务器controller按冻结协议在异常时留证停止，不自动盲重试。

轻量证据下载、本地独立验收、最终报告和GitHub push仍由本机工作区执行，需开机并重新打开应用/继续当前对话后完成。服务器结束状态为SERVER_FINISHED_AWAITING_LOCAL_AUDIT，不能把它冒称最终本地验收完成。

官方文档核验：[Scheduled tasks](https://learn.chatgpt.com/docs/automations?surface=app)明确要求使用本地文件的桌面定时任务保持电脑开机及应用运行。此要求影响本机监控与后处理，不影响已脱离SSH启动的远端Python队列。
