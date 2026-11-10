# ALIGN进度：OBJECT训练完成，评估工程故障

实际观察2026-11-10T15:11:31.619770+08:00。状态PARTIAL_STOPPED_EVALUATION_ASSET_FAILURE，不能将LOCAL_AUDIT中的预算/前缀PASS_PROGRESS_NOT_FINAL当成运行健康或最终接受。

OBJECT41/42/43均400完整轮、105500生产更新、四阶段各5次隔离ALIGN准入通过，自己的train.exit均0。43于2026-11-10T14:25:47.379322+08:00自然退出。累计1200/2400轮、316500/544800生产、60/120隔离、0训练采集、0/1200完成正式。GOAL三个seed未启动。135548所有已记录epoch/准入前缀保留，冻结合同verify通过。

调度器随即启动OBJECT41评估，于2026-11-10T14:26:04.991228+08:00失败并退出1。实际日志FileNotFoundError指向新部署source/libero/libero/assets/scenes/libero_floor_base_style.xml；环境子进程加载场景失败，主进程记录ConnectionResetError。错误属于新目录资产部署遗漏，先前只导入预检未覆盖真实资产加载，不是已定位的GPU故障。当前nvidia-smi正常，4080SUPER利用率0%、显存10MiB，无本轮进程。

原attempt有environment_construct_start/complete、vector_attempt_start（Task6/task6、初态0–9）和technical_exception，0完成回合。异步构造complete日志不代表所有子进程成功；不能将该attempt抹去或声称环境成本为零。原controller.exit1/evaluate.exit1、日志SHA、失败记录、已完成训练和模型全部保留。本轮检查只读，没有重启、重训、重复评估或修改资产。

此前18:10–19:50估时已失效，当前队列已经停止，不能继续按原速度倒计时。恢复前需要版本化核验既有资产来源/路径、失败attempt与RNG边界，保留原失败及实际成本，再从未完成评估接续；OBJECT600正式完成后才允许GOAL，不能跳过该顺序或重新运行原controller。最终完成源也须按真实恢复工程适配，禁止伪造旧exit0。
