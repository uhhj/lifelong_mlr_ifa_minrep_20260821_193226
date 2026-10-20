# ZR-v3 OBJECT seed41监控与验收

最新2026-10-20 13:16 CST：v3已40epoch/10550step/105240曝光完成，calibrate.exit=0；16全dataset结构检查PASS、最大实际KL0.4999999999568453。独立evaluate进程已启动，当前快照0/200，不提前标COMPLETED。见progress_20261020_131602、LAUNCH_REPORT.md与LOCAL_CALIBRATION_ACCEPTANCE.json。下文30epoch为历史检查。原selected-Gaussian训练NLL较混合NLL大很多、有限且下降，不因此改科学参数。

用户要求利用现有资产完成附件低成本验证。先读PROTOCOL.md、USER_PROTOCOL.txt、RUN_STATUS.json、最新progress、ORCHESTRATION.json和SOURCE.patch。原RA/ZR-v2均已完成，只读复用，不恢复任何旧控制器。当前新自动化`zr-v3-object-seed41`每小时，临近结束可30分钟，全部授权工作完成交付push后删除。

服务器N=/data/mlr_ifa/experiment_workspaces/ZRv3_OBJECT_RA_seed41_20261020；新输出/data/mlr_ifa/experiments/ZRv3_OBJECT_RA_seed41_20261020_v3，条件消融v3a同前缀。新源在N/v3/source/zr3_exp，v3a仅达到固定门槛后创建。GPU3090，数据盘约25GiB足够。SSH已有授权，凭据仅内存/getpass。supervisor于2026-10-20 13:11 CST启动，初始PID37501、controller37502、calibrate37509，PID只作历史，必须核验实际进程。

服务器有限队列已经部署：v3校准40epoch10550step，200原协议闭环，独立audit+4CP加载和进程退出验收。相对RA父AUC>701/960、FWT>=3/4、OldFinal>=47/60且全部工程PASS才新建v3a，执行相同预算但无投影；否则停止。最多两组，不自动扩seed/GOAL。不使用旧ZR-v2 UV。每阶段10epoch，校准结束ep9选择固定，不挑最好epoch。

v3 preflight全部退出0：25继承软件测试；9新核心性质CPU+CUDA各一次；真实语言/缓存门准入；恰好4次隔离真实optimizer调用含恢复重放；真实图片history1/4/8原生分布与配对动作一致；成本/自身梯度隔离；冻结。0环境step，不重复已通过预检。正式从新零UV和重新初始化RNG开始。最新已确认Task6–8完成30epoch6930step；13结构扫描全PASS，候选KL最高24.093636，实际最高0.499999999786。此为过程检查非闭环效果。最新轻量完整快照以progress目录为准。

科学方案：只修正argmax分量均值并优化该Gaussian NLL；v3 epsilon=.5 nat逐输入投影，FP64计算后按实际FP32输出再核验，浮点超界才nextafter向父均值收缩，最多16ULP，阈值不能变。NativeHead与decode共用execution_core.calibrate_distribution，原尺度/logit/其他均值精确保持。rank8、固定官方T0、FP64双向门、0.1幅度、旧UV本方法暖启动、自身梯度屏蔽grad=None、父模型和缓存原样。实际新loss可显著大于原混合NLL，不能据此擅自改loss/早停；仅非有限或结构断言失败才工程/数值阻塞。

运行本地`python execution/20261020_zrv3_object_seed41/check_progress.py`：只读快照、输入冻结hash、生成allowlist轻量包、校验安全路径/每文件hash，本地audit_download.py复算，写PARTIAL或COMPLETED。脚本密码getpass不保存。远端可用`/data/mlr_ifa/env/bin/python N/monitor_snapshot.py`、`N/package_light.py`。下载仅元数据/episode，禁止checkpoint、RNG二进制、latent、buffer、原动作。最终package包含两个历史控制组逐episode，不是新结果。

原生评估0–19初态、20/cell、10lane、env41,42、600步/5zero、stage global RNG41、选中分量Normal；不引入独立轨迹RNG、不重评基线或增加预算。每次在线前向实际KL检查并在cell保存最大值/投影样本数。训练检查16次，覆盖每阶段全实际dataset（含合法旧buffer），记录unselected/zero/selected/gates/bound。

每组需要40epoch10550step105240exposures4CP200episode20vector、所有结构/载荷/父输入hash和pipeline/calibrate/evaluate/audit退出0，PROCESS_EXIT_ACCEPTANCE PASS。全队列还需controller.exit/supervisor.exit0、N/PROCESS_EXIT_ACCEPTANCE PASS、无scientific worker，再本地独立精确Fraction复算全部矩阵/指标/差、投影率KL/成本。v3-A若运行也须完整验收，不能v3完成就结束总任务。v3-A正负都不扩seed。

工程故障先确认进程退出和合同输入一致，保存原失败，只从完整epoch显式恢复全部UV/optimizer/scheduler/grads/modes/RNG/cursor，幂等存储。部分global RNG评估不能盲重启、正常失败不重试；无法保持顺序预算则BLOCKED。冻结科学代码不得悄改/重贴合同。只读分析工程错误可外置版本化修复，不重新运行环境。已授权主动解决可恢复工程错误，但不能更改实验预算或科学条件。

每轮检查按AGENTS执行本地publish.py向publish_mlr_branch远端MLR推送审核后的相关文件，不forcepush；本地receipt记录远端SHA。SOURCE.patch相对既有v2模板，不修改旧源。ORCHESTRATION记录部署时模板和控制器hash；后续本地监控/发布脚本不属于服务器科学合同，不要因此重新freeze。服务器N/RUN_STATUS若还显示初始PREPARING，以真实子目录进度和进程为准，勿据此重启。

估时参考旧相同RA41父下ZR-v2评估十cell总86.25分钟，校准约数分钟。首次v3预计13:15后1.5–2小时结束；如gate通过，再增加约1.5–2小时v3-A。以后用实际新cell吞吐修正，不守候。

报告必须披露：seed41依据旧负结果挑选，单seed探索性；global轨迹噪声未隔离；新校准额外预算；官方pretrain状态元数据unknown；真实逐状态KL不保证轨迹安全或成功率。通过完整v3且补A才可讨论投影增益，仍不能宣称统计显著或普遍因果。
