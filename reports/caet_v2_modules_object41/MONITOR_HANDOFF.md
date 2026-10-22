# 独立模块 OBJECT seed41 交接

观察时间：2026-10-22 12:15:08 CST。整体 PARTIAL。用户明确确认三个模块分别独立叠加：A=TAC，B=RSET（不含TAC），C=EFTR。权威科学参数见 PROTOCOL.md；不可根据负结果调门槛。RA41及原CAET-v2 seed41只读复用。

远端Q=/data/mlr_ifa/experiment_workspaces/CAETv2_TAC_RSET_EFTR_OBJECT41_20261022。controller PID82742，顺序 TAC→RSET→EFTR，不依赖前一方法成绩。各worker为Q/对应variant；输出/data/mlr_ifa/experiments/CAETv2_{variant}_OBJECT_seed41_20261022。旧COVR全部取消，LIBERO50预训练已取消，任何旧controller均不得复活。

最新证据 progress_20261022_121508/SNAPSHOT.json、LOCAL_AUDIT.json。689个源码/调度hash通过；TAC实际train PID82968，Task6完成6epoch/942主步，正式0。prepare/labels6/gpu_preflight/variant_gpu_preflight均exit0。原8次真实隔离优化器更新、关闭原生一致性、全状态恢复及功能检查PASS；TAC额外7步梯度有限非零检查PASS、0额外优化器更新。RSET/EFTR尚未GPU准入，不可套用TAC通过。

A固定128留出样本的7vs8执行分布KL均值10937480.735087，分量切换5.46875%，动作均值RMS差.0706796335，夹爪均值绝对差.1093750009。差异很大，不能把功能PASS称为分布等价；训练辅助7步、部署8步的限制必须随结果披露。后续各阶段ep99继续留存INPUT_SHIFT审计。

零训练Gate1已完成，不重复：TAC5601合法同动作标签，Task7无支持回退，Task8/9支持197/213；RSET/EFTR维持v2图178/206。RSET排除同demo的专家动作自查询在严格动作门下跨任务支持为0，保留负结果；实际策略候选与拟合分层可靠性仍待实测。EFTR Gate1复用相同标签/图，有来源hash绑定。

每组固定400epoch105500主更新、4ep99、200正式20vector10cell；效果拟合额外更新、C每批两个AdamW候选试算分别记账。C从同一完整状态比较native/aux，拒绝恢复native全状态；参考仅上一阶段本run ep99。局部候选通过不代表全旧记忆或闭环保证。

检查：在本目录运行 python -X utf8 check_progress.py，SSH凭据仅内存/getpass。脚本保存只读快照及hash核验，不应重复启动。健康观察后保存报告/push结束，不守候。当前/data普通7.184GB、root全部12.054GB，system普通5.902GB；data保留2GiB，system不使用保留块，不删旧资产。训练正常不重启；失败先确认停进程、冻结hash与证据，采用新版本工程恢复，仅完整epoch全状态恢复。不能覆盖冻结源/合同/失败码或盲重启部分globalRNG正式评估。

完整supervisor/pipeline退出0、PROCESS_EXIT_ACCEPTANCE、4CP严格100键/RA载荷/IFA/初始化Task6tensor、机制与200/20及无残留之后：package_final.py <variant> --manifest-only；resume_light_download.py <包目录>；全SHA通过后 audit_final.py <包目录> <variant>。这些新适配最终脚本目前仅静态语法通过，须实证验证。包内legacy v1目录实际绑定原CAET-v2 seed41，报告必须标v2 baseline，不可混同v1。最终分别与RA/v2复算完整矩阵和AUC/FWT/NBT/OldFinal/FinalAll及成本，不做单seed显著性结论。

每轮按AGENTS运行本目录publish.py，向publish_mlr_branch远端MLR精确stage、非force push并核SHA/本地receipt；仅审查后代码与轻量元数据，禁止权重/原动作状态/latentbuffer/RNG二进制或BUNDLE。监控automationId=caet-v2-object41，ACTIVE，每30分钟。无实质变化保持安静；三组全部独立验收和发布后才删除该监控。


## 2026-10-22 零训练RSET诊断追加

见rset_diagnostic_20261022_123604/REPORT.md、LOCAL_AUDIT及SCIENTIFIC_FINDINGS。原门首要问题为状态cos>=.90：Task8/9原效果支持178/206中排同demo仅3/0有任一状态邻居，最多1demo，自身demo加回上界仍2。取消Top20或仅用当前任务库仍0联合支持，不能将主要原因归于混合动作尺度。新局部嵌套LODO原型专家交集160/178及197/206，但所有20个夹爪连续OOD阈值为0，随机执行动作几乎全拒绝，判NOT_READY_DEGENERATE_GRIPPER_BOUND；不能以分量均值高覆盖冒称随机执行可靠性。0训练0优化器0环境0服务器写入，50批CPU冻结原v2前向、4CP严格100键与SHA、标量独立复算PASS。不更改冻结RSET、不放宽阈值、不把诊断原型当新增训练授权。

进度快照progress_20261022_123307：TAC171完整epoch31817主步/0正式，Task6实际tensor同RA、689冻结hash通过。RUN_STATUS读取比epochs早一个epoch，属于只读快照时序；没有恢复或重启。RSET/EFTR尚未GPU准入。原10MB标量证据仅本地，发布27个结构化片段及可逐字节重建的manifest。


## 2026-10-22 用户修订：停止七步TAC，转独立八步辅助头

用户明确要求主8步BC/Temporal Transformer/GMM/闭环不变，新增训练专用下一动作H_aux后重启。tac_user_stop_20261022_131004/STOP_RECEIPT.json：旧dispatcher82742和TAC训练树停止，297完整epoch/68373主步、正式0，689原冻结hash通过。旧实验标USER_STOPPED_FOR_REVISION，不是完成或NO_GAIN；原产物只读，不能恢复旧队列。新工作目录../20261022_tac_aux8_object_seed41，新实验TAC_AUX8，从官方初始化/空本run缓冲开始；原RSET/EFTR仍独立授权、按原版本待安全衔接。新版本协议与实际准入以新目录为准。

### RSET夹爪语义零训练诊断完成（不替换冻结B）

rset_semantic_v2_20261022_125521/REPORT.md、LOCAL_AUDIT.json与LIGHT_PACKAGE_AUDIT PASS。18份安装源追踪、28个纯formatter测试确认PandaGripper用sign命令增量修改内部目标：负OPEN/零HOLD/正CLOSE，非瞬时物理开合。0训练0优化器0环境0服务器写入。专家效果图交集Task8 160/178、Task9 197/206，与先前专家覆盖相同；冻结原v2后期Normal采样支持157/178和194/206。6维机械臂±10sigma压力768例全部拒绝，独立嵌套LODO排自身demo。解决了旧原型夹爪连续零阈值的系统误拒绝，不证明闭环增益。

冻结原v2预测器分层仅TRANSLATION/open通过：Task8 MAE.387732 vs阶段均值.630151/臂动作置换.406426，Task9 .345851 vs.605613/.381036；此open为物理夹爪间距组，不是OPEN命令类别。STATIC/open不如分层均值，其他组证据不足；当前任务只有1个留出demo，不能宣称当前任务多demo泛化。原v2模型代理不能替代未来RSET时序注意力预测器自身验收。完整12.42MB标量证据本地保留，发布31个可逐字节重建结构化片段，SHA771cc40f011aa447e73ca5ebe01c750b5e8d738557e9c1320bbf594c964e47e8。
