# CAET-v2 + TAC / RSET / EFTR：OBJECT seed41

截至2026-10-22 12:15:08 CST，状态PARTIAL。三个独立实验已部署，TAC正式训练已启动；RSET、EFTR顺序排队，各自GPU准入尚未执行。基线直接复用已有RA和原CAET-v2 seed41。

| 实验 | 零训练审计 | GPU准入 | 正式进度 |
|---|---|---|---|
| CAET-v2+TAC | PASS；Task8/9支持197/213时刻 | 8隔离优化器更新PASS；7步梯度PASS | Task6 6epoch，942主步，0正式评估 |
| CAET-v2+RSET | 标签/图PASS；严格动作自查询剩余跨任务支持0 | 待执行 | 排队 |
| CAET-v2+EFTR | 同v2标签/图，来源绑定PASS | 候选更新/恢复检查待执行 | 排队 |

每组固定四任务各100epoch、105500主更新、4个ep99主评估模型、200正式闭环20vector。原RA成员配额和载荷、BC+IFA及执行策略保持；辅助拟合和候选更新额外成本另报。目前没有正式闭环结果，不能报告性能收益或最终验收。

TAC的7/8步输入差异已实际测得：128个固定留出样本中，KL(qexec8||qexec7)均值10937480.735087，分量切换5.46875%，动作均值RMS差0.0706796，夹爪均值绝对差0.109375。功能检查通过不说明分布等价；这一显著差异是A的科学限制，后续结合各阶段审计和闭环结果判断。

RSET零训练专家动作自查询排除同演示后，在新冻结严格门槛下没有保留跨任务支持。该结果不等同于未来策略候选必然全被拒绝；实际候选支持和分层留出可靠性均待实测。保持阈值，完整记录拒绝和激活率，不为制造迁移信号放宽条件。

EFTR将检查真实AdamW候选及完整native状态回退；尚未执行其GPU验证，不能提前宣称可靠保护。即使局部约束通过，也不保证闭环或全部旧记忆性能。

零训练证据见gate1_TAC、gate1_RSET、gate1_EFTR；最新进程/退出/hash证据见progress_20261022_121508，689个冻结源码/调度hash通过。每个入口223源文件，其中209与原v2不变。预算和探索性阈值见PROTOCOL.md。新增监控caet-v2-object41每30分钟续查，正常进程不重启；COVR及LIBERO50预训练保持取消。

尚无足够的后续阶段实际吞吐，暂不提供三组总完成时间；尤其EFTR候选试算成本需GPU实测后估算。下一步继续TAC Task6父张量/初始化/成员验收，再看Task7回退、Task8/9留出与激活，最后完整闭环验收。


## 2026-10-22 零训练RSET诊断追加

见rset_diagnostic_20261022_123604/REPORT.md、LOCAL_AUDIT及SCIENTIFIC_FINDINGS。原门首要问题为状态cos>=.90：Task8/9原效果支持178/206中排同demo仅3/0有任一状态邻居，最多1demo，自身demo加回上界仍2。取消Top20或仅用当前任务库仍0联合支持，不能将主要原因归于混合动作尺度。新局部嵌套LODO原型专家交集160/178及197/206，但所有20个夹爪连续OOD阈值为0，随机执行动作几乎全拒绝，判NOT_READY_DEGENERATE_GRIPPER_BOUND；不能以分量均值高覆盖冒称随机执行可靠性。0训练0优化器0环境0服务器写入，50批CPU冻结原v2前向、4CP严格100键与SHA、标量独立复算PASS。不更改冻结RSET、不放宽阈值、不把诊断原型当新增训练授权。

进度快照progress_20261022_123307：TAC171完整epoch31817主步/0正式，Task6实际tensor同RA、689冻结hash通过。RUN_STATUS读取比epochs早一个epoch，属于只读快照时序；没有恢复或重启。RSET/EFTR尚未GPU准入。原10MB标量证据仅本地，发布27个结构化片段及可逐字节重建的manifest。


## 2026-10-22 用户修订：停止七步TAC，转独立八步辅助头

用户明确要求主8步BC/Temporal Transformer/GMM/闭环不变，新增训练专用下一动作H_aux后重启。tac_user_stop_20261022_131004/STOP_RECEIPT.json：旧dispatcher82742和TAC训练树停止，297完整epoch/68373主步、正式0，689原冻结hash通过。旧实验标USER_STOPPED_FOR_REVISION，不是完成或NO_GAIN；原产物只读，不能恢复旧队列。新工作目录../20261022_tac_aux8_object_seed41，新实验TAC_AUX8，从官方初始化/空本run缓冲开始；原RSET/EFTR仍独立授权、按原版本待安全衔接。新版本协议与实际准入以新目录为准。

### RSET夹爪语义零训练诊断完成（不替换冻结B）

rset_semantic_v2_20261022_125521/REPORT.md、LOCAL_AUDIT.json与LIGHT_PACKAGE_AUDIT PASS。18份安装源追踪、28个纯formatter测试确认PandaGripper用sign命令增量修改内部目标：负OPEN/零HOLD/正CLOSE，非瞬时物理开合。0训练0优化器0环境0服务器写入。专家效果图交集Task8 160/178、Task9 197/206，与先前专家覆盖相同；冻结原v2后期Normal采样支持157/178和194/206。6维机械臂±10sigma压力768例全部拒绝，独立嵌套LODO排自身demo。解决了旧原型夹爪连续零阈值的系统误拒绝，不证明闭环增益。

冻结原v2预测器分层仅TRANSLATION/open通过：Task8 MAE.387732 vs阶段均值.630151/臂动作置换.406426，Task9 .345851 vs.605613/.381036；此open为物理夹爪间距组，不是OPEN命令类别。STATIC/open不如分层均值，其他组证据不足；当前任务只有1个留出demo，不能宣称当前任务多demo泛化。原v2模型代理不能替代未来RSET时序注意力预测器自身验收。完整12.42MB标量证据本地保留，发布31个可逐字节重建结构化片段，SHA771cc40f011aa447e73ca5ebe01c750b5e8d738557e9c1320bbf594c964e47e8。


## 2026-10-22T13:52:37.786541+08:00 用户停止所有训练（覆盖此前继续执行指令）

用户要求停止训练并清理废用数据。所有当前训练与调度已停止，GPU 空闲；caet-v2-object41 自动化已删除。不得恢复 TAC_AUX8、旧 TAC、RSET 或 EFTR 队列。TAC_AUX8 182 完整 epoch/34314 主更新/0 正式；旧 TAC 297 完整 epoch/68373 主更新/0 正式；RSET/EFTR 未启动。状态 USER_STOPPED，不是实验完成或工程失败。

已清理 37 个通过预检的临时序列化副本，实测释放 11.460 GB，6 个合同引用候选保留。正式检查点、实际训练恢复点、数据、特征、buffer 与审计记录保留。清单和独立核对见 ../20261022_stop_cleanup/REPORT.md、LOCAL_CLEANUP_AUDIT.json。历史进度和协议保留但不再是继续运行授权。
