# RSET任务内机械臂LODO × 执行夹爪类别：零训练v2

2026-10-22。先读取实际安装源码并取得gripper_semantics_20261022_125239/SOURCE_EVIDENCE.json，再冻结本规则。原执行链为eval_metric→CompletePolicy/GMM argmax Normal→vector/control wrapper→BDDL→RobotEnv→SingleArm.control→Manipulator.grip_action→PandaGripper.format_action→actuator range。PandaGripper源码用np.sign(action)以speed=.01更新内部二指目标，再clip[-1,1]；负值OPEN，精确零HOLD，正值CLOSE。命令类别不是瞬时物理开口，HOLD不是默认OPEN/CLOSE。无额外死区。相同历史内部目标和同类命令产生相同formatter增量，28个纯函数边界检查PASS，无环境构造/步进。

仅对上版诊断估计器进行夹爪语义修正：当前任务训练库、外层整demo剔除、内层LODO及demo等权校准、状态描述/每demo联合检索、最多3且至少2个独立demo规则不变。机械臂只用前6维总体std floor1e-3；联合平方距离=状态弦长²+6维arm RMS²。夹爪只决定候选命令类别是否匹配，不进入距离/归一化/分位数。tau中位数、联合95%与state/arm各99%分位OOD界限不改，不追加容差。C_arm为过OOD后的exp(-d²/tau²)，C_gripper要求同类命令候选至少2独立demo，否则0；两者共享同状态条件邻域并分别记拒绝原因。C_effect原支持图不变，B仍保留合法下一动作时序，不混入TAC。

专家全训练嵌套LODO、未用于校准的原效果整demo留出、原图交集均报告。冻结原CAET-v2 seed41同4个入/出阶段ep99，CPU原8步eval严格100键；GMM均值和固定seed20261022 Normal采样分别统计，不把均值替代实际采样。不追加策略或效果模型训练，不改已冻结RSET队列。

OOD仅合成动作数值、无环境：对每个原支持专家查询，保持state/gripper，以其外层训练库sigma对6维机械臂全部分别±10sigma（两组），以及每个维度单独设置±1（12组，归一化边界压力测试而非保证每点OOD）。±10sigma组须检查仍能拒绝明显离群；边界组仅报告不强称每个应拒绝。同类夹爪±1e-6、±.2、±2按源符号替换，要求支持/权重与原专家完全一致；精确零单独作为HOLD，不能并入开或闭。

C_predictor只审已有原v2冻结均值聚合预测器，不冒称新RSET时序注意力模型已验收。复核其整demo留出，并按原STATIC/TRANSLATION/ROTATION×aperture分层（物理开口分层不同于命令类别）。对照：训练全局均值、同物理阶段训练均值、留出中同task/phase/gripper命令类别内只置换机械臂6维的固定循环动作（夹爪不换）。检查真实动作梯度有限非零。分层通过需>=12留出窗口、>=2独立(task,demo)、>=12个被实际改变的置换动作、expert MAE严格低于以上三对照且梯度有效。

额外检查同执行语义的夹爪幅值替换（±1e-6、±.2、±2）对冻结效果预测的影响；各MAE也须低于阶段均值基线，否则该阶段不得宣称已通过本轮保守可靠性检查。该条件在本次前向前冻结，不能因失败放宽。完整当前任务单demo留出另报，不把混合任务的多demo数伪装成当前任务多demo。只验证这些观测邻域内的离线预测，不证明策略候选机械臂OOD外梯度可靠或闭环增益。C_effect×C_arm×C_gripper有支持不等于四门通过。

原始证据不覆盖；输入/source/CP前后hash与本地标量/身份/校准复算。只输出身份/误差/计数/哈希，不下载原动作或latent。按AGENTS发布；本轮0优化器更新0环境调用0服务器写入。正常训练进程保持运行，新增结果不得偷偷改现有冻结源码。
