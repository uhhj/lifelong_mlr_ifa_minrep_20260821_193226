# 完整 S 剥离后的 MLR+IFA / MLR+IFA+ALIGN 配对重跑

实际观察：2026-11-10T22:01:27.229126+08:00。证据 `check_readonly_20261110_220126`，`LOCAL_AUDIT=PASS_PROGRESS_NOT_FINAL`。

用户最新授权两个套件、两种方案、seed41/42/43，共12run。旧结果和完整S只读保留。这次使用真实完整S的build/初始化、模型、优化器、每任务100轮与评估实现；Random样本顺序与Random的IFA任务对冻结。两组仅INPUT.variant不同：ALIGN用next_action与causal_valid；MLR用原窗口末动作和全有效掩码。E/R/q/RA/候选门控/训练采集均删除，不复用上一轮global Normal评估器。

唯一root `/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1`；私有输出同NAME下experiments目录；合同 `b54b0c2d8d8b490d6f7c394f7f99fa6689222861240454b4dd4eca870158b598`。禁止再次prepare/freeze/deploy/controller或已完成部分。每次先读ACTIVE_ROOT/RUN_STATUS/PROTOCOL/IMPLEMENTATION_NOTES/PARENT_BINDINGS/PREPARATION_VALIDATION和最新实际证据。初态tensor与nonCUDA RNG必须实际相等，CUDA实际比较记录。Task6对完整S仅观察（被删机制及硬件皆可能影响），不预称相等。

|case|完整轮|生产optimizer|已实际通过的stage准入|正式|
|---|---:|---:|---|---:|
|OBJECT_ALIGN_41|400/400|105500|6,7,8,9|0/200|
|OBJECT_MLR_41|400/400|105500|6,7,8,9|0/200|
|OBJECT_ALIGN_42|19/400|2983|6|0/200|
|OBJECT_MLR_42|19/400|2983|6|0/200|
|OBJECT_ALIGN_43|0/400|0|未开始|0/200|
|OBJECT_MLR_43|0/400|0|未开始|0/200|
|GOAL_ALIGN_41|0/400|0|未开始|0/200|
|GOAL_MLR_41|0/400|0|未开始|0/200|
|GOAL_ALIGN_42|0/400|0|未开始|0/200|
|GOAL_MLR_42|0/400|0|未开始|0/200|
|GOAL_ALIGN_43|0/400|0|未开始|0/200|
|GOAL_MLR_43|0/400|0|未开始|0/200|

总预算4800轮、1089600生产、240独立准入optimizer、0额外autograd、0训练采集、2400正式。每stage准入5次独立BC/IFA公式和完整state/序列化下一步精确，通过后恢复原state/RNG才生产，不套未来PASS。OBJECT每run105500、GOAL76100。最多2训练，独立c10d localhost:0+UUID；OBJECT六run训练→1200正式串行→GOAL六run训练→1200正式串行。评估完全沿用完整S的缓存前端、600步/5zero、0–19初态、每轨迹固定Normal、done置零、每vector新环境及benchmark=False。

只用本目录transport.py check_readonly.py取证→audit_progress.py NEW PREVIOUS→record_progress.py NEW。检查前核已有请求和root，不重复入口。任何异常先保留进程/原失败/冻结hash及已耗成本，不盲重启部分轮或环境attempt。当前现场更新不是科学接受。

仅自然root/controller.exit0且无自身进程后，唯一final_readonly.py实际CPU48CP/12完整恢复/2400初态noise/预算→audit_final_local.py。最终包装STATIC_ONLY，尚未执行，不得提前称PASS；错误保留原证据后版本化修复，不重训重评。FULL_S_BASELINE_BINDING/OUTCOMES是已接受完整S1200条，每case200各给两variant配对一次，非2400独立S；BASELINE_BINDING/OUTCOMES是历史Random，仅背景，切勿混淆。之后报告三方案及MLR-vsALIGN配对、所有seed/任务和硬件/开发集复用/预训练来源边界，不按成绩选优。

每次按根AGENTS用publish.py审查轻量证据并串行nonforce推MLR核remoteSHA与本地receipt。禁CP/model/state/RNG/latent/demo/cachepayload/凭据。服务器空间预算24GiB新峰+2GiB留存，root留1GiB+64MiB；不删除旧结果。无新的定时任务被创建；controller会自行运行完整有限队列。


# 当前任务检查与下一检查点

实际观察：2026-11-10T22:01:27.229126+08:00，PARTIAL。进度前缀审计和独立轻量记录复算通过。真实代码合同、训练/评估退出与准入记录已核；工程失败0。当前科学状态尚未最终接受。

|run|完整训练轮|正式回合|
|---|---:|---:|
|OBJECT_ALIGN_41|400/400|0/200|
|OBJECT_MLR_41|400/400|0/200|
|OBJECT_ALIGN_42|19/400|0/200|
|OBJECT_MLR_42|19/400|0/200|
|OBJECT_ALIGN_43|0/400|0/200|
|OBJECT_MLR_43|0/400|0/200|
|GOAL_ALIGN_41|0/400|0/200|
|GOAL_MLR_41|0/400|0/200|
|GOAL_ALIGN_42|0/400|0/200|
|GOAL_MLR_42|0/400|0/200|
|GOAL_ALIGN_43|0/400|0/200|
|GOAL_MLR_43|0/400|0/200|

当前检查范围：CURRENT_COMPLETE_TRAINING_WAVE。下一检查点：2026-11-10T23:21:27.229126+08:00（Asia/Shanghai）。按当前完整训练波次或整个套件正式评估余时估计，不因内部Task切换短周期检查。自身完成训练墙钟、纯epoch、借用吞吐与保存准备余量见CHECKPOINT_AUDIT.json；未实测部分不视为完成保证。

核验10份实际阶段准入、8份阶段CP记录绑定、2份完整训练账；正式0条记录及0个完成vector。初态/噪声身份采用记录交叉比较与噪声独立重生成，真实初态文件字节和实际CP加载待全队列结束后的CPU验收。

证据：check_readonly_20261110_220126。跟踪仅在当前会话执行，没有创建后台定时自动化。没有新增优化器更新、环境回合、服务器科学写入或重启已完成工作。每轮发布以PUBLICATION_RECEIPT为准。
