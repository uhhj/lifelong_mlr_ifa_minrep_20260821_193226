# 已授权接续：OBJECT结束后运行GOAL固定λ_R六run

用户2026-11-03明确：OBJECT跑完后把GOAL一起跑，同样协议、同样训练方式。该指令扩展后续范围，不中断当前OBJECT，不重跑已完成数据。本目录是持久排队协议，尚未构建、冻结或部署GOAL训练入口，不表示GOAL已启动。

新增最新完整S的GOAL λ_R=0.05、0.2，每档seed41/42/43；已有GOAL λ_R=0.1三个完整S只读复用。只改变保持合成项λ_R，不跑新的C/Q/旧机制，不按正式结果选λ、追加seed或重评。

每run用同seed官方GOAL初始化，Task6→9各100轮。GOAL原RA/manifest/mask/模型/优化器、实际接受更新线性学习率终点、warmup500不变。Task6无保持，Task7–9自己的成功缓存q、固定4槽分母、10步C/Q完整状态双候选、两epsilon0与探针规则沿用。各lambda的q=1参照使用同lambda，不把0.1旧轨迹当参照。

GOAL每轮批次数100/139/185/337，每run76100接受、142200生产物理optimizer调用（66100未选候选另含在生产总数内）、6700门控块。六run2400轮456600接受、853200生产物理、40200门控；自身546隔离optimizer/246准入autograd、最多216方向autograd另记。不能套OBJECT的633000/1171800/54600预算。

每阶段固定10训练采集初态20–29，成功缓存quota128、不补采，不复制别的run或0.1的缓存；总240训练采集。正式0–19、600steps/5zero、原配对Normal，每run200，总1200正式。已有0.1的600回合与两档配对，不视为1200独立基线回合。

前置边界：OBJECT所有训练和正式自然结束，完成OBJECT实际独立CPU验收、三档解释发布及自有环境hook安全清理后，才唯一启动GOAL；不得与尚未结束的OBJECT训练或环境重叠。若OBJECT最终外置包装出错，保留证据修包装，不重训重评。

GOAL使用独立目录/root/合同/根锁。候选目录 execution/20261103_lambda_r_goal_sensitivity，候选root S_LAMBDA_R_GOAL41_42_43_20261103_v1；部署前核唯一性，尚无活动PID/冻结合同。复用已审查的device snapshot/检查合并及两训练进程工程方式，显式c10d localhost:0独立UUID，禁止共享默认29400或--standalone。GOAL适配必须重新检查自身数据、lambda与完整状态等价证据适用范围，不能套OBJECT的全部GPU验证结论。

调度沿用用户最终授权方式：0.05按seed41/42/43优先，最多两个独立训练，空槽可接0.2 seed41；0.05三seed训练完成即优先600串行正式，已启动0.2 seed41可以继续；0.05正式全部完成后接0.2 seed42/43，最多两个训练，最后0.2的600串行正式。全部训练采集及正式共享准确GOAL目标路径的全局环境锁，不并行环境。

唯一部署前完成只读容量预检：按GOAL实际checkpoint/恢复备份/cache峰值及当前剩余空间计算，root保留1GiB/data保留2GiB；不足时先保留证据和说明需行动，不自动删除旧模型、扩盘付费或超容量启动。每阶段自己的GPU准入/COPY_EQ/BRANCH_COPY_EQ/CHECK_EQ实际通过。官方初态tensor/RNG须一致；跨GPU Task6比较记录真实结果，不伪称相等。3090迁移、开发集复用与预训练来源未闭合边界继续保留。

全部GOAL自然真实controller.exit0且无自身进程后，实际CPU24CP/6完整恢复/24cache teacher/1200初态noise/40200门控及预算验收，再与既有GOAL0.1配对，报告三档、新旧任务、SF/FS、所有负单元、训练代理、硬件/成本边界。不直接套OBJECT的scale特例；若有具体差异，独立核验并绑定该差异，不能通用放宽。

每次检查按根AGENTS审核轻量代码/报告/hash/scalars/outcomes，经publish_mlr_branch串行nonforce推MLR并核远端SHA和receipt；禁止模型/CP/state/RNG/cache/凭据。沿用同一个监控并更新活动目录，不重复创建旧监控。OBJECT完成不是整个新增授权任务完成；仅GOAL六run也独立接受、解释发布并清理自有hook后，才关闭该监控。
