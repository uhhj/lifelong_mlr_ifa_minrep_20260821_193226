# 完整 S 剥离配对实验：MLR+IFA 与 MLR+IFA+ALIGN

## 授权、范围与状态

本轮执行用户 2026-11-10 的明确要求：“重跑一遍，把完整 S 中的其他机制全部去掉，仅留下 MLR+IFA+ALIGN”。用户随后追加纯 MLR+IFA 对照。OBJECT、GOAL 各运行 MLR+IFA 和 MLR+IFA+ALIGN 两方案，每方案 seed 41、42、43，共十二个独立 run。旧完整 S、旧 MLR+IFA、上一轮 MLR+IFA+ALIGN、支持度消融及所有失败证据保持原样，不重训、不重评，也不覆盖旧目录。

新工作目录为 `execution/20261110_s_stripped_mlr_ifa_align`，候选远端根为 `/data/mlr_ifa/experiment_workspaces/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1`，模型私有根为 `/data/mlr_ifa/experiments/S_STRIPPED_MLR_IFA_PAIR_OBJECT_GOAL_20261110_v1`。正式启动以唯一部署记录、实际 PID 和冻结合同为准；本协议及静态审查本身不证明已经启动、通过 GPU 准入或接受实验。

## 保留项与删除项

生产更新源为已接受完整 S 的训练运行骨架，OBJECT 对应 `20261028_support_gate_full_s_object/seed41..43`，GOAL 对应 `20261029_support_gate_full_s_goal/seed41..43`。不是给上一轮训练脚本改名，也不是在完整 S 中只把保持系数设零。

两方案的每个 batch 均只有 BC 与 IFA 两项，区别仅在 BC 目标与有效性处理：

\[
L_{\mathrm{ALIGN}} = L_{\mathrm{BC,ALIGN}} + 0.1 L_{\mathrm{IFA}},\qquad L_{\mathrm{MLR}} = L_{\mathrm{BC,original}} + 0.1 L_{\mathrm{IFA}}.
\]

两方案保留完整 S 的模型构造、官方初始化、特征解释、`hidden → masked_bc → IFA` 顺序、`loss_scale`、backward、有限值检查、原梯度裁剪、AdamW 和 scheduler 更新语义。`INPUT.variant` 冻结为 `ALIGN` 或 `MLR`。ALIGN 调用 `masked_bc(..., True)`，使用窗口末端的真实下一步动作与有效性标记；无效窗口的 BC 不纳入有效样本均值，有效样本为空时使用可微零值。MLR 调用 `masked_bc(..., False)`，使用原始 `actions.squeeze(1)[:, -1]` 目标且所有样本有效，不因 ALIGN 标签而过滤。两者 IFA 仍只在本阶段当前任务样本上按原公式计算，不因 BC 无效而任意改变其掩码。

删除 RA-Coreset 选样，使用固定的 Random 成员和顺序。完全删除当前执行专家 E、历史保持 R、成功缓存与教师、支持度 q、动作归一化 scale、双候选更新与门控、门控探针、方向诊断、成功轨迹采集。每个生产 batch 只执行一次 optimizer 更新；不能以原 C 分支或 `retain_weight=0` 代替，因为完整 S 的执行 E 项仍会保留。

## Random 输入、ALIGN 标签与 IFA 绑定

每个套件/seed 的两个 variant 共用同一 Random 输入；十二个 run 各自绑定 `20261109_mlr_ifa_align_baseline/cases/{SUITE_SEED}` 已审计的 Random manifest，逐行保留成员、顺序、sample/task/demo/u、文件 SHA 和 `bc_valid`。每行通过其原 task/sample 身份读取完整 current 标签，不借用 RA 的历史子集。当前与历史样本的原 latent 文件、ALIGN 标签及官方模型只读复用。

IFA 任务对信息绑定该 Random 人口对应的 `ifa_task6..9.json`；不能换成 Random 数据却继续使用 RA IFA 信息。每阶段 manifest 与 IFA 文件 SHA 写入实际训练输入证据和 checkpoint；父源与外部文件绑定见 `PARENT_BINDINGS.json`、`EXTERNAL_FILES.json` 及各 case 的 `INPUT.json`、`SOURCE_HASHES.json`。有效窗口数以各 seed 的实际 manifest 为准。

| 套件 | Task6/7/8/9 窗口人口 | 每轮 batch 数（batch=10） | 每 run 更新 |
|---|---|---|---:|
| OBJECT | 1562 / 2261 / 3081 / 3620 | 157 / 227 / 309 / 362 | 105500 |
| GOAL | 997 / 1388 / 1848 / 3365 | 100 / 139 / 185 / 337 | 76100 |

## 训练、调度和成本

每 run 按 Task6→7→8→9 顺序训练，各 100 epoch，共 400 epoch；十二 run 共 4800 epoch。使用完整 S 的阶段初始化 optimizer/scheduler 方式、500 步 warmup 和该阶段实际 `100 × batch 数` 的线性调度终点。每阶段实际核调度闭包；`DistributedSampler` 固定 seed 0，每 epoch `set_epoch(epoch)`，并记录索引顺序 SHA。保留 full-state 恢复点，但禁止发现错误后盲重启部分轮。

最多两个训练进程并行；每个训练 launcher 使用独立 c10d endpoint/标识，不共享默认 rendezvous 端口。顺序为：OBJECT 六个 run 训练全部完成→OBJECT 1200 回合正式闭环评估→GOAL 六个 run 训练全部完成→GOAL 1200 回合正式闭环评估。套件内正式评估串行，不跨套件抢跑，不启动训练期环境采集。

| 类别 | OBJECT 六个 run | GOAL 六个 run | 总计 |
|---|---:|---:|---:|
| 生产 optimizer 更新 | 633000 | 456600 | 1089600 |
| 自身阶段隔离准入 optimizer 更新 | 120 | 120 | 240 |
| 额外 `autograd.grad` 诊断 | 0 | 0 | 0 |
| 训练期环境采集 | 0 | 0 | 0 |
| 正式闭环回合 | 1200 | 1200 | 2400 |

240 次隔离更新是每 run 四阶段各 5 次，必须与生产分开记录。准入期间正常 backward 属这 240 次隔离更新，不冒称没有梯度计算。评估缓存检查的前向调用另记，不是环境回合。异常发生时保留已开始、已完成及未知部分的真实成本，不把未完成部分记为零。

## 每阶段自身准入与历史比较

生产开始前，用本阶段真实 Random 样本执行 5 次隔离更新：实际更新与独立显式 `BC + 0.1 IFA` 组合式首步比较；第二步结果与序列化/加载后的下一步重放比较；再与直接式第二步比较。比较完整 model、optimizer、scheduler、grads、train/eval 模式、requires_grad、RNG 和全局 step，最后恢复准入前状态。ALIGN 无效 BC 掩码的可微零值也必须通过；MLR 同一个全无效 ALIGN 标记输入仍必须采用全1 BC 掩码，表明原始 BC 不受该标签过滤。参照名称为 `direct_bc_ifa`，不能声称已经与原 `V3MLR.observe` 的完整调用链做了逐位验证。

官方初始模型 tensor 及非 CUDA RNG 必须匹配绑定证据；CUDA RNG 与当前硬件实际记录，历史相等性单独披露，不把跨设备状态的相等当作未验证前提。Task6 保存后只读比较完整 S 的对应模型逐 tensor 实际相等与差值，不以数值不等作为重跑或停止条件。完整 S 在 Task6 已可能含当前执行 E 项，故差值不能仅归因于硬件，更不能预称应当相等。

未开始的阶段没有准入结论。静态审查不替代实际 GPU 准入。任何失败先保留证据、hash、进程/退出、完整恢复状态与已耗成本，禁止覆盖失败或盲目重复实验入口。

## 正式评估严格继承完整 S

评估程序复制各 case 的完整 S `evaluate.py`，仅修改 runtime 导入名、模型 arm 名和本 case 冻结评估配置路径，具体父源 SHA 与限定差异见 `PARENT_BINDINGS.json`。保留完整 S 的单帧视觉 latent 缓存、最多 8 帧历史、argmax GMM 分量、逐轨迹配对 Normal、done 轨迹动作归零及每组环境生命周期；不使用上一轮 global-RNG Normal 的评估实现。

每个 stage6..9 评估所有已学 task6..stage，形成 10 格三角矩阵；每格初态索引 0–19，两个各 10 回合 vector。每 run 共 200 回合。每回合原始最长 600 步，正式动作前 5 次零动作；每个 vector 重新创建并关闭环境。轨迹 Normal 由 `20261024 + SEED*1000 + taskid*100 + initial_index` 单独生成 `600×7` float32 eps。同 task/seed/initial 的 noise 在不同 stage 及旧完整 S 比较中可实际逐项配对。

正式评估前实际检查缓存与整段前向在历史长度 1–8 的一致性，沿完整 S 的 `atol=rtol=1e-4`。这是指定数值容差的局部检查，不是跨硬件闭环逐位保证。每回合记录 checkpoint、初态文件/初态数组/noise/action-stream SHA、环境 seed、success、steps 和 `attempt=1`。不补采、不重评、不按测试成绩选 checkpoint 或选择随机种子。

## 最终接受与解释

仅本新根的真实 controller 退出 0 且无自身训练/评估进程后，才进行最终只读 CPU 与本地复算：十二个 run、48 个阶段 checkpoint、十二份完整恢复状态、400 epoch/run、实际 scheduler/顺序/准入、1089600 生产与240隔离调用、0训练采集、2400正式及逐轨迹初态/noise/attempt完整性。无保持或成功缓存，不能沿用旧 S 的 teacher/cache、q、gate 或采集预算验收项。

旧完整 S 的实际 1200 正式结果仅只读复用；两方案分别与同一旧 S 对齐，共形成2400条配对，但不是2400个独立旧 S 回合。新 MLR 与新 ALIGN 也在同套件/seed/初态/noise上直接配对。旧 S 按套件、seed、stage、task、initial 对齐后比较；完整 S 的绑定必须明确指向其已接受源。当前从上一轮复制的 `BASELINE_BINDING.json` 与 `BASELINE_OUTCOMES.json` 是历史 Random/MLR+IFA 背景，不能误称它们是完整 S，也不能把不同评估实现当成完全受控的 ALIGN 单因素因果证据。

报告同时给出两个套件、三个种子的 AUC/FWT/NBT/OldFinal/NewFinal/FinalAll、任务矩阵、配对成功/失败变化和成本。既有 S 与本轮硬件及执行时期可能不同；对旧 S，本轮还同时删除多个机制并将 RA 换回 Random，不能将全部差值解释为某一个机制的独立效应。新 MLR 与新 ALIGN 使用同一运行实现、Random输入、初态和评估协议，可用于考察 ALIGN 的目标/有效性处理差异，仍需报告逐 seed 实际结果及其不确定性。保留开发集多次使用、预训练来源未完全闭合等已有事实，不宣称独立确认或全局无泄露。不得因负结果加实验、扩 seed 或选择性丢弃单元。

## 发布与存储

冻结后不修改生产科学源；包装错误在保留原失败后版本化修复。容量以实际 preflight 为准，不删除旧模型、结果或当前依赖，不付费扩容。每次检查和最终结果按根 AGENTS 经审查发布当前相关代码、报告、hash、标量及 episode outcomes，串行 nonforce 推送 MLR 并核 remote SHA/本地 receipt。禁止发布模型、checkpoint、state/RNG payload、latent/demo/cache、凭据或未审 bundle。运行中统一标注 PARTIAL 与实际观察时间，发布不等于实验接受。
