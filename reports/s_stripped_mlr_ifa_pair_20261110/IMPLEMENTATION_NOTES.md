# 实现说明：完整 S 剥离后的 MLR / ALIGN 配对

## 文件职责与来源

- `case_runtime_template.py`：从完整 S runtime 保留 `build`、`hidden`、`masked_bc`、`angular_mean`、训练状态 capture/restore 和 sampler/setup；改为本 case 的 Random manifest、current ALIGN 标签、Random IFA 与配置。使用原通用 CPU 状态复制，没有引入新的 device-snapshot 工程优化。
- `updates.py`：直接从完整 S 的更新顺序剥离为 `bc + .1 * ifa`，每 batch 一次 backward、clip、AdamW.step、scheduler.step 和全局 step 增加。通过 `INPUT['variant']` 控制 BC：ALIGN 传 `True`，MLR 传 `False`。删除而非置零 E/R/q/gate 调用；不存在专家/保持额外前向、probe、teacher 或采集。
- `admission.py`：每个实际阶段五次隔离更新，十二个 run 共240次，另写 `direct_reference` 显式组合保留项，核首两步、序列化下一步以及完整状态恢复。仅模块静态可审不能证明实际 GPU PASS。
- `train.py`：100 epoch/stage 的 full-S 生命周期；生产与准入独立计数，保存每 epoch 恢复状态、阶段模型、输入/调度/顺序证据。只允许新运行，没有自动恢复或覆盖完成结果入口。
- `prepare_local.py`：本地文本适配。逐 case 检查核心 Random 与 S 配置相同，复制已绑定 Random manifest 和对应 IFA 信息，生成布局和 source hashes。正式启动后的禁止重构条件由冻结/部署合同约束。
- 各 case `evaluate.py`：两个 variant 均使用完整 S 原文件，允许的差异仅 `runtime → case_runtime`、模型 arm `S → A`、本地冻结评估配置路径；其余评估计算、Normal、done 动作和环境生命周期保留。

新模板不得通过旧完整 S 的部署器、旧训练入口或上一轮 reference/global-Normal evaluator 启动。旧外部 `caet_v3_exp`/`coreset_exp` 库仅作为完整 S 已绑定的构造与公式依赖，文件 SHA 必须实际核实；模块中存在旧代码不表示可触发附加机制，生产入口调用链以新 `updates.update` 为准。

## 运行接口

远端 `cases/{OBJECT|GOAL}_{ALIGN|MLR}_{41|42|43}/` 包含 `INPUT.json`、`SOURCE_HASHES.json`、`resolved_random.json`、`resolved_eval.json`、`layout.py`、`case_runtime.py`、三个共享训练文件、评估文件、`manifests/task6..9.json` 与 `ifa_task6..9.json`。`ARMS='A'`，`OUTPUT['A']=DATA_PRIVATE/{case}/A`。

训练命令由唯一新 controller 以独立 torchrun rendezvous 启动 `train.py A`；最多两训练。正式评估运行 `evaluate.py`，套件内各 seed 串行。OBJECT 六个 run 全部训练和 1200 正式闭环结束后才启动 GOAL 六个 run。训练中没有 collect 子进程，没有 site-packages hook，也没有历史采集锁依赖。

`case_runtime` 导出与训练模板一致的 `Dataset/make/setup/capture/restore/equal/initialization`、损失 helper、`CALLS/set_count_mode`、布局常量与读写/哈希函数。AdamW hook 计量实际完成的 `optimizer.step`；`COUNT_MODE` 区分 `admission` 与 `production`。`restore` 恢复模型、优化器、scheduler、梯度、模式、requires_grad、Python/NumPy/Torch RNG 与 step；全局诊断调用账不会随恢复撤销。

## 数据与训练顺序

`Dataset` 按每行 Random manifest 的 `relative_path` 读取原只读 latent，并按 task/sample 查找 current 标签，核 sample SHA 和 task/demo/u/bc_valid。返回原 feature、task 与 ALIGN 的 `next_action/causal_bc_valid/v3_index`。每阶段的 `VALID` 来自实际 Random manifest，不能使用 RA 计数。MLR 与 ALIGN 使用相同样本人口和排序；MLR 的有效 BC 计数为 `POP`、无效为0，ALIGN 的有效计数为 `VALID`。

`setup` 使用 `DistributedSampler(num_replicas=1, rank=0, shuffle=True, seed=0)`、batch10、num_workers0、persistent_workers=False，调用原 `a.start_task(-1, loader)`。该调用更新阶段 optimizer/scheduler；全局 `a.step` 不应重置。生产每 epoch 核累计 step 等于已完成生产调用，最终精确到 OBJECT105500/GOAL76100。每阶段实际核 LambdaLR 的 total_steps 和 warmup500。

语言参考由本阶段 Random 数据中各 task 的首个实际样本语言槽构造，IFA `selected_pairs` 读取本 case Random IFA 文件。`TRAIN_INPUT_A_N.json` 和 checkpoint 记录 manifest/IFA SHA。每 epoch 记录实际 sampler index SHA、开始/结束 LR、batch 数、valid/invalid 数、损失累计和真实用时，便于独立复算。

## 准入的确切五次调用

1. 实际 `update` 首步，捕获状态 one。
2. 恢复阶段初态，`direct_reference` 首步，与 one 的统计与全状态精确比较。
3. 从 one 执行实际第二步，捕获状态 two。
4. one 序列化至内存并 CPU 加载后恢复，重放实际第二步，与 two 精确比较。
5. 再恢复序列化 one，执行直接式第二步，与 two 精确比较。

最后核空 BC 掩码产生可微零值并恢复完整阶段初态。MLR 则核全无效 ALIGN 标签不会过滤原始 BC，即输出掩码全1且 BC 有限。该检查有额外前向，但不执行 optimizer 或额外 `autograd.grad`。阶段准入样本按确定性索引覆盖可用的历史有效、当前有效、无效窗口，不使用正式评估环境，也不改变生产 sampler；恢复包括 RNG。成功文件 `ADMISSION_A_N.json` 明确记录 `optimizer_calls=5`、`extra_autograd_calls=0`、`production_calls_added=0`、`environments=0`。

这项准入证明同一实际运行状态下，两 variant 各自的新更新与显式保留公式的一致性及恢复可靠性；它不证明历史完整 S 的整个 observe 路径、跨硬件训练或闭环结果逐位等价。

## 初始化和 Task6 观察

初始化证据 `INITIAL_A.json` 比较官方模型 tensor SHA 与绑定的非 CUDA RNG，记录实际硬件、PyTorch/CUDA/cuDNN、TF32 配置和 CUDA RNG 的历史相等性。CUDA 状态仅记录，不使用未经验证的跨设备断言。

Task6 保存后，`TASK6_S_COMPARISON.json` 只读加载 `INPUT.full_s_task6_path` 并核绑定 SHA，逐 tensor 记录精确相等、差异张量及最大绝对差。不会增加 optimizer、模型前向或环境调用，核读取前后 RNG 不变。与历史 S 不相等不触发重跑；历史 S Task6 可能包含已被删除的 E 项，差值没有单因归属。

## Checkpoint 与实际证据

每 epoch 私有 `recovery.pth` 含完整状态、阶段、next_epoch、全局步数、全部轻量 history 与输入哈希；前 3 个阶段保存模型 `stageN.pth`，stage9 与最终 recovery 硬链接，避免重复私有 payload。所有模型保留原 `state_dict` 接口，完整 S evaluator 严格读取 100 个 policy 键。模型与状态不发布。

轻量证据包括 `TRAIN_STATUS`、`TRAIN_INPUT`、`SCHEDULE_A_N`、`INITIAL_A`、`ADMISSION_A_N`、`EPOCHS_A`、`STAGE_A_N`、`TASK6_S_COMPARISON`、`TRAIN_A`、真实 launcher/评估退出、`EVAL_ADMISSION`、`EVAL_EVENTS`、`OUTCOMES` 和 controller 事件。十二份完整恢复/逐步/LR/order 验收与正式2400条配对结果在全队列结束后独立执行；不能把 `PASS_EXECUTION_PENDING_FINAL` 当最终接受。

## 评估和比较口径

完整 S 的逐轨迹 noise、600步、5zero、done动作归零、每vector创建关闭环境、cached历史与严格CP加载都保留。评估前的 cache parity 仅按原 `1e-4` 容差；逐条 outcomes 核初态/noise/env_seed/attempt/CP身份与实际200回合。

完整 S 比较须读取其已接受实际正式结果；当前历史 `BASELINE_*` 文件来自旧 Random，不与完整 S 混称。新结果对完整 S 的比较反映整个附加机制组合删除后的变化，含 RA→Random；对旧 Random 的比较还受到历史 evaluator/硬件差异限制。新 MLR 与 ALIGN 的直接配对同时核相同 Random manifest/IFA/初始化和逐轨迹 noise。所有 seed、正负单元与成本同时保留，禁止通过重评、选checkpoint或舍弃seed改善结果。

## 审查边界

`PRELAUNCH_REVIEW.json` 是 `STATIC_NOT_EXECUTED`。它记录静态源码/接口/预算审查，不代替实际容量、新目录唯一性、冻结合同、当前硬件、每阶段 GPU 准入或正式评估的运行证据。部署与科学运行由主任务统一执行，本子任务未启动验证、训练或环境，也不独立 push。

## 双方案预算与识别

`INPUT.variant` 必须为 `ALIGN` 或 `MLR`，内部模型 arm 仍统一 `A`，防止评估推理代码产生方案分叉。MLR 的独立直接式使用 `actions.squeeze(1)[:, -1]`、全1有效掩码；ALIGN 使用 `next_action`、`causal_bc_valid`。两者 IFA、batch顺序、optimizer/scheduler/clip/loss_scale及正式评估相同。十二个 run 合计4800轮、1089600生产optimizer、240隔离optimizer、0训练采集、2400正式；OBJECT为633000生产/1200正式，GOAL为456600生产/1200正式。
