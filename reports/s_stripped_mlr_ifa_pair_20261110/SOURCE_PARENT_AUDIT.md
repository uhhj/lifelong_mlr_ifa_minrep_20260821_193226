# 从完整 S 剥离为 MLR+IFA+ALIGN：只读父源审计

本文件只读审计现有代码，不代表新运行或准入已经执行。原文件未修改；没有训练、模型前向或环境调用。源码 SHA 见同名 JSON，发布由主任务统一完成。

权威完整 S：OBJECT 为 `execution/20261028_support_gate_full_s_object/seed41..43`，已接受根合同 `8594c0b609b8bf948e7914fbf071567ed991d1698a9d8357b90fcb03dec3573d`；GOAL 为 `execution/20261029_support_gate_full_s_goal/seed41..43`，已接受根合同 `e6a2242424286d4de4a6c5621037ccd34946e0593a2dd040e0136a78d77cb032`。OBJECT 最终自然成功源为旧工程 copy/controller，原报告保留；此次不运行这些入口。

## 最小科学剥离

完整 S 的 `updates.py:8–10` 已包含需要保留的 `hidden → masked_bc(aligned=True) → IFA` 路径。新损失应精确为 `bc + .1 * ifa`，其余继续一次 backward、原 clip、一次 optimizer.step 和 scheduler.step。

`updates.py:13–24` 的专家前向、保持前向及其模式切换/随机状态保存均应删除。第34行 `.1*extra` 是执行专家 E，即使 `retain_weight=0` 仍存在。因此不能直接选择 C 分支、把 R 系数设零，或仅调用 `legacy=True` 来冒充纯 MLR+IFA+ALIGN。

`train.py` 的 memory load、scale_for、setup_probes、gate_admission、directional_diagnostic，及第89行起双候选复制/恢复/choose全部移除，改为每batch单更新。第129行的collect子进程及40次成功轨迹采集全部移除。不得加载旧成功缓存或生成teacher。没有E/R以后 `scale_for` 无科学用途；不要继承旧的动作scale 1ULP特例。

## 移除RA必须同时处理输入和IFA

`runtime.py:37` 的Dataset绑定RA manifest与effects/memory标签；需要换成已核实的各seed Random成员与顺序。旧样本标签可以通过其原task和sample身份取对应完整current标签；不能把RA memory子集改个名字。

`train.py:62` 读取 `ra_coreset/ifa_task{stage}.json`，移除RA后也必须改为Random人口对应的IFA任务对信息，或用原 `build_ifa_similarity_matrix` 对绑定Random输入重新只读生成并核对。仅换dataset而继续使用RA的IFA-info不完整。

语言参考可沿用S的 `runtime.setup` 机制（每task实际样本的语言槽），但需用Random自己的成员核验其稳定性；每stage的population/valid/old计数必须按新manifest计算，不沿用S布局中的RA VALID值。

## 初始化与实际准入

S `runtime.make` 使用 `caet_v3_exp.train.build`，其 `V3MLR` 继承PaperMLR且没有新增构造器；`VARIANT=ALIGN`不创建TAC辅助头。`configure_effect` 不必调用，因为它读SUPPORT并初始化不必要的教师字段；S原make也只设置refs、统计列表和disable_all=False。

可在各stage实际Random batch上，用保存/恢复后的完整state，将剥离更新与底层 `V3MLR.observe(..., VARIANT='ALIGN')` 比较。后者的 `native=bc+.1*ifa` 和ALIGN分支提供独立参照；需包含`v3_index`，并设置disable_all=False和caet_epoch_statistics。全状态、下一步重放及恢复初态必须实际通过，不提前套PASS。把新增准入调用单独记账。

官方初始tensor及RNG应与同seed完整S/官方模型实际比较。新Task6最终模型不必等于历史Random，因为ALIGN BC不同；也不能拿历史训练结果直接代替新Task6。

## 评估必须沿完整S

保留完整 S `evaluate.py`：单帧latent缓存，完整历史最长8帧；按 `20261024 + SEED*1000 + taskid*100 + initial_index` 生成每轨迹600×7 Normal eps；argmax GMM分量采样；done轨迹动作置零；每10回合组新建并关闭环境；600步、5zero；初态0–19；四阶段三角10格。

runtime显式cuDNN benchmark=False也属于这一执行路径。仅改根/模型路径、case命名和真实退出适配，不复制上一轮Random global-RNG/done/环境生命周期实现。缓存容差仍为原S的atol=rtol=1e-4；此标准不是闭环逐位保证。逐轨迹noise与旧S应实际SHA配对。

每run400epoch，单候选OBJECT105500、GOAL76100，六run总544800更新；不继续S双候选1012500物理预算。六run1200正式回合，训练期环境0。Random/ALIGN都已有明确身份数据，可复用只读输入而不重跑旧结果。

## 新实现与路径依赖复核（2026-11-10）

已独立静态审阅 `train.py`、`updates.py`、`admission.py`、`case_runtime_template.py`；本次没有导入GPU模块、训练或环境调用。各文件SHA及Python3.8语法检查见JSON。生产目标仅为BC+0.1IFA，E/R/q/双候选/在线采集均已从调用链移除；Task6也移除S的当前执行E项，因此Task6与旧S模型的比较仅记录事实，不把差异当作失败或单一原因证据。

实际准入实现采用独立显式BC/IFA组合 `direct_reference`，并未执行前文建议的另一种 `V3MLR.observe` 参照。每stage计划5次隔离更新，验证首两步完整状态和保存/加载后的下一步重放，最后恢复生产初态；这些目前是待实际执行的检查，不能称GPU准入已经通过。AdamW hook同时记录生产和直接式调用，capture/restore覆盖模型、动量、scheduler、梯度、模式、requires_grad、RNG及累积step。

`capacity_readonly_20261110_202315/EVIDENCE.json`提供了六case实际父源build/model文本与配置。OBJECT41的build使用固定OBJECT(0)，其余五case使用cfg的benchmark_name和task_order_index；必须各case使用自己的已冻结源路径。common的CORESET_OUT仅在导入时绑定旧O；build不通过O读写文件，其构造链没有E/R或辅助头构造器。BriefPolicy从官方c.pretrain读初态，随后strict_downstream精确复核下游key。保留旧align_parent/align_source和模型/data/optimizer/IFA配置，只把experiment_dir、run_root、buffer_dir改至新输出；不调用原main/store/configure_effect/finish_epoch。

Random旧buffer文件由`tN__`前缀解析task，当前features由父目录解析task；已增加实际getitem task与标签task相等断言。标签按原task和未加前缀的sample身份读取，样本SHA与标签sample_sha256、task/demo/u/bc_valid逐项匹配。setup使用seed0的DistributedSampler，每epoch set_epoch；实际LambdaLR函数闭包含total_steps和warmup_steps，已改成精确核100×len(loader)与500，不沿用双候选长度。初始化仍须实际tensor和非CUDA RNG相等，历史CUDA RNG另行记录。

新增publish.py仅处理本目录经语法/大小/敏感字段扫描的轻量文本，并限定新reports前缀、MLR分支与既有远端；排除部署bundle、receipt、缓存目录和所有二进制后缀。尚未执行发布，交由主任务串行执行并核remote SHA。

## 最新追加的成对纯MLR范围

用户最新追加纯MLR+IFA，当前范围扩为OBJECT、GOAL各seed41/42/43的MLR与ALIGN两种配置，共12run；先前六run预算描述仅为当时范围。当前4800轮、1089600生产optimizer、计划240自身准入optimizer、2400正式、0训练采集。

纯MLR的定义与原 `BasePolicy.compute_loss → BriefPolicy.forward → GMMHead.loss_fn` 对齐：目标为 `actions.squeeze(1)[:, -1]`，全batch的NLL均值；S `masked_bc(aligned=False)`使用同一末端action和全True mask，实际loss_coef为1。ALIGN使用next_action和causal_bc_valid，两者IFA不变。CAET_VARIANT的父类导入常量仍为ALIGN，不会使纯MLR带入ALIGN目标，因为实际生产唯一分支读取INPUT.variant；不调用父类observe/configure_effect。

已静态逐对核六个suite/seed的Random manifest与IFA文件字节相同、初态和标签绑定相同、除输出路径外配置相同、train/update/admission源码相同。MLR准入还检查将ALIGN有效标记全部设零后仍使用全样本；实际GPU准入均待执行。旧S的1200唯一正式记录另存FULL_S_BASELINE_BINDING/OUTCOMES，逐fraction复算吻合原验收；每case200回合分别复用于两种新配置，不能计作2400独立S回合。

发布前缀已改为 `reports/s_stripped_mlr_ifa_pair_20261110`；本地目录名保留历史名称。首次review尝试遇GitHub TLS握手失败且无stage/push；已将--review-only限定为本地检查，实际发布仍需真实远端检查。随后本地大小/语法/敏感模式审查通过，具体文件清单由PUBLICATION_REVIEW记录，最终发布须主任务串行执行。
