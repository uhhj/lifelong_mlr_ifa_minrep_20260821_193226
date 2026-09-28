# LIBERO-GOAL seed42 MLR+IFA state_v2 最终执行报告

状态：**COMPLETED，独立产物验收 PASS**。四阶段各100个完整epoch，总计76,100个优化器步骤；四个epoch_index=99最终checkpoint均完整保存、可读取且在正式评估中严格加载；200条正式闭环记录，4×4矩阵有10个有效单元。正式运行无需中途恢复，训练、评估和汇总进程退出码均为0。

运行标签：`goal_seed42_mlr_ifa_paper_aligned_state_v2`。单训练seed=42，非多seed均值。

**按论文描述启用状态模态；官方checkpoint原始模态配置若缺失，仍记为未知。** 本机核验完整原始cfg.policy后，use_state确实缺失，因此pretrain_use_state=unknown、effective_use_state=true、DISCLOSED_GAP。未改写官方权重，不宣称精确恢复作者原始轨迹。

## 结果

每个有效单元20次闭环，以下为成功次数/20（成功率）；“—”表示未评估，CSV原值为NaN。

| 训练完成阶段 | 任务6 | 任务7 | 任务8 | 任务9 |
|---|---:|---:|---:|---:|
| 任务6后 | 13/20（65%） | — | — | — |
| 任务7后 | 14/20（70%） | 20/20（100%） | — | — |
| 任务8后 | 12/20（60%） | 20/20（100%） | 13/20（65%） | — |
| 任务9后 | 14/20（70%） | 20/20（100%） | 12/20（60%） | 17/20（85%） |

| 指标 | 本轮单seed结果 |
|---|---:|
| FWT | 78.75% |
| NBT | 1.1111 个百分点 |
| AUC | 78.4375% |
| FinalAvgSR（辅助） | 78.75% |

FWT按四个对角单元平均；NBT按前三个任务列的对角值减后续表现、先列内再列间平均；AUC按四个任务列从对角至最后阶段平均、再列间平均。NBT逐任务为−1.6667、0、5.0000个百分点，未截断负值。AUC逐任务为66.25%、100%、62.50%、85%。独立验收重新从200条记录计算，和metrics.json一致。没有把最后一行或下三角整体均值当作AUC。

[逐episode原始记录](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/evaluation/rollouts.jsonl) · [成功率CSV](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/evaluation/success_matrix.csv) · [指标JSON](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/metrics.json) · [独立最终验收](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/final_acceptance.json)

## 训练与时间

GPU：NVIDIA GeForce RTX 3090 24GB，单GPU/world_size=1。Python3.8.20、torch1.13.0+cu116、驱动535.230.02、MuJoCo3.2.3、robosuite1.4.0、robomimic0.2.0。完整环境版本与requirements.lock已存档。

| 阶段 | 开始（北京时间） | 结束 | 时长 |
|---|---|---|---|
| 正式特征导出 | 2026-09-27 23:34:03 | 23:42:30 | 8分27秒 |
| 四阶段训练 | 2026-09-27 23:42:30 | 2026-09-28 00:20:14 | 37分44秒 |
| 独立闭环评估 | 2026-09-28 00:20:14 | 01:33:41 | 1小时13分27秒 |
| 独立汇总 | 2026-09-28 01:33:41 | 01:33:42 | 1秒 |

正式导出至汇总总时长1小时59分39秒，不含前期软件修复及隔离smoke。最后验收时无残留训练/评估GPU进程。

| 任务 | 名称 | 新窗口/epoch | 旧窗口/epoch | batch/epoch | epoch | 总步骤 | 存储新窗口 |
|---|---|---:|---:|---:|---:|---:|---:|
| 6 | put_the_cream_cheese_in_the_bowl | 997 | 0 | 100 | 100 | 10000 | 499 |
| 7 | turn_on_the_stove | 889 | 499 | 139 | 100 | 13900 | 445 |
| 8 | put_the_bowl_on_the_plate | 904 | 944 | 185 | 100 | 18500 | 452 |
| 9 | put_the_wine_bottle_on_the_rack | 1969 | 1396 | 337 | 100 | 33700 | 985 |

每任务使用demo_0..demo_9完整演示，窗口长度8，每epoch每个拼接样本恰好出现1次、重复数0。旧样本实际比例由数据量决定，未设置固定30%或50% batch配额。存储比例0.5按窗口文件随机选择，最后buffer共2381文件，158,687,539磁盘字节；不等同于精选五条完整演示。

## 参数与修改

三个实际Hydra入口共享`++policy.use_vision=true ++policy.use_language=true ++policy.use_state=true`。最终三份resolved config均已保存，验收确认字段为布尔True，训练batch10、eval.n_eval20、gamma0.7。BriefPolicy字段只声明合同，正式latent全部从官方权重重新导出。

[导出配置](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/resolved_export.yaml) · [训练配置](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/resolved_adaptation.yaml) · [评估配置](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/resolved_eval.yaml) · [完整修改diff](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/source.patch) · [协议审计](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/protocol_audit.md)

主要修改：统一三入口配置；模态语义检查和合同指纹；权重前缀与完整加载校验；IFA自身语言正参考及样本/任务对平均；相似度等价计算避免大矩阵；按optimizer-step推进原500步warmup/linear scheduler；固定epoch99及可恢复checkpoint；事务式幂等buffer；独立闭环逐episode日志与严格汇总；真实HDF5句柄及spawn子进程ObsUtils初始化修复。纯latent训练使用0读取worker，真实数据导出保留12，已通过样本顺序/父Torch RNG对照。

CompletePolicy模型主体与固定commit相比无diff。沿用已有state_mod分支，token顺序language、agentview、eye_in_hand、state；状态输入joint_states(7)后gripper_states(2)，保持4×512维。空间前端及已有LoRA冻结，只有6层时间解码器与GMM头更新。

## 测试与产物完整性

- A/B/C/D真实模态测试全部PASS：第4 token与state_mod公式最大误差0；状态扰动引起第4 token最大变化0.0107689/0.1107986/0.4784026，前三token逐位不变；False软件负对照变化0；完整/潜策略GMM means/scales/logits误差均0。正式10个评估单元在线复核也全部PASS。
- 官方完整策略615键严格加载；每阶段100个下游键覆盖率100%，missing/unexpected为空，515个空间键未被阶段权重覆盖；12个状态编码/调制参数存在并兼容。
- 冻结哈希一致、两类可训练模块实际更新、2步warmup检查、checkpoint/RNG重载后下一步一致、buffer幂等性、读取worker顺序/RNG、真实12-worker HDF5读取均PASS。
- IFA混合batch旧样本首位/排列/空pair/多pair平均及梯度测试PASS；相似度恒等式测试误差4.65661e-10；已知矩阵指标及负NBT、不完整记录拒绝测试PASS。
- 隔离smoke执行600步真实闭环，success=false、无异常，仅用于软件验证，不计入正式200次。正式评估200条exception均为null。
- 最终只读验收复核4759个feature文件、各buffer代文件及4个checkpoint哈希，逐步学习率、400个epoch样本计数、连续parent哈希、RNG/optimizer/scheduler保存内容、20个唯一初态与worker seeds、正式矩阵和指标，全部PASS。原始/data/mlr_ifa/src仍clean。

[模态测试](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/modality_tests.json) · [模态合同](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/modal_contract.json) · [特征清单](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/feature_manifest.json) · [数据清单](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/data_manifest.json) · [权重来源清单](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/pretrained_manifest.json)

## Checkpoint与路径

服务器正式产物根目录：`/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/run`。新缓存：`/data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/run/features`；空初始buffer：`run/buffer`；不可变buffer各代：`run/buffer_manifests/after_task_6..9`。旧缓存、旧checkpoint与失败准备目录全部保留，没有改标签混用。

四个文件均位于上述run目录的`checkpoints/`：

| 文件 | 累计global_step | SHA-256 |
|---|---:|---|
| final_task6_epoch99.pth | 10000 | `dd5a6a80bcc1cfb2d16a95f1dc8df226e8412b1a48c8f30e5857605b672e07ff` |
| final_task7_epoch99.pth | 23900 | `146f11436350f8643709c49b3bbb97b42952539aed05ba3a01d529299074b6d2` |
| final_task8_epoch99.pth | 42400 | `aaddd3480aa889244e3ed29dd3c84a05a62784d7172064584188886b24dad588` |
| final_task9_epoch99.pth | 76100 | `59f2bf55f55e9e0ec4f3a44746e02dcb07ba1447cd06bdc9ab0b8a0154046e29` |

权重仅保留服务器，未上传或下载到本地。本地`final_evidence/`保留495份初次同步轻量证据及本次最终验收/报告。

固定commit：`a721807f2f7fa7f9833dcc85c592d78e9accd00b`。

官方GOAL权重SHA-256：`241d3370acfc04d694df04256956a97aa13cfdc2e651cf258d338f8e72988be7`。

完整source.patch SHA-256：`57e9566ab44c74d88563892080e1dd34304e36e4204d7acd8ede088b6c2363a0`。

共同modal fingerprint：`4c0a99a8c2f88695fc91292748e4f809a3bf38866bb78dc32d8fc7d0f5001ecc`。

## 实际执行命令

```bash
source /data/mlr_ifa/activate.sh
cd /data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/source
bash scripts/run_goal_seed42.sh --mode full \
  --pretrained /data/mlr_ifa/checkpoints/goal_multitask_model_ep96.pth \
  --data-root /data/mlr_ifa/datasets \
  --bddl-root /data/mlr_ifa/src/libero/libero/bddl_files \
  --init-states-root /data/mlr_ifa/src/libero/libero/init_files \
  --run-dir /data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/run \
  --smoke-dir /data/mlr_ifa/experiments/goal_seed42_mlr_ifa_paper_aligned_state_v2/smoke_v2_release3
```

该命令已完整执行，不能在已有目录重复full。三个真实Python/Hydra子进程的完整参数、开始结束时间和退出码见[commands.log](D:/论文课题研究/MLR+IFA/execution/20260927_state_v2/final_evidence/run/logs/commands.log)。本轮最终验收通过单独的evidence/verify_final_artifacts.py只读检查，没有追加训练或rollout。

## 仍存在的公开协议缺口

1. 官方checkpoint完整cfg.policy中缺use_state；权重存在和状态依赖通过不能证明作者预训练时实际启用了状态。本轮True为用户明确决定，原设置仍unknown。
2. 官方完整六任务预训练流程未公开；本轮从官方GOAL权重开始，未重建预训练。checkpoint记载seed42，但不据此声称论文正文列出了42或独立恢复全部预训练轨迹。
3. 500步warmup来自发布配置，scheduler按step推进、阶段终点0会改变发布脚本优化轨迹；论文没有公开完整曲线，不能宣称精确恢复。
4. epoch99固定最终模型、评估RNG隔离、20个初态/600动作等本轮补充约定已预先披露；原模型导出train模式与增广保持。配置中的augmentation prob字段照录，实际构造调用保持源码network_kwargs行为，不能仅凭prob字段推断所有增广实际应用概率。
5. 本轮仅单训练seed点估计，不代表论文三seed均值或方差，不以论文数值作为通过阈值；闭环结果未用于选择开关、采样、seed或checkpoint。

本轮可称为“paper-aligned implementation with disclosed gaps”。没有未解决的技术阻塞。定时检查在最终报告和产物验收完成后关闭，保留自动化记录供追踪。
