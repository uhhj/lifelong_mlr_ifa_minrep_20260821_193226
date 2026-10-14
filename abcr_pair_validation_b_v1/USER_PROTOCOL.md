# ABCR 实验 B：物理状态与未调制视觉特征复核（预注册协议 v1）

**性质**：阶段 B 是离线候选关系审计，不训练 ABCR，不等价于真实机器人闭环成功率评价。不能通过在 Task 7/8/9 上调整阈值让候选数变多来宣称实验成功。

## 1. 先验证 HDF5 与 `.pt` 的逐窗口映射

在原始 LIBERO HDF5 文件上运行：

```bash
python abcr_experiment_b.py inspect-h5 --h5 /path/to/libero_object_dataset.hdf5 --limit 150
```

确认每个 `.pt` 记录对应的 HDF5 文件、任务、`demo_id` 和最后动作的时间索引 `t`。**不得假定 `.pt` 文件的编号是轨迹编号或时间索引**。

建议依据缓存中的完整动作序列（例如 `[1,8,7]`）与各条原始演示的滑动/填充动作序列逐元素对照。只有唯一匹配、最后监督动作一致、窗口填充规则一致，并且重新提取相应视觉特征不产生异常的记录，才可标记 `alignment_verified=True`。若同一动作窗口重复出现而匹配不唯一，该记录必须标记为 false 或剔除；不能用文件排序猜测映射。动作序列仅能作为必要检查，不能单独证明像素观测对应关系。

原始 HDF5 的观测字段和四元数排列需现场检查。不可假定一定包含 `robot0_eef_pos`、`robot0_eef_quat` 等键；若无法恢复末端姿态，就不应声称完成了本物理状态门控实验。

## 2. 冻结的物理状态距离与 τ_s

对于同一机器人基坐标系下的两个时刻（跨任务），取：

- 末端位置 `p`，单位米；
- 末端旋转四元数 `q`，导出为 **xyzw**，单位范数；
- 物理夹爪开度 `g`，按真实机械行程归一化至 `[0,1]`（0 关、1 开，**不是专家夹爪动作命令**）；
- **可选但推荐**：同一坐标系的末端线速度 `v`（m/s）和角速度 `ω`（rad/s）。若 HDF5 无法提供，就只允许报告“位姿可比”，不宣称完整动态状态等价。

四元数角距离（考虑 `q` 与 `-q` 等价）：

\[
\Delta_R(q_i,q_j)=2\arccos\left(|\langle\hat q_i,\hat q_j\rangle|\right).
\]

本轮**预注册的工程初值**：

- 位置门限 `r_p = 0.03 m`（3 cm）；
- 旋转门限 `r_R = 15°`；
- 夹爪开度门限 `r_g = 0.10` 全行程；
- 如果导出末端速度：线速度 `r_v = 0.05 m/s`，角速度 `r_ω = 0.35 rad/s`。

物理距离采用最大归一化偏差，而非加权平均，避免某项巨大的物理失配被其他项掩盖：

\[
d_s(i,j)=\max\left\{\frac{\|p_i-p_j\|_2}{r_p},\frac{\Delta_R(q_i,q_j)}{r_R},\frac{|g_i-g_j|}{r_g}\right\}
\]

如果可用且冻结为“包含速度”，再额外取和 \(\|v_i-v_j\|_2/r_v,\ \|\omega_i-\omega_j\|_2/r_\omega\) 的最大值。**固定 `τ_s=1`，判定 `d_s≤1`**。

上述容差是明确声明的待检验设计初值，不是 LIBERO 经验证的物理阈值。若原始动作控制差异对这些公差敏感，需要在**只使用基础任务**的新协议里重新预注册，不能依据 Task 7/8/9 的候选覆盖率来事后修改。

## 3. 未调制视觉相似度与 τ_v

使用基础阶段结束时的**同一冻结视觉编码器 checkpoint**，相同图像预处理，提取两路原始图像的特征，且位置必须位于 FiLM 语言调制之前。不能从已有经过 FiLM 的 `[T,4,E]` 特征直接恢复“未调制视觉特征”。

对双视角分别 L2 归一化，采用保守的双视角最小相似度：

\[
s_v(i,j)=\min\big(\cos(v_i^{agent},v_j^{agent}),\ \cos(v_i^{eih},v_j^{eih})\big).
\]

**不能沿用实验 A 中的 0.70 / 0.80 / 0.90 / 0.95 作为 τ_v**，因为 A 使用的是语言条件化视觉特征，B 的视觉度量也从均值变成双视角最小值。

确定 `τ_v` 的冻结流程：

1. 只使用 OBJECT 基础任务（本次实现默认任务 ID `0…5`）跨任务的校准样本。样本必须拥有**人工独立标注**的 `label=1`（物理局部状态和相关场景/物体配置足够可比）或者 `label=0`（不可比），并保留容易混淆的 hard negatives。标签不能由待校准的视觉余弦阈值自动生成。
2. 首先应用预注册 `d_s≤1`，仅在物理上可比的校准样本内计算视觉阈值；缺少正/负标注时**停止**，绝不假设 `τ_v=0.70`。
3. 本脚本采用简单、可复现的保守工作点：\(\tau_v = Q_{0.95}(s_v\mid label=0, d_s\le1)+10^{-7}\)（使用 nearest-higher 分位数）。它意图让**基础校准样本中已标注的 hard negatives** 只有少量通过，但不是总体精确率保证，且依赖校准样本的抽样方式。
4. 另行留出**轨迹级不相交**的基础任务验证对，报告匹配精确率、召回率、假阳性率（按任务对分层），而不是只报告校准集本身的分数。若验证失败，宣告此协议匹配质量不足；后续协议修改需重做并注明版本。
5. 冻结 `frozen.json`、视觉编码器 SHA、两路相机 preprocessing、输入/标注文件 SHA、动作标准化尺度。所有增量任务（7/8/9）只执行一次冻结规则，不得针对每个阶段重调。

最终状态门控（真实实验 B）是：

\[
C_{ij}=\mathbf 1[d_s(i,j)\le1]\ \mathbf 1[s_v(i,j)\ge\tau_v].
\]

此门控仍不能替代场景/物体状态的人工抽检。末端姿态相似不等于目标物体相对状态相同。

## 4. 专家未来动作片段

从**已验证的 HDF5 锚点时间**开始导出 `future_actions[i,:q]`，默认 `q=3`，近末端轨迹不足 q 步时 `future_valid=False`。绝不能把 `.pt` 中的历史 `[T,7]` 当成未来片段。

动作前六维的 IQR 尺度只在基础任务冻结，逐维下限 `0.1`；夹爪专家命令按照是否 `>0` 比较，须确认原始动作定义满足此判定。定义：

\[
D_a(i,j)=\sqrt{\frac1{6q}\sum_{h=0}^{q-1}\sum_{d=1}^{6}
\left(\frac{a_{i,h,d}-a_{j,h,d}}{\sigma_d}\right)^2}.
\]

- `SHARE`：`C=1` 且 `D_a≤0.5` 且 q 步内无夹爪命令冲突；
- `BIF`：`C=1` 且 `D_a≥1.5` **或** q 步内出现夹爪命令冲突；
- 中间区间：`UNCERTAIN_ACTION`；
- 无有效未来窗口：`UNUSABLE_FUTURE_ACTION`。

这些动作阈值从实验 A 继承作为**前置冻结的探索性对照**，不是物理控制冲突的验证结论。人工审计仍然必需。

## 5. 输入契约：records.npz 和 pair CSV

`records.npz` 必备数组（首维都是 N）：

| 数组 | shape | 说明 |
|---|---|---|
| `ids` | `[N]` | 稳定的 `.pt` 记录 ID，与候选 CSV 完全一致 |
| `task_ids` | `[N]` | 真实任务 ID |
| `traj_ids` | `[N]` | HDF5 唯一演示标识，须包含任务归属 |
| `alignment_verified` | `[N]` bool | 唯一精确 HDF5 窗口映射已核验 |
| `eef_pos_m` | `[N,3]` | 同机器人基坐标系末端位置（m） |
| `eef_quat_xyzw` | `[N,4]` | 末端朝向 xyzw |
| `gripper_open_fraction` | `[N]` | 物理夹爪开度 0–1 |
| `agent_unmod` | `[N,D1]` | 冻结、未调制的 agent-view 特征 |
| `eih_unmod` | `[N,D2]` | 冻结、未调制的 eye-in-hand 特征 |
| `future_actions` | `[N,Q,7]` | 锚点时刻起的未来专家动作 |
| `future_valid` | `[N,Q]` bool | 未来动作是否在当前 demo 内有效 |

如从 HDF5 得到速度，额外同时提供 `eef_linvel_mps[N,3]` 与 `eef_angvel_radps[N,3]`。校准时有速度就冻结使用速度；筛查数据必须提供同样字段。

基础任务人工校准 CSV 使用 `id_a,id_b,label` 三列，跨不同基础任务的匹配（1）/不匹配（0）对。至少有 20 个符合物理门控的正例和 40 个负例（仅是脚本防呆下限，科学报告还需更多独立轨迹和分层不确定度）。

增量候选对 CSV 使用 `id_a,id_b`，或沿用实验 A 的 `new_file,old_file`；其中 record IDs 必须能映射到 `records.npz`。对于 A 报告使用绝对路径的情况，需要确保 NPZ 中也使用相同路径，或者先用可追溯映射统一转换为稳定相对路径。

## 6. 命令

```bash
# Inspect actual HDF5 keys; no assumed observation schema.
python abcr_experiment_b.py inspect-h5 --h5 /data/OBJECT_TASK.hdf5 --limit 150

# Calibration only from BASE task human-reviewed, trajectory-aligned pairs.
# Replace 64HEX_SHA with the *actual* frozen pre-FiLM image encoder SHA256.
python abcr_experiment_b.py calibrate \
  --records /data/base_records.npz \
  --pairs /data/base_annotated_pairs.csv \
  --encoder-sha256 64HEX_SHA \
  --max-base-task 5 --q 3 \
  --pos-m 0.03 --rot-deg 15 --grip-frac 0.10 \
  --out /data/abcr_B_frozen.json

# Apply frozen thresholds to experiment-A candidate pairs.
python abcr_experiment_b.py screen \
  --records /data/verified_incremental_records.npz \
  --pairs /data/candidates_task8_seed42.csv \
  --encoder-sha256 64HEX_SHA \
  --frozen /data/abcr_B_frozen.json \
  --out-dir /data/abcr_B_task8_seed42
```

Produces `pair_audit.csv` (per-pair physical/visual/action distances + decision) and `audit_summary.json` (counts and available-anchor coverage in the given candidate list). The screen command refuses unverified HDF5 mappings and encoder version mismatch and will not overwrite an existing output directory.

## 7. Reporting and stopping conditions

Report each seed/stage/old-task separately. Required: proportion passing physical gate, proportion passing calibrated two-camera visual gate, share/bif/uncertain among gated pairs, new-anchor coverage, unique demo-pair counts, per-old-task concentration, 95% trajectory-bootstrap intervals where feasible, and manually audited label accuracy by category. Include candidates from **below the old experiment-A threshold** in a stratified audit; screening only old A-positive pairs cannot estimate discovery recall.

Stop rather than retune on incremental tasks if: HDF5 mapping ambiguous; incompatible encoder checkpoints; unmodulated images/features unavailable; base calibration lacks labeled hard negatives; or validation demonstrates poor semantic state comparability. No formal ABCR training is authorized by Experiment B alone.

**Key caveat:** The physical thresholds in this protocol are proposed engineering tolerances, not claims about LIBERO ground truth. Positive/negative classification remains an offline surrogate relation requiring manual verification and, ultimately, closed-loop FWT/BWT tests.
