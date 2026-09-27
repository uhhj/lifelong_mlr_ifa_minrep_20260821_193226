# 关系信号系统筛选与独立确认 v2

依据 `reports/task_relation_signal_screen_v2/PROTOCOL.md` 与 `USER_INSTRUCTIONS.md`。历史 `task_relation_forecast_v1` 全部只读。本代码不恢复正式 lifelong training。

## 已实现入口

在服务器仓库根目录、现有 `.migration/activate.sh` 环境中运行：

```bash
python -m task_relation_signal_screen_v2.prepare
python -m unittest task_relation_signal_screen_v2.tests -v
python -m task_relation_signal_screen_v2.queue
```

`queue` 是有限 Stage A 队列：四任务无语言匹配缓存 → 六点技术重复 → registry 冻结 → 全36点补测 → 开发分析 → 1000次开发块置换/重采样 → 图表与报告 → 独立重算。锁避免重复启动；完成单元依据原子 COMPLETE 文件跳过。

单独入口：

```bash
python -m task_relation_signal_screen_v2.measure --kind static --seed 45 --source 6 --target 7
python -m task_relation_signal_screen_v2.measure --kind probe --seed 45 --source 6 --target 7
python -m task_relation_signal_screen_v2.analyze_dev
python -m task_relation_signal_screen_v2.development_checks
python -m task_relation_signal_screen_v2.report_dev
python -m task_relation_signal_screen_v2.audit_stage_a
```

运行分析前必须已有全部36点和冻结 registry。开发结果不构成独立新任务确认；不能根据中途结果跳包或换设置。

## 独立面板资产

```bash
python -m task_relation_signal_screen_v2.panel
python -m task_relation_signal_screen_v2.confirmation_assets
python -m task_relation_signal_screen_v2.confirm_cache --seed 48 --task 0 --role D_signal
```

`panel` 仅用当前共同基座的语言编码器和官方任务文本；`confirmation_assets` 只下载六个被选任务的官方文件，固定远端 revision 与 LFS SHA256，核验动作/观测接口、旧演示重复性和已有seed目录，冻结整轨迹数据角色。缓存入口使用角色限定的 HDF5 view，不通过完整数据集低维缓存提前读取其他角色。

确认阶段入口：

```bash
python -m task_relation_signal_screen_v2.freeze_confirmation
python -m task_relation_signal_screen_v2.confirm_queue
```

先固定 seed48 的任务0与3及双向适应作为技术单元（计入总预算），然后其余源/目标单元和234个唯一闭环组合。每组合40个原生trial。`--sources-only` 仅执行18个source，不能称为完成确认。独立目录内的缓存、source、C0信号、Cq封存、target、E_eval与行为均可按完成标记续做。

`confirm_runtime` 使用历史seed45配置模板并显式设置新seed与suite，不伪造不存在的历史配置。源轨迹实际前300步作为同目标基座reference；目标分支核对相同初始optimizer/scheduler、数据流和LR。Cq与reference按相同原生call比较，即使有效更新计数不同也记录差异。正式lifelong不恢复。

数值汇总入口 `adjudicate_confirmation` 要求90对和9360个trial均齐全，重建共享参照标签、5000次配对初态重采样和max-T区间。队列最后状态为 `MEASUREMENTS_AND_STATISTICS_COMPLETE_REVIEW_REQUIRED`：仍需独立核验、任务删除敏感性解释及最终报告，不能把机械门限或资产准备当作独立确认完成。

## 解释约束

- 原36行是12种有向关系×3优化seed，只有4个任务。
- C1/5/20按 LR>0 且真实参数位移非零计数，保留原零LR调用。
- 静态与探针进程禁止打开闭环标签和 E_eval；访问清单单独保存。
- S10在语言FiLM之前提取视觉表示；它不是直接复用已调制的视觉缓存。
- 常数/缺失列在训练折处理，原始null保留。一个组件不可用不会伪造数值；整个包不可用才排除候选资格。
- 动态组合在外层训练折内选择C0，最终部署系数仅在开发结束后用全开发集冻结。
- S01/S04/S05/S06的增量计时使用保守共享计算包；S10缓存构造单列。增量时间不冒充部署总时间。
- 大二进制留在 `diagnostic_storage/task_relation_signal_screen_v2/`；提交源码与小报告。

固定：`formal_training_resumed=false`。探针真实执行原生更新，因此 `diagnostic_training_performed=true`，即使更新只发生在隔离的实验状态。
