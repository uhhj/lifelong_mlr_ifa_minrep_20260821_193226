# CCBP-v3 候选原型：回退C路由 + 纠错方向保持

**状态：本地CPU合成测试通过；未加载真实检查点，未训练，未运行环境。**
本目录是损失原型，不是完整仓库补丁，不会自动调用GPU、GitHub或服务器。
不要覆盖任何已冻结实验。运行数据、对照和结论都需要新协议。

## 依据

核查仓库：`uhhj/lifelong_mlr_ifa_minrep_20260821_193226`
固定提交：`decaa233260d7056e5c343b6dd35ebf5394d2dfa`

主要材料：
- `reports/ccbp_v2_full_goal41_20261024/final_readonly_20261025_054702/FULL_REPORT.md`
- `reports/ccbp_v2_full_goal41_20261024/PROTOCOL.md`
- `reports/ccbp_v2_full_goal41_20261024/losses.py`
- `reports/ccbp_v2_full_goal41_20261024/updates.py`

本次是完整Task6→9，不是上一轮Task8→9分叉。最终成功率：

|组|Task6|Task7|Task8|Task9|AUC|OldFinal|
|---|---:|---:|---:|---:|---:|---:|
|C|30|100|85|90|83.125|71.6667|
|N|30|100|85|75|78.75|71.6667|
|V|45|100|80|80|82.5|75|

这些是单seed探索结果，不是显著性、独立确认或因果归因。

## 原型修改

1. 恢复原C的“最低专家cost分量 + margin hinge”路由，保留其原始并列规则。
2. 专家执行cost保持完整系数，不除以 `1+w`。
3. 将到教师单点的W2锚，替换为到“教师→专家端点”线段的距离；允许沿纠错方向移动。
4. 第一轮保留V2已有的阶段教师、旧窗口、尺度、冻结weights、outer lambda=0.1和beta=0.1。
   rho=1和margin=0.1仅是未优化初始设置，不是有效性结论。

**不做：**新增网络、按失败初态加权、在线使用专家标签、重采样回放、增加环境数据、
自动增大权重、自动恢复已停止实验、宣称部署硬保护。

## 接口

```python
from losses_ccbp_v3 import auxiliary_v3

aux, stats, terms = auxiliary_v3(
    student=(means, stds, logits),        # [B,5,7], [B,5,7], [B,5]
    target=next_action,                   # [B,7]，仅原ALIGN合法旧窗口
    teacher=(teacher_means, teacher_stds, teacher_logits),
    scales=group_scales,                  # [3]，原阶段训练回放RMS尺度
    weights=frozen_weights,               # [B]，原阶段冻结的tau/(tau+cT)
    anchor_strength=1.0,
    routing_weight=0.1,
    routing_margin=0.1,
)
loss = aligned_bc + 0.1 * ifa + 0.1 * aux
```

学生辅助前向必须eval模式但保留梯度。沿既有updates.py恢复所有模块模式及RNG，
不改变BC dropout流。此函数不管理这些状态。teacher、target、scale、weights均detach。
空旧batch返回可微零。需要在实际训练器上另做状态/恢复/一步更新准入；本地损失测试
不能替代真实优化器和分布式训练验收。

`anchor_strength=0`退化为原C辅助损失。本地已检查损失值和各学生输出梯度逐元素一致，
但未声称真实模型优化器更新逐tensor一致。

返回terms中的expert/anchor/route分开测量。总损失中它们的系数依次为0.1、0.1、0.01；
anchor已经乘rho*w，不要重复乘权重。不要仅凭损失值或极少数梯度样本判断真实优化影响。

## 参考教师

首轮默认沿用上一阶段教师，避免同时改变路由、锚几何、教师来源。

独立消融可对每个任务在刚完成训练、执行RA选样后缓存这些样本的本组执行分量均值和
标准差，后续不覆盖；只需要缓存输出，不需要保存所有历史模型。使用原始坐标参数，
在后续阶段按当前共同尺度重新计算几何；metadata记录任务、样本、教师CP和尺度。
这种“任务获得时参考”仅检验累计漂移假设，不保证优于最近阶段教师，也可能保留旧错误。
该缓存逻辑未在本原型中实现，不能把传入某个teacher张量当作已验证来源。

## 建议的有限对照

- A：同协议ALIGN，无辅助。
- C：当前最有竞争力的专家cost + 原C路由。
- R：完整V2，仅将softplus路由换成原C型margin路由，排序仍用原J。
- G：本原型；上一阶段教师和原w保持不变。

R用于隔离路由形式；G相对R是目标几何和路由依据的整体重设计，不能拆分归因。
G相对C判断新锚是否值得保留。“任务获得时参考”另作G的一项消融，不并入首轮。

采用完整Task6→9、统一主步数/样本流、同task/index逐轨迹噪声；单列额外前向成本。
主要观察Task6/Task8最终保持、Task9学习、OldFinal、AUC及全部配对SF/FS。
保留原预登记端点，不用FinalAll抵消局部退化。seed41和初态0–19仍属已见开发数据；
确认实验须使用事前固定的独立种子/任务顺序/初态。任何停止条件与超参选择限额在运行前冻结。

## 运行本地测试

依赖：Python 3.10+、PyTorch；本次测试为CPU PyTorch 2.10.0。

```bash
cd ccbp_v3_directional_prototype
python test_ccbp_v3.py
```

输出`synthetic_test_results.json`。包括坐标等价、纠错路径零锚、距离上界、输出空间梯度、
教师冻结、关闭锚精确还原C、唯一最小值时分量重排、FP32、空batch及小标准差测试。

**输出空间的几何性质不等于网络参数梯度兼容，不等于Task9无损，不等于闭环成功。**
