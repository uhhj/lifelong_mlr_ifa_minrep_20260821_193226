# OBJECT 前置实验 P1–P3

P4 按用户后续指令暂停，包括采样模拟和 20-epoch 训练。本轮不训练、不修改主干、MLR 或正式 buffer。

## 数据与时序

使用同一冻结 OBJECT 基础 checkpoint。诊断池每个旧任务 demo0–4，当前 Task6/7/8/9 各 demo0–9，全部时间步。
这是本次方案指定的池（包含基础任务0–5），不是正式 MLR 实际只包含已学增量任务的回放快照。
70 条轨迹的 obs[t] 与 states[t+1] 关节位置逐位一致。真正未来从 actions[t+1] 开始；目标物体坐标读 states[t+1]。
这与历史 B 的同下标动作定义不同，历史报告不改写。末尾没有真实后继的项目不填充；缺失数量显式报告。
官方数据转换源码同样先 step(action) 后保存 obs：[create_dataset.py](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py)。

## P1：数量覆盖，而非物理真实性认证

归一化本体状态为 7 个关节与 2 个夹爪位置，尺度只来自基础池。每阶段全量跨任务对精确计算 Q15/Q85；
可比对内 q4 动作差异 Q25/Q75。动作差异为标准化 7 维命令的 RMS，运动尺度基础 IQR，下限0.1，夹爪完整幅度2。
主 beta=0.5，0.3/0.7 均单独报告，不挑最优。30%状态覆盖、10%共享覆盖、10%分叉覆盖为主数量门，8%仅为原设计低端敏感性。

|当前任务|状态覆盖|共享锚点覆盖|分叉锚点覆盖|数量门|
|---|---:|---:|---:|---|
|6|100.00%|90.59%|89.56%|True|
|7|100.00%|94.19%|91.55%|True|
|8|100.00%|95.32%|95.26%|True|
|9|100.00%|96.13%|92.42%|True|

分位数会按定义产生动作两端类别，因此 pair 级约25%/25%不是独立成功证据。anchor覆盖仍需实测，可能集中于重复时间步或特定旧任务。
这些阈值使用当前阶段自身分布，是探索性标定，不是对独立确认集的物理匹配验证。它不会推翻实验 B 已记录的负结果。

## P2：检索物理误差

H 使用原生 CompletePolicy 当前4-token输出；实际 use_state=False，第四token是另一组FiLM(eih视觉)，不是本体状态token。
为保证配对比较，两侧从相同原图、相同 checkpoint、eval、无随机增强重新导出，未混用旧随机增强缓存。
C 主版本先按同一 Ds 门选候选再按视觉排序。因此其 EEF 误差优势可能来自输入本体状态，不能直接归因为去除 FiLM。
补充相同物理候选门下 H 以及纯未调制视觉检索两组控制。下表是 beta0.5、按当前轨迹配对的平均误差差（负值表示前者更好）。

|任务|比较|指标|均值差(m)|轨迹bootstrap95%CI|
|---|---|---|---:|---|
|6|C_physical_gate_then_visual_vs_H_full_cosine|eef_error_m|-0.073838|[-0.07907951541151903, -0.06813445591683309]|
|6|C_physical_gate_then_visual_vs_H_full_cosine|object_relative_vector_error_m|-0.037259|[-0.05029416338522462, -0.02257656548102695]|
|6|C_physical_gate_then_visual_vs_H_full_cosine|object_eef_distance_error_m|-0.026631|[-0.03694841148054468, -0.014587033548966002]|
|7|C_physical_gate_then_visual_vs_H_full_cosine|eef_error_m|-0.004903|[-0.009551146757397006, -0.00011403674853723185]|
|7|C_physical_gate_then_visual_vs_H_full_cosine|object_relative_vector_error_m|-0.001258|[-0.008346924768043064, 0.005041579005086016]|
|7|C_physical_gate_then_visual_vs_H_full_cosine|object_eef_distance_error_m|-0.010770|[-0.017340514244221922, -0.0053375546203530124]|
|8|C_physical_gate_then_visual_vs_H_full_cosine|eef_error_m|-0.003621|[-0.005254437286316118, -0.0022242089764212204]|
|8|C_physical_gate_then_visual_vs_H_full_cosine|object_relative_vector_error_m|0.002764|[-0.0035745018777261295, 0.00885981545093396]|
|8|C_physical_gate_then_visual_vs_H_full_cosine|object_eef_distance_error_m|-0.004784|[-0.00940888815675633, -0.0005815718567856484]|
|9|C_physical_gate_then_visual_vs_H_full_cosine|eef_error_m|-0.067156|[-0.07188898882577598, -0.0626744129248196]|
|9|C_physical_gate_then_visual_vs_H_full_cosine|object_relative_vector_error_m|-0.002489|[-0.0052135153769975445, 0.0005673937714797291]|
|9|C_physical_gate_then_visual_vs_H_full_cosine|object_eef_distance_error_m|0.007046|[0.003798473951031805, 0.010041346008778582]|
|6|unmod_visual_only_vs_H_full_cosine|eef_error_m|-0.070519|[-0.07632695520051047, -0.06403996970560671]|
|6|unmod_visual_only_vs_H_full_cosine|object_relative_vector_error_m|-0.024119|[-0.03782082680035051, -0.00864215667242607]|
|6|unmod_visual_only_vs_H_full_cosine|object_eef_distance_error_m|-0.013462|[-0.024394867220681203, -0.0006291919138949828]|
|7|unmod_visual_only_vs_H_full_cosine|eef_error_m|-0.005104|[-0.00789433355191413, -0.0024623527546748076]|
|7|unmod_visual_only_vs_H_full_cosine|object_relative_vector_error_m|0.001541|[-0.0033312497623427446, 0.006414375061798499]|
|7|unmod_visual_only_vs_H_full_cosine|object_eef_distance_error_m|-0.008301|[-0.01340484497887001, -0.004281173498096269]|
|8|unmod_visual_only_vs_H_full_cosine|eef_error_m|-0.003040|[-0.004486417501706598, -0.001810414419218515]|
|8|unmod_visual_only_vs_H_full_cosine|object_relative_vector_error_m|0.007424|[0.001271678591884397, 0.014287126124100318]|
|8|unmod_visual_only_vs_H_full_cosine|object_eef_distance_error_m|-0.000325|[-0.004259457385135098, 0.004111406522560877]|
|9|unmod_visual_only_vs_H_full_cosine|eef_error_m|-0.063939|[-0.07040193699484919, -0.05678056577082519]|
|9|unmod_visual_only_vs_H_full_cosine|object_relative_vector_error_m|-0.001367|[-0.004063483487154665, 0.0015024719375608126]|
|9|unmod_visual_only_vs_H_full_cosine|object_eef_distance_error_m|0.008329|[0.005955587337856528, 0.010736343432921585]|
|6|C_physical_gate_then_visual_vs_H_same_physical_gate|eef_error_m|-0.051573|[-0.05854677953528913, -0.044093728371255345]|
|6|C_physical_gate_then_visual_vs_H_same_physical_gate|object_relative_vector_error_m|0.003913|[-0.0030335773198330667, 0.01244433797108055]|
|6|C_physical_gate_then_visual_vs_H_same_physical_gate|object_eef_distance_error_m|0.014316|[0.008874081945852218, 0.020114097250435878]|
|7|C_physical_gate_then_visual_vs_H_same_physical_gate|eef_error_m|-0.005190|[-0.007533695025964434, -0.002961562377045512]|
|7|C_physical_gate_then_visual_vs_H_same_physical_gate|object_relative_vector_error_m|0.005692|[0.0027940364628361264, 0.008622347024461111]|
|7|C_physical_gate_then_visual_vs_H_same_physical_gate|object_eef_distance_error_m|-0.003087|[-0.004300566867779245, -0.0018866666052276275]|
|8|C_physical_gate_then_visual_vs_H_same_physical_gate|eef_error_m|-0.002779|[-0.004601213223380085, -0.0011603115949360317]|
|8|C_physical_gate_then_visual_vs_H_same_physical_gate|object_relative_vector_error_m|0.007445|[0.0008377042328606807, 0.01408694316980539]|
|8|C_physical_gate_then_visual_vs_H_same_physical_gate|object_eef_distance_error_m|-0.000111|[-0.004909499740369157, 0.004634441565594747]|
|9|C_physical_gate_then_visual_vs_H_same_physical_gate|eef_error_m|-0.059460|[-0.06868012318833851, -0.04916225301380201]|
|9|C_physical_gate_then_visual_vs_H_same_physical_gate|object_relative_vector_error_m|-0.001320|[-0.004246515905961368, 0.0018595381868566085]|
|9|C_physical_gate_then_visual_vs_H_same_physical_gate|object_eef_distance_error_m|0.008275|[0.00552973308567789, 0.010985836451616902]|

物体误差使用目标物体free-joint原点到末端的向量/距离；不同物体原点并非统一语义接触点。
每个新轨迹为统计单元，旧检索池固定且被重复使用；不能把全部时间步当独立样本，也不能仅凭一个 seed 的配对区间推广所有模型。

## P3：后续动作分叉

所有 q 固定使用各阶段 q4 得到的同一 tau+/tau-，不按 q 重设分位数。主对比使用相同 q6 完整对集，避免末尾删除造成假改善。
“单步相同但后续动作不同”是时序动作分叉代理，不是经人工/环境验证的伪共享真值；不预设25%–40%或低于5%的结果。

|任务|q|q1共享后变为分叉|当前q共享、后续至q6变分叉|该q截断比例|
|---|---:|---:|---:|---:|
|6|1|0.00%|0.15%|1.21%|
|6|2|0.00%|0.03%|2.36%|
|6|3|0.00%|0.00%|3.42%|
|6|4|0.01%|0.00%|4.41%|
|6|5|0.04%|0.00%|5.35%|
|6|6|0.14%|NA|6.24%|
|7|1|0.00%|0.09%|1.17%|
|7|2|0.00%|0.03%|2.31%|
|7|3|0.00%|0.00%|3.38%|
|7|4|0.00%|0.00%|4.40%|
|7|5|0.03%|0.00%|5.34%|
|7|6|0.09%|NA|6.22%|
|8|1|0.00%|0.18%|1.08%|
|8|2|0.00%|0.04%|2.11%|
|8|3|0.00%|0.00%|3.08%|
|8|4|0.02%|0.00%|3.99%|
|8|5|0.07%|0.00%|4.84%|
|8|6|0.18%|NA|5.67%|
|9|1|0.00%|0.13%|1.13%|
|9|2|0.00%|0.03%|2.22%|
|9|3|0.00%|0.00%|3.25%|
|9|4|0.02%|0.00%|4.26%|
|9|5|0.05%|0.00%|5.19%|
|9|6|0.13%|NA|6.07%|

q6 后的分叉未观测，因此不把其0次观测当作低假阳性率。各 q 的可用数量与共同分母见原始表。

## 解释边界与产物

输出 P1/P2/P3 全部 beta 结果、逐检索锚点、逐新轨迹误差、旧任务贡献、q窗口曲线、固定阈值与运行成本。
KDE只用于展示：全量阈值先计算，绘图随后固定种子抽样2000对，使用Scott带宽；不据图调阈值或挑beta。
本轮不证明真实技能共享、动力学冲突、ABCR有效性、未调制特征必胜或最优q；没有改动历史结果。
`formal_training_resumed=false`。
