# CCBP-v3：GOAL seed41 完整 C／R／G 有限验证

用户于2026-10-25明确选择“审查通过后执行三组完整序列”。三组各自从官方 GOAL multitask ep96 初始化，完整 Task6→7→8→9，各阶段100 epoch。不是共同Stage8分叉，也不把旧C权重挪作本轮已完成训练。附件中的执行建议作为待审查设计；本协议以用户三组要求为准，不额外加A、固定任务获得时教师、其他seed或扫描。

## 固定输入、模型与预算

保持已验收完整v2的原RA-Coreset成员、顺序、载荷、ALIGN下一步动作标签及mask、BC+0.1IFA、时序modality0读出、5分量7D GMM和部署方式。父输入原地只读，不复制features或labels。各组独立初始化，组内随机流连续，阶段按原流程重置优化器和scheduler。

|阶段|人口|合法BC|屏蔽|合法旧窗口|每轮更新|
|---|---:|---:|---:|---:|---:|
|6|997|917|80|0|100|
|7|1388|1286|102|477|139|
|8|1848|1723|125|899|185|
|9|3365|3218|147|1329|337|

每组400 epoch、76100正式更新；三组1200 epoch、228300更新。FP32、batch10、sampler0、seed41、AdamW 1e-4 / wd0.1、500 warmup、clip100、IFA gamma0.7保持。每阶段100轮后固定ep99检查点，不按测试挑轮次。Task6辅助为零，须实际复现官方ALIGN Task6 tensor。

Task7–9只使用本组上一阶段检查点生成教师，eval/no_grad、保存恢复全RNG，整阶段冻结。对当时全部合法旧窗口计算尺度（分组动作RMS、floor0.1）和tau（教师执行专家cost的float64中位数、floor1e-8），冻结w=tau/(tau+cT)。不使用当前batch重估，不读取未来任务数据。

## 三组目标（全部在运行前固定）

c为含方差的原分组归一化专家期望平方误差，d为到教师实际argmax执行Normal的原分组W2平方。学生k为argmax logits，教师编号与学生编号不绑定。lambda=0.1、beta=0.1、rho=1、margin=0.1，不调参。

- C：原c[k]加原margin路由，目标为argmin c，完全并列按原C第一编号。
- R：保留V2的J=(c+w*d)/(1+w)及J[k]回归，只将路由函数改成margin。目标仍为argmin J，并保留V2最低评分并列时选最高logit的规则。因此R不偷偷改成C的专家排序目标。
- G：c[k]+w*Rseg[k]+0.1*原C路由。Rseg为归一化均值/标准差14维输出到教师—专家端点线段的平方距离。专家端点标准差零仅为距离定义，不执行零方差Gaussian。保持上传原型的投影及detach语义，不除以1+w。

三组总目标均为原BC+0.1IFA+0.1辅助均值。学生辅助前向eval但保留梯度，恢复原模式和全部随机状态，保持BC dropout流。部署仅原argmax分量Normal采样，逐步重新观测；无专家动作、教师、w或投影参与推理。

G-C检验“在持续专家监督和原路由上添加方向保持”的整体增量。R保留混合评分，检验回退路由形式的长期效果；R-G同时涉及专家缩放、锚几何、路由评分，不能仅归因为方向几何。滚动教师后续输出随训练而变，不声称已独立隔离一致度权重或固定教师的作用。

## 准入与诊断

每个实际组/阶段自己的准入，不套旧PASS：真实GPU完整保存/加载、下一批次精确重放、恢复正式初始状态、长度1–8和argmax Normal部署语义。

各Task6为3隔离恢复更新+1原ALIGN更新对照。旧阶段均先3恢复更新+1有历史梯度状态的诊断开关更新；C额外1原实现对照，R额外1独立公式对照，G额外2次分别关闭锚与原C更新对照，必须全状态精确相等。预计隔离optimizer调用60次，0环境。G关锚对照也恢复完整状态，不挪用其权重。实际以准入记录核算。

旧阶段epoch0/9/49/99首个合法旧batch分别记录BC、加权IFA、加权执行、加权路由参数梯度范数，G另记录加权方向锚。autograd.grad不得改p.grad/RNG/更新；须在已有梯度状态实测。正式额外autograd为C48/R48/G60共156次，准入C12/R12/G15共39次，成本单列。这些仅尺度诊断，不等于因果证明。

每完整epoch原子保存model、optimizer、scheduler、grads、modes、requires_grad、Python/NumPy/CPU/CUDA RNG、step/cursor、history、教师/输入SHA。工程失败先核停止/hash/完整恢复并保留原失败，再版本化接续，禁止盲重启、覆盖冻结代码、重复完成epoch或正常科学负结果。

## 配对闭环、主要终点和停止条件

完整三角矩阵：各Stage评所有已见Task，每cell20初态0–19。每组200回合20vector10cell；三组600回合60vector30cell。每task/index跨组和阶段配对同600×7 FP32 Normal噪声，沿用seed=20261024+41000+task*100+index；10环境、envseed41/42、600步、5zero。全部三组训练完成才开始评估。没有额外测试重试。

预登记主要继续研究标准：**G最终OldFinal严格高于C，且G最终Task8和Task9均不低于C。Task9可接受下降范围为0个百分点。** 这是保守的探索决策规则，不是统计显著性检验。Task6/7分别报告，不能以OldFinal宣称每个旧技能无损。R对照只作设计诊断，不按结果改主终点。报告全部矩阵、AUC、FWT、NBT（低好）、OldFinal、FinalAll、G-C/G-R/R-C配对SF/FS及每个负单元。若只降低离线误差而未改善闭环，或未达主要标准，停止本版本；不追加seed、权重、margin、教师来源或其他组。不把单seed既有初态称为确认集。

闭环为主要证据。最终CPU逐合法旧窗口记录五分量专家cost、教师距离、方向距离/投影系数、logits、冻结权重及SHA；仅轻量标量，不导出动作、状态、特征或权重。本地用独立Gram距离公式复算方向距离、实际损失和路由，报告离线与闭环不一致，不把更低regret当作效果保证。

## 结束验收、容量与发布

全部controller自然exit0且无自身进程后，才运行外置CPU final_readonly→audit_final_local。必须实际12CPU strict完整CP、3完整最终恢复、9教师源/缓存、Task6四方（C/R/G/原ALIGN）tensor一致、全部epoch/更新、600初态真实字节和noiseSHA、30cell/60vector唯一attempt及标量独立验收。最终wrapper静态检查不是实际PASS。

新科学目录CCBP_V3_FULL_GOAL41_20261025_v1；不得恢复旧v2/Task9分叉/CAET/TACR/readout。新布局暂以root1.10GiB/data1.40GiB新增峰值预检（含原子保存），独占inode抵扣实际已分配；system普通保留1GiB、data root保留2GiB。这是拟定界，必须新远程实际核算才准入。容量不足先查明确废用中间资产，用户既有授权允许精确留证清理；旧最终模型/最终恢复/原始数据/活跃或未验收资产不可当废用。不得付费、降预算/精度或下载私有payload。

每次检查/审计/实施按根AGENTS只发布精确审核的代码、报告、hash、标量及outcomes，非force推MLR并核remoteSHA和本地receipt。进行中标PARTIAL并记真实观察时间。按预计整组训练/整段评估完成点调整同一监控；连接故障退避明确区分于完成ETA。全验收、分析和发布后删除监控，不自动扩展。
