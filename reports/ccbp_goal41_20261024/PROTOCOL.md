# CCBP GOAL Task8→Task9，有限 seed41 验证 v1

用户附件是科学目的；本文件补齐未指定实现，须在任何正式更新前冻结。先前语义筛查停止结论保留，本实验不假设已有因果失败阶段。

## 四组与同起点
仅 seed41，A ALIGN；B 固定执行行为蒸馏；C 执行分量专家监督；D CCBP。全部从已验收 ALIGN GOAL seed41 Stage8 ep99 的100键权重开始；同一初始化 RNG seed41、原 Stage9 manifest 3365样本顺序、DistributedSampler seed0、epoch0–99、batch10、337步/epoch，四组共134800主更新。按原阶段规则新建 AdamW lr1e-4/wd.1，500步warmup及原逐主步scheduler，FP32/clip100/BC+0.1IFA/gamma.7，原 modality0 Temporal读出、5分量7维GMM、ALIGN next-action合法mask不变。不声称与历史Stage9 RNG续流完全相同；A也重新从共同Stage8分叉。

仅对原回放Task8合法窗口添加辅助项，Task6/7仍参与原BC，Task9仍参与原BC和native-only IFA。原manifest身份、顺序、feature字节和labels SHA逐项验证；不遍历已扩大的当前buffer来构造训练人口。教师 eval/no_grad 固定Stage8，仅缓存Task8合法窗口的5分量和logits；不访问额外旧演示。数据/教师准备无优化器更新，成本单列。

## 损失一次性选择
全部辅助项 lambda=0.1，routing beta=0.1，logit margin=0.1，D epsilon=0.05。这是附件未定参数的预设实现，不是已优化值，不扫阈值/距离/权重。每组translation(3)、rotation(3)、gripper(1)尺度s为固定Task8合法专家动作组RMS，floor0.1；只从训练回放拟合并保存SHA，不用闭环选尺度。

cost(mu,sigma,target)=各组 mean((mu-target)^2+sigma^2)/s^2 的和。为未裁剪Normal期望平方误差代理，不是物理风险或成功概率。教师budget=教师argmax分量cost+epsilon。

A辅助0。B固定执行Normal蒸馏：学生argmax分量与教师argmax分量的分组 W2平方（均值差平方+标准差差平方），另路由至当前最接近教师执行Normal的分量；不绑定任意分量编号。C直接降低学生argmax分量expert cost，另路由至当前最小expert cost分量。B/C/D均使用相同lambda/beta和同一合法旧窗口，额外eval前向与计算独立列示，不能称等算力。

D为relu(selected_cost-budget)，可接受集合(cost<=budget)每次forward后detach；good/non-good logits margin。全good路由0；空good时再加relu(min_cost-budget)，并路由到当前最小cost分量。argmin/argmax/集合身份detach，连续cost对均值/标准差可微，路由对logits可微。原BC保留完整mixture NLL。辅助在学生eval模式前向，之后恢复所有modes/RNG；不改变原BC dropout随机序列。教师始终冻结，推理结构与原ALIGN完全相同。

## 有限评估与停止
全部四组完成100epoch后评估各Task8/9，各20初态0–19，seed41/42两批，每轨迹最多600步+5个初始zero，10并行；共160正式回合16vector8cell。相同(task,index)四组用完全相同预生成600×7标准Normal噪声，argmax分量mean+std*noise，每步重新观测。噪声seed=20261024+41*1000+task*100+index；这是新配对诊断，不能直接拼成历史正式AUC。无中途评估选CP、无重试科学负结果、无额外环境准入rollout。

描述成功率、paired SF/FS、Task8/9和二任务等权平均，所有负单元与Task9损失保留。预设支持标准：D Task8严格高于A/B/C各组，Task9不低于A/B/C任一组，故二任务mean也严格更高。未满足只报告有限/不支持，不以平均掩盖新任务下降；样本小，不称显著。即使满足也不自动扩seed；下一阶段需新决定。若仅超过A而未超过C，不归因CCBP独立收益；离线改善但闭环不改善，停止扩展模块。本轮不开发保护网络，不训练更多权重/容量/阈值取值。

## 工程与验收
自己的四组隔离GPU准入：真实下一步内存序列化保存/恢复重放、零辅助A与原ALIGN更新一致、teacher冻结、mask/空集合/全good/路由梯度/长度1–8和部署Normal语义。每组3次隔离optimizer step、A另1次原ALIGN对照更新（合计13次）另列，不计主预算；正式重新从共同Stage8开始。每完整epoch保存model/optimizer/scheduler/allRNG/grads/modes/requires/cursor/合同及输入SHA；中断不盲重启，保留后才版本化处理。无修改旧源、旧CP和旧输出。

新私有输出在/root，额外总界2.3GiB（含原子保存临时峰值），保留system普通1GiB及data root2GiB；仅代码/报告/metadata/hash/episode outcomes发布，不下载/发布权重、buffer、actions/states/噪声载荷。全部exit0/no-residual后实际4CPU strict CP、160episode、全账目/父输入/教师预算验收；进度不是最终接受。按实测整组训练/整个评估预计完成点安排监控，不固定短周期。全部验收发布后删除监控。
