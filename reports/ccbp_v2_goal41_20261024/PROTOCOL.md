# CCBP-v2 GOAL seed41：固定四组有限分叉

用户2026-10-24授权：按给出的新版公式及ccbp_v2_prototype.zip，在GOAL seed41运行原B、原C、v2关闭教师锚、完整v2四组。此协议在任何正式训练前固定。v1负结果及已停止队列保持原状；本轮探索，不追加uniform组、seed或扫描参数。

## 起点与训练

四组B/C/N/V均重新从同一已验收ALIGN GOAL seed41 Stage8 ep99（100键）独立开始，仅训练Task9各100epoch33700主步，总400epoch134800。沿用原Stage9有序manifest3365、3218合法BC/147屏蔽/430合法Task8窗口，RA成员和payload不变；不枚举已经扩大的当前buffer。原modality0 Temporal读出、5分量7维GMM、FP32、batch10、sampler seed0、分叉global seed41、AdamW1e-4 wd.1、500warmup、原逐主步scheduler/clip100、BC+0.1native-only IFA/gamma.7不变。Task6/7回放仍BC，无新增保护项。与v1共同新分叉的随机流对照，不冒充历史Stage9续流。

仅在合法旧Task8窗口添加辅助损失。B/C沿用v1精确代码语义：B执行Normal到教师执行Normal的分组W2平方及最接近教师分量的margin路由；C执行分量专家cost及最小cost的margin路由。B/C旧margin0.1保留作为控制；N/V没有epsilon、可接受集合或margin。

## 固定v2公式

c_m=sum_groups mean((mu_m-y)^2+sigma_m^2)/s_g^2；d_m=sum_groups mean((mu_m-muT)^2+(sigma_m-sigmaT)^2)/s_g^2，教师只用其argmax实际执行分量。尺度沿用固定训练回放值[0.46725568175315857,0.10000000149011612,1]。

只读复用已验收Stage8教师430窗口cache，逐身份/标签/源CP/hash绑定，不重新训练教师。tau在这430窗口的教师执行cost上按原型float64 quantile(0.5)取中位数，含数值floor1e-8，随后冻结；w=tau/(tau+cT)，不是成功概率。N用rho0，V用rho1；J=(c+rho*w*d)/(1+rho*w)。不存在按具体失败窗口或初态的单独权重。

执行项J[argmax(logits)]；路由项sum_m stopgrad(J_m-minJ)*softplus(z_m-z_best)/(K-1)，K5。minJ并列时按最高logit选择参考，严格沿用上传原型；若评分和logit同时完全并列，采用原型PyTorch max的子梯度约定，不宣称所有退化并列下梯度编号不变。argmax部署语义不变。完整损失=BC+0.1IFA+0.1*mean(exec+0.1route)。外部lambda.1、beta.1、rho1一次冻结，不扫参数。

学生辅助eval但保留梯度，保存恢复modes和RNG；原BC dropout流不改。教师detach。新增项均为软正则，不保证选到最优分量。推理绝不访问专家动作、tau或w，不新增网络。

## 自身准入与分项梯度记录

上传ZIP及其声明的合成PASS仅作源材料，不能替代本轮准入。本轮自己的CPU合成检查验证与原型N/V损失和梯度精确一致、B/C与旧损失梯度精确一致、teacher无梯度、低误差仍有纠错梯度、代价差调制路由、并列/重编号和全同分数。自己的每组GPU准入包含3次下一步序列化恢复调用，再1次分项autograd诊断开关更新精确对照；B/C另1次旧辅助实现更新精确对照，共18隔离optimizer调用/0环境。正式重新初始化，不使用准入训练权重。

分项损失每batch记录。每组epoch0/9/49/99的第一个合法Task8 batch，记录BC、加权IFA、加权exec、加权route四项在所有可训练参数上的L2梯度尺度及非零tensor数。autograd.grad retain_graph不写p.grad，不推进RNG或优化器；准入必须证明诊断开关不改变实际下一步完整状态。正式64额外autograd调用单列耗时；这不是逐样本全面梯度冲突诊断，不依据诊断调权或挑CP。

## 闭环、支持标准与停止

全部训练完成后，对每组Task6/7/8/9各20初态0–19，共320正式回合32vector16cell。10并行/seed41,42/最多600policy步+5zero，argmax分量Normal采样、每步重观测。相同task/index四组使用相同600x7 float32标准Normal序列，seed=20261024+41000+task*100+index；Task8/9沿用v1配对噪声方便比较，仍是重复探索，不是新确认集或历史AUC。无额外准入环境，无科学负结果重试。

主要探索性依据：V的Task8严格高于C且Task9不低于C（首轮新任务允许损失固定0个百分点）；报告相对B、N的全部差值，不要求V每个任务都严格第一，不以两任务平均掩盖Task9退化。Task6/7分别及旧任务平均完整报告；若任何一项低于C，则不能声称其他旧技能无损，暂停扩大保护结论。即使主要条件满足也无自动扩seed。教师锚的增量只能通过V对N评估；若N更好或离线改善未转为闭环，不为保留名称扩展V。不加入uniform对照，不能独立归因一致度权重优于固定权重。

固定seed41、已探索初态用于有限筛查；不称显著、因果失败阶段或硬保护保证。后续确认需用户新决定及未用于选型的种子/顺序/初态。本轮结束验收发布后停止监控。

## 存储与工程

新root /data/mlr_ifa/experiment_workspaces/CCBP_V2_GOAL41_20261024_v1。B/N私有输出在/root，C/V在/data，各自完整epoch恢复文件同时包含state和共享storage的state_dict别名；阶段完成后stage9.pth为该不可再写恢复文件的硬链接。完整optimizer/scheduler/RNG/grads/modes/requires/model/cursor不丢弃，不降精度。自己的准入验证序列化精确及storage共享；最终CPU核文件inode、SHA、state_dict和恢复model相等。每volume保守新增峰值1.04GiB（含原子保存临时文件），按本轮稳定唯一inode计已分配，硬链接不重复抵扣；/root普通留1GiB，/data root f_bfree留2GiB。不删旧资产、不下载payload、不付费扩容。

新源合同冻结后不覆盖。工程失败先核停进程/冻结hash/完整epoch状态/原失败留证，再版本化处理；不盲重启controller或部分正式评估。最终真实CPU strict4CP、400epoch134800主步、18隔离更新、tau/weights独立复算、全320初态文件字节/noiseSHA/outcomes及梯度诊断成本核验。外置wrapper仅静态时不得称最终PASS。每轮按AGENTS精确审核轻量文件后publish_mlr_branch非force推MLR、核远端和receipt。监控按实测预计整段训练/评估结束点，不固定短轮询。
