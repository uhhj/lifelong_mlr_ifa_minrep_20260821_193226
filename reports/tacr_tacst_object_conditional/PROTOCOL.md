# TAC-R 与 TAC-R＋TAC-ST：OBJECT 条件三 seed

用户2026-10-22确认两组独立新实验：ALIGN＋TAC-R（TACR）；ALIGN＋TAC-R＋TAC-ST（TACRST）。后者从官方初始化独立训练，用其与前者比较ST增量，不从前者检查点续训。范围仅OBJECT，seed41先行。每组seed41完整工程验收后，只有AUC严格高于同seed RA41的701/960（73.020833333333%）才扩该组至42、43。不是高于三seed RA均值703/960，不增加其他性能条件，不用四舍五入或部分结果。工程失败不能当性能失败跳过。R不通过也仍运行RST41；某组41通过后其42、43固定完成，不按途中负结果取消。当前已运行ALIGN OBJECT42/43→原门槛→GOAL41/42/43合同不改；新有限队列等待该旧队列有效exit0、所有实际完成receipt及源hash核验、GPU空闲与互斥锁释放后才准入，不中断旧训练/正式评估。旧队列工程失败则新队列留证停止，禁止盲重启或越过。

共同协议：原OBJECT各自RA种子、官方object ep99初始化，八步(8,4,512)、原Temporal Transformer与五模式七维GMM、FP32、batch10；每任务6–9各100epoch，人口1562/2261/3081/3620，每epoch157/227/309/362主步，总400epoch105500主更新。BC使用ALIGN的a[u+1]，身份明确/真实八帧/存在下一动作mask；效果另要求s[u+2]，缺效果不删除BC。保留原DataLoader人口、concatshuffle、DistributedSampler seed0、RA有序成员781/740/780/660、标定42、原选样字段、.1meanIFA/gamma.3/warmup500与逐主步线性调度。初始化主tensor/allRNG、IFA与成员字节必须同父；BC目标改变允许Task6最终不同RA。原RA及已验收ALIGN基线只读不重训重评。

严格按USER_METHOD.txt：h=T(x).last；c=R(stopgrad(h))，R为LayerNorm512→Linear128→ReLU→Linear128→LayerNorm128。Δh=.2·RMS(stopgrad(h))·sigmoid(MLP[stopgrad(h),stopgrad(c)])·tanh(Uc)，RMS平方均值下限1e-16；U128→512首次weight/bias零；门控640→64→1首末层last权重零，bias=log(.1/.9)，初值.1；h_policy=h+Δh。R/U/gate全部部署，完整主100键与部署分支14键分开严格加载，正式推理通过同一融合函数调用原GMM/argmax分量Normal，不允许只加载主干。部署不读专家动作/任务ID/真实物体效果。E为135→128→128含ReLU，D128→7，仅训练使用；效果监督为各样本自身a[u+1]及Effect(s[u+1],s[u+2])，尺度[.01,.01,.01,pi/18,pi/18,pi/18,.01]，SmoothL1，不替换跨物体真标签。

总loss=BC_next+.1IFA+.03effect+.02ST，TACR的ST恒0。IFA只用基础h。BC更新T/G/R/U/gate；effect只R/E/D；ST只R/E；IFA只T。自动微分检查实际GMM及本地结构，自己的GPU准入还需验证原实际Temporal/IFA路径、零残差初始GMM输出精确同ALIGN、U初始BC可学、分支完整恢复。控制优化器原AdamW(lr1e-4,betas.9/.999,eps1e-8,wd.1)，参数T/G/U/gate；表征AdamW(lr3e-4,wd.01)参数R/E/D，严格无重叠；一次联合反向、每主batch各一次optimizer/scheduler。原T/G裁剪100、新U/gate与R/E/D分别裁剪100，不混为全局范数。每阶段优化器与scheduler按原RA重建、500warmup及实际batch×100；全部学生参数跨阶段继承，U不逐任务清零；完整epoch恢复含两优化器scheduler/grads/modes/requires_grad/allRNG及游标/缓存身份。表征更新105500单列，非额外epoch但不称等算力。

RST保留原物理效果/相对末端/开度/互top10/两侧3demo12时刻剪枝，新增专家夹爪指令sign相等必须在邻居选择与剪枝之前；不要求机械臂动作相同。每seed重新零训练全窗口/图审计，不能套178/206。负样本仅合法实际旧RA，不同运动主方向且归一化效果距离≥2；每query按固定身份顺序轮换一个正键最多四负键，温度.1，游标独立保存。无合格对时ST0，BC/effect/RA继续。每阶段使用本run上一阶段单一T/R/E教师，在eval/no_grad/冻结参数下对全部合法旧RA计算128维归一键，绑定样本/featureSHA/主CP+训练分支CP SHA/标签版本/阶段/缓存SHA；全阶段固定，阶段切换整体重建，不用当前网络detach冒充教师，不混不同阶段坐标系。教师缓存不更新、不进入部署、不发布payload。未来任务不进入当前训练；图排除最后demo不等于效果模型留出验证。

每seed独立GPU准入8次优化器调用（5主、3表征）、0环境，通过后从官方初始化正式重建，CPU检查不算GPU准入。主400epoch、4ep99主100键+4部署14键+4训练20键及教师引用链、原成员载荷/IFA/初始化、逐步BC/effect/ST/游标/分组优化账目和200正式20vector10cell完整验收。原globalRNG=s、env[s,s+1]、初态0–19/10lane/600步5zero/阶段重置/argmaxNormal不变；多余分支构建须保存恢复RNG。逐cell/episode记录部署分支SHA。正常失败不补rollout，不盲重启部分globalRNG评估。

资源按实测完整ALIGN41 1,314,732,242字节加新分支/教师/日志余量，每新run上界1.6GiB。最多六run分散三system三data：TACR41/system、42/data、43/system，TACRST41/data、42/system、43/data；新feature父只读链接、buffer逐字节新复制。data root真实f_bfree留2GiB，system仅普通f_bavail且留1GiB；在排队部署及每run起步重新核剩余全队列容量，容量不足留证停止，不使用system保留块、不删旧资产/降精度batch预算/付费扩容。并发只允许当前旧队列占GPU。

服务器独立原episode矩阵精确Fraction决定自动扩展，最终交付仍需轻量白名单全SHA下载和本地独立AUC/FWT/NBT/OldFinal/FinalAll与三seed均值样本SD(ddof1)/配对差复算。比较R对ALIGN、RST对R、各组对RA；保留全部负结果，单seed门槛为探索性选择，不作显著性或保护保证。每轮按AGENTS精确review/stage经publish_mlr_branch非force推MLR并验证remoteSHA/receipt，只代码报告/hash/episode outcome，禁权重/原actions/states/demos/latent/buffer/RNG或缓存payload。整体PARTIAL至实际所选分支全部本地验收发布。
