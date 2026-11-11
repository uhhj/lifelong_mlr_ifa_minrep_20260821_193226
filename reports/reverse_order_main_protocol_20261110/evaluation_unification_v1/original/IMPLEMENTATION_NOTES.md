# 逆序输入与主表协议继承

两种方法逐文件绑定主结果表实际来源。MLR+IFA保留原Random示范、窗口末动作、PaperMLR.observe、IFA与原生global-Normal评估；完整DEBR保留RA成员、合法next-action监督、执行专家项、固定获得时成功缓存、支持度校准、C/Q候选门控及其原生逐轨迹Normal评估。没有把当前stripped-S实现作为新MLR基线。

顺序映射在新进程内注册index21，原order0和基底任务0–5不变。学习位置6/7/8/9对应物理任务9/8/7/6；特征只做路径别名，语言和物理初态随原任务，标签task只重编号；每个新阶段重新生成混合清单和IFA。Random成员按各主表seed的原规则核验，RA成员绑定原冻结前端选样。不复用正序模型、教师、成功缓存或阶段IFA。

预备工程史完整保留：v1因历史seed白名单在CPU导入前退出；engineering02因旧LIBERO依赖可解析EGL编号而CPU导入退出；engineering03因准备检查错误使用固定42而拒绝seed41成员。以上均0优化器、0模型前向、0环境，没有开始任何科学run。旧根和文本bundle保留。engineering04只修这些准备兼容与核对逻辑，实际CPU准备仍以PREPARATION_COMPLETE和本地独立审计为准。

CPU检查保留旧库导入时的EGL设备0编号，所有张量map_location=cpu，检查torch.cuda.is_initialized为False；不构造仿真环境。GPU阶段的自身准入和最终CPU载入是不同环节，不能互相替代。

训练计数通过AdamW.step包装累计物理调用，隔离准入单列且完整恢复原状态/RNG。逆序人口导致学习率终点和预算重算；实际科学停止条件不随结果修改。最多两个训练，采集与正式环境通过共同文件锁互斥。当前旧队列先自然完成并实际独立验收、分析和发布，新GPU入口才允许执行。

最终外置CPU和本地复算路径先静态编写，只有实际调用成功才记接受。所有CP/动作/特征/初态/RNG/成功缓存payload留在服务器，只返回轻量代码、标量、身份与SHA、episode outcomes。

engineering05补充：逐一读取真实冻结native build函数发现OBJECT_DEBR_41依赖仍写死order0（只验证独立benchmark对象不足以覆盖实际训练工厂）。已退休未启动的04完整源码/准备合同，旧root不覆盖。新case-local build_ordered使用原factory原全局变量，仅把该处order0改为cfg索引；其余五个DEBR工厂源函数不变。12个实际GPU初始化返回的benchmark必须再逐任务核对原9→8→7→6和基底0–5，且在任何优化器更新前写ACTUAL_BUILD_MAPPING。损失、模型、优化器、选样和预算不变。新的CPU导入proof尚待实际完成与本地审计，不能套04的PASS。
