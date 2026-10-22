# PARTIAL：TAC-R / TAC-R＋TAC-ST 已排队

观察2026-10-23 00:15:45 CST。snapshot_readonly_20261023_001540/LOCAL_AUDIT PASS：新controller95924实际存活，1462冻结hash一致，六个输出均不存在、尚未GPU准入，无失败。唯一活跃GPU为旧ALIGN42评估94045，未中断。新队列等待旧ALIGN OBJECT42/43及其条件GOAL分支自然有效结束，故没有新实验训练吞吐或完成ETA，不能拿CPU测试伪造GPU进度。

两组为ALIGN＋TAC-R与独立官方初始化的ALIGN＋TAC-R＋TAC-ST。各自完整seed41 AUC严格>原RA41 701/960（73.020833333333%）才自动扩展自身42/43。不是三seed RA均值，不增加其他性能门槛；R未过也照常运行RST41。旧ALIGN→GOAL原三seed均值门槛保留。

三seed新图Task8/9独立支持为156/207、152/205、180/200；零训练身份、动作前观测/下一动作/下一效果、RA成员SHA及新增夹爪sign条件检查PASS。实际Temporal/GMM CPU梯度与部署融合PASS；不代表GPU准入通过或闭环收益。每run预算仍400epoch105500主步，新增105500表征optimizer调用独立计，4主CP100键＋4部署14键、200正式20vector最终验收待后续。

六run及旧待运行四run最坏起步容量界通过，普通system留1GiB、data root留2GiB；不删旧资产、不下载payload。每run启动前重新核容量。当前没有新性能结果，全部科学结论待完整验收，最终保留所有负结果及探索性选择限制。

2026-10-23 00:21补充：六入口15模块实际独立环境导入PASS、无输出/模型前向/优化器/环境，1462冻结hash不变；同一监控保存的30分钟调度与完整新旧队列提示已复核。最终包装仍待完成实验真实验收。
