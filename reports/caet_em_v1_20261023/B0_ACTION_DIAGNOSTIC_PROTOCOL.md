# B0已有checkpoint分步动作与GMM诊断

2026-10-24用户明确授权：已有checkpoint补充第一步与未来三步动作诊断、分量选择和方差；不重训、不改当前实验。本文件在新增前向之前登记分析范围，不改变冻结科学合同。

- 仅B0 Task6/7/8/9四个完整checkpoint，原阶段保存的训练人口manifest、原current与old-RA样本、原四步action mask。全部使用已有标签和原生feature，不创建环境，不重评闭环，不读取未来未授权数据。不扩展到B1/B2，不改变其正在进行的工作。
- 独立进程CPU FP32，单线程、低调度优先级、eval/no_grad，批32与原诊断一致，CUDA隐藏。服务器不保存脚本、模型、缓存或报告，不修改训练文件/合同/进程/RNG；只读输入，JSON标量流回本地。CPU/I/O仍有额外成本，单独记录，不宣称零资源影响。
- 每阶段实际CPU严格加载全部B0参数；冻结源码/CP/codec/数据manifest及输入身份mask绑定。原诊断四步grouped Huber按样本复算，任务级均值与原GPU记录比较，预先记录容差abs<=1e-4+1%*abs(original)，同时报告全部差值与分量切换可能性，不把容差通过当闭环验收。
- 对每个合法预测位置k=0..3记录误差标量：selected-component均值的分组Huber、平移/旋转/夹爪MAE/MSE；selected Normal各组sigma RMS、1/2sigma经验覆盖；其采样期望MSE=均值MSE+sigma²，无随机采样。单位为原动作单位，不是毫米物体定位误差。训练mask无效的位置不进入统计。
- GMM分量诊断：最高权重/前二权重差/归一化熵、所选分量编号、其目标后验责任、目标后验最大分量、事后最小均值误差分量与差距、mixture与selected-component NLL、前二分量均值距离。事后oracle只用于解释且不可部署，不能按这些结果替换分量规则。
- 每组sigma分布、标准化残差覆盖、Normal采样落出[-1,1]参考区间的解析概率均为离线诊断；区间概率不等于环境失败率，不据此改变采样、裁剪或温度。误差按task/stage/horizon和current/old区分，后3步既保留各步也汇总。每条仅保存误差/概率/方差统计等标量、身份hash；不回传原action/state/feature/模型/逐分量向量。
- 本地独立验证所有标量有限、合法位置完整唯一、manifest身份一致、CP绑定、误差公式及聚合，记录各阶段前向窗口/调用数/CPU时间，0optimizer/0environment/CUDA。现有数据是训练演示与旧RA，不能伪称独立测试泛化或新闭环收益；没有新留出演示对照。
- 完成后报告支持/不支持的机制与剩余限制，按既有publish_mlr_branch精确审核轻量文件推MLR并核remote/receipt。不更改训练、主实验监控或最终准入条件。
