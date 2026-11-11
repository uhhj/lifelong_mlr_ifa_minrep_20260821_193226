# cuDNN设置、随机噪声与实验解释

2026-11-11。仅源码与官方文档核对，没有新训练、模型加载、环境或SSH调用。

## 实际设置

历史 `20261013_coreset_extension/seed43/source/coreset_exp/evaluate.py:23` 和OBJECT原入口显式设置 `torch.backends.cudnn.benchmark=True`。后期完整S `seed41/runtime.py:18`、C runtime和新剥离runtime设为False。原 `libero/lifelong/utils.py:19–24` 的 `control_seed` 设置 `torch.backends.cudnn.deterministic=True`，旧、新评估均调用它。

所以不能把此变化描述为“旧实验开启cuDNN，新实验关闭cuDNN”，也不能描述为“旧实验无确定性限制，新实验才开启确定性”。变化的是benchmark算法搜索；enabled、deterministic和allow_tf32是不同设置。

|设置|功能|
|---|---|
|cudnn.enabled|控制是否使用cuDNN后端|
|cudnn.benchmark|为输入尺寸测试多种卷积算法并选择最快者|
|cudnn.deterministic|限制cuDNN卷积使用确定性算法|
|cudnn.allow_tf32|控制支持设备上cuDNN卷积能否使用TF32|

官方PyTorch1.13文档说明，benchmark启用时的计时与硬件差异能改变被选中的算法；关闭可固定算法选择方式，但不单独保证运算确定性。不同算法以及输入batch形状仍可产生不同浮点结果。已有deterministic=True限制时，更不能将benchmark变化说成从“任意随机卷积”变为“确定性卷积”。benchmark也不是TF32或FP16开关。

依据：[官方1.13后端说明](https://github.com/pytorch/pytorch/blob/v1.13.0/docs/source/backends.rst)、[官方1.13复现说明](https://github.com/pytorch/pytorch/blob/v1.13.0/docs/source/notes/randomness.rst)、[官方1.13 CUDA／TF32说明](https://github.com/pytorch/pytorch/blob/v1.13.0/docs/source/notes/cuda.rst)。

## 实验意义

这些评估设置主要是执行层变化，不是新增的学习模块，不直接增加模型容量、改变训练目标、训练步数或成功阈值。不能因为存在该开关差异，就推断方法优势无效，或要求为此重跑全部实验。反过来，也没有受控证据可把其闭环影响定为零，或给它分配一个具体AUC差额。缓存、硬件、环境生命周期和随机流同时变化的结果不能唯一归因于benchmark。

固定逐轨迹噪声仍沿用标准Normal及argmax分量mu+sigma×epsilon的名义动作分布；不是减少动作方差、取消采样或用专家筛动作。它将随机输入配对，不能从设计上推出必然偏向DEBR或必然对DEBR更不利。每格20回合的有限噪声实现可使具体方法成绩升降，方向须由实测结果判断。

更准确的结论是：工程差异不等于算法证据全部失效；规则不同也不应直接宣称全体结果完全等价。后期C/Q/S采用共同B规则，其内部对照保留意义；A/B跨版本仍需保留来源，不能把整段差值无条件写成单模块因果。动作目标、合法mask和新基线训练入口重写则是另外的训练差异，不应借“cuDNN属于工程设置”一并忽略。
