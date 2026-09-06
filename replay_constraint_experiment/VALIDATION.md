# 交付验证记录

日期：2026-09-06。

## 已完成

- Windows本地，Python3.12，PyTorch2.5.1+cpu；仅用于测试，不修改服务器Python3.8/PyTorch1.13复现环境。
- 16项unittest全部通过。
- 冲突投影、对齐/正交不变、整体而非分块投影、零/极小参考、无旧样本/纯旧样本、非有限梯度拒绝、loss_scale与裁剪、随机向量投影与幂等性通过。
- 关闭投影时，含dropout和GMM NLL的小模型经过多个AdamW步骤后，损失对应更新、参数、AdamW状态、调度状态、随机状态精确一致；未使用参数保持grad=None。
- 从工作区`publish_repo`实际读取GMMHead和BasePolicy.compute_loss的代码，验证`reduction=none`再取均值与原mean损失及梯度精确一致。该项使用小维度输入，不等同于完整机器人模型运行。
- 实际新增子类的observe及epoch/step日志通过合成策略集成测试。
- 固定ep99矩阵解析、拒绝缺失/重复/其他epoch、原FWT/NBT/AUC计算和启动参数通过。
- 安装器新增范围、重复安装幂等与拒绝覆盖不同文件通过。
- 所有Python源文件通过Python3.8语法解析。
- 使用工作区实际基线文件构成的临时接口样本，安装预览、安装、再次安装及训练命令预览通过。原工作目录中的基线文件没有被修改。

## 未执行

- 服务器Python3.8/PyTorch1.13上的完整真实策略一致性检查。
- 单GPU torchrun训练入口冒烟检查。
- 正式三seed训练、MuJoCo rollout和实验效果比较。

`parity_real.py`和`run.py --smoke`已提供相应检查入口。真实一致性检查要求逐项精确一致，不自动放宽容差。正式结果目前不存在，不根据数学单元测试推断成功率或遗忘改善。

## 运行测试

在服务器既有环境：

```bash
python -m unittest discover -s replay_constraint_experiment/tests -v
```

如果服务器没有本机的`publish_repo`快照，实际GMM源代码小模型测试会明确跳过；随后必须执行README中的`parity_real.py`，它直接读取服务器目标模型和真实缓存。其余核心测试不依赖LIBERO或MuJoCo。
