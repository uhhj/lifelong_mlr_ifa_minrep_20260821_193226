# 实验 B：人工标注准备

`original_experiment_b.py` 和 `USER_PROTOCOL.md` 是用户提供文件的原样副本。
尚无人工基础标签，因此当前只执行输入审计、合同测试和基础任务标注包准备。
不生成 `frozen.json`，不对增量任务输出 SHARE/BIF，不训练模型。

```bash
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
python -m unittest abcr_pair_validation_b_v1.tests -v
python -m abcr_pair_validation_b_v1.prepare \
  --root /data/lifelong_mlr_ifa_minrep_20260912
```

打开 `reports/abcr_pair_validation_b_v1/annotation/index.html`，独立人工判断
局部状态和相关物体配置是否可比，导出 CSV。不能使用余弦或动作类别生成标签；
不能为凑正负样本配额强行标注。不确定项标 U，导出后 label 为空而不是负例。
图片只为浏览从 OpenGL 存储方向上下翻转；这不是模型的图像预处理定义。

包设计在看标签前冻结：task0–5，demo0–24 校准、demo25–49 验证；
每 8 帧取锚点，排除未来不足 3 帧及超机械行程的锚点并计数。
15 个跨任务组合各选最多 20 个校准对和 10 个验证对，同一集合内每 demo 对最多一次。
均通过协议位姿门，从合格池按固定随机种子选取，未使用视觉分数/动作相似度。
这是一份固定的待标注样本包，不保证标注后必有至少 20 正/40 负；
不足时应报告，不自动补采到阈值通过。

后续仍须：导出同一基础 checkpoint 的 pre-FiLM 特征、核验原 `.pt` 与图像对齐、
仅从人工校准 CSV 冻结视觉阈值、在独立验证轨迹报告性能；然后才允许增量筛查。
direct-HDF5 基础锚点不冒充已验证的增量 `.pt` 映射。

`formal_training_resumed=false`。
