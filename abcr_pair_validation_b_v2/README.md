# ABCR B v2：基础任务视觉校准与轨迹隔离验证

继续原实验 B，唯一已获授权的研究协议修改是使用明确标为 AI 的视觉标签。
不将 AI 标签作为人工或客观物理真值。原脚本和原协议保持原文件 SHA。

`export_features.py` 只加载冻结基础 checkpoint 的 CLIP 视觉模块、LoRA 与 image_proj。
使用正式代码的图像预处理与 forward，输出 FiLM 之前的 512 维表示；不构造策略或初始化 CUDA。
逐锚点重新核对 HDF5 像素、姿态、夹爪与未来动作。这里输入是直接 HDF5 锚点，
不是增量 `.pt` 对齐的替代证明；后者仍须另行核验后才能筛查增量候选。

`evaluate.py` 先调用未修改原脚本冻结负类 Q0.95+higher+1e-7 阈值并写入封存文件，
然后才计算验证指标。动作 IQR 只读基础校准角色的唯一锚点，不读验证动作。
calibration 与 validation 的 demo 互斥；基础编码器原先训练使用基础数据，
因此这里的隔离仅是标注校准与验证隔离，不声称视觉编码器没有见过验证演示。

参数、输入和代码哈希绑定缓存；每个锚点原子保存，可重启；不同绑定拒绝复用。
不改阈值、不改标签、不新增训练。验证质量不足时依原协议停止增量筛查。
原协议没有定量 precision/recall 通过线，报告不可事后编造一个 PASS 阈值。

```bash
python -m unittest abcr_pair_validation_b_v2.tests
python -m abcr_pair_validation_b_v2.export_features --repo REPO --out REPORT
python -m abcr_pair_validation_b_v2.evaluate --out REPORT --labels AI_LABEL_REPORT
```

运行前继承历史环境激活方式，设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、
`CUDA_VISIBLE_DEVICES=''`，将本隔离代码目录加入继承的 PYTHONPATH。
只使用已有依赖及完整 CLIP 本地快照。
