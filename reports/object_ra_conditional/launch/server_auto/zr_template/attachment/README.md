# ZR-v2 / seed43 执行设计包

先读 `PROTOCOL.md`，`spec.yaml` 是机器可读冻结配置。当前状态为**设计完成，真实校准与闭环尚未运行**。

- `reference_core.py`：双向门递推、7维有界均值修正、数值前向不变的训练梯度归属参考。不是LIBERO完整实现。
- `compare_metrics.py`：可直接使用的离线结果/父模型身份检查与精确指标复算器。
- `test_reference.py`：25项**合成CPU**测试；本容器测试通过，输出见`TEST_OUTPUT.txt`。不代表真实指令/父GPU环境已验收。
- `SOFTWARE_TEST_STATUS.json`：软件验证范围及本机版本。
- `MANIFEST.json`：本执行包文件哈希；不是远端实验输入验收。

不包含权重、latent、专家示范、实际prototype产物或真实新增episode。

```bash
python test_reference.py
python compare_metrics.py \
  --baseline-dir /path/to/repeat1/coreset/evaluation \
  --new-dir /path/to/zrv2/evaluation \
  --output /path/to/new/METRIC_COMPARISON.json
```

新episode需包含`arm=coreset_ifa_zrv2`、`seed=43`、`stage/task/trial`、`success`布尔值、`parent_epoch=99`、`calibration_epoch=9`、`parent_checkpoint_sha256`、`checkpoint_sha256`、`modal_sha256`、`initial_state_index`和`initial_state_sha256`。父checkpoint必须与比较的repeat1记录相同。

复算器仅检验记录与指标，不加载模型，不检查实际GPU/结构证据，不启动其他seed。正式`zr2_exp`的prepare/calibrate/evaluate/audit等入口仍需按协议集成。
