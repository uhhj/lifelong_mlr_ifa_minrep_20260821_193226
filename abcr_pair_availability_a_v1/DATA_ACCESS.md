# 原始资产与复现

原始结果对应提交 `84094706672f45777c03093bfa25c3fcb21e0c64`。
Git 包含代码和结果，没有公开托管原始 `.pt`、原始 HDF5 或模型。
下列服务器目录是已有私有资产位置，不是公开下载链接；没有这些数据，
不能声称克隆 Git 即完成端到端复现。

服务器工作根：`/data/lifelong_mlr_ifa_minrep_20260912`。

- `features/object_seed{42,43,44}/{6,7,8,9}/*.pt`：逐文件路径、大小、SHA256
  见 `reports/abcr_pair_availability_a_v1/INPUT_MANIFEST.json`。
- `runs/own_reference_control/ifa/seed{42,43,44}/run_001/config.json`：
  原运行配置及 SHA 见同一 manifest 的 `configurations`。
- seed42 的存续 replay 快照位于 `runs/a1_object_v1/seed42/transactions/task{6,7,8}/buffer`；
  PLAN/COMMITTED 哈希见 `STAGE_MANIFEST.json`。每个成员已经与对应源特征逐字节验证。
- seeds43/44 的 replay 路径视图由原 `_list_pt` / `_sample` 和上述源特征确定性重建；
  它们不是存续的历史 buffer 快照。
- 实验 B 所需原始图像/机器人状态另在 `data/libero_object/*.hdf5`；
  不能从 A 的语言条件化缓存逆推出未调制视觉特征。

将上述目录从已授权资产备份按相同相对结构恢复至 `ASSET_ROOT` 后先执行：

```bash
python scripts/verify_abcr_assets.py \
  --asset-root ASSET_ROOT \
  --report-root reports/abcr_pair_availability_a_v1
```

此脚本只读、校验显式清单，不下载或合成缺失数据，不递归扫描历史资产。
校验通过后，在已记录的原 CPU 环境运行 README 命令，输出使用新目录。
重新运行与原 CSV 的字节一致性，不等于外部独立环境复现。
