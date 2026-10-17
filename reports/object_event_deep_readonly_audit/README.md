# Event 深度只读审计

[完整报告](REPORT.md)分析首次 seed40–44 的 Event+Coreset 与 Coreset；seed43 重跑不替换首次结果。本次发布没有训练或新增闭环。

## 本地复算

在仓库根目录执行（仅依赖 Python 标准库）：

```sh
python reports/object_event_deep_readonly_audit/summarize_readonly.py
```

输入为本目录 `DIAGNOSTICS.json` 与已发布的 `../object_five_seed_readonly_analysis/ANALYSIS.json`；`INPUT_SHA256.json` 固定两者哈希。脚本验证输入未变并重建 `SUMMARY.json`。后者上游报告提供既有结果背景，本脚本的任务汇总来自 DIAGNOSTICS。

`collect_readonly.py` 是原服务器采集脚本，按原字节归档，仅供核查采集逻辑。它依赖服务器既有目录、NumPy 和 h5py；本地复算无需运行它，也无需模型、GPU或原始示范。发布时没有再次运行采集脚本。

## 证据边界

- DIAGNOSTICS 含20个seed-task单元、逐事件覆盖、汇总控制量、15个排序位置比较、20个epoch日志比较，以及55份原服务器元数据的SHA256。
- 服务器原元数据读取前后哈希一致；原55份文件未在本目录全部重复打包，离线复算只验证已归档诊断的完整性，不能替代服务器原始输入复核。
- 四个原HDF5以只读模式读取，前后仅核验大小与mtime，未重新计算完整内容哈希。原动作数组未打包。
- 不包含checkpoint、权重、HDF5、latent、buffer或训练数据。事件索引和统计摘要作为轻量审计证据保留。
- 本审计支持机制线索，不构成独立因果识别；沿用原闭环全局动作RNG等解释边界，详见报告。

`PROVENANCE.json` 保存原本地来源及哈希；`PUBLISH_AUDIT.json` 保存复算与发布范围检查。
