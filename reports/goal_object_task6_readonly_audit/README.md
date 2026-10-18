# GOAL / OBJECT Task6 入库与遗忘只读审计

[完整报告](REPORT.md) · [结果与所有任务矩阵](RESULTS.json) · [独立验收](INDEPENDENT_ACCEPTANCE.json)

仅分析已有数据，没有新增训练、模型前向或闭环。核验10,801个feature、10,804个实际buffer文件、64份RNG快照；从3,600条既有闭环记录复算。仅派生统计和轻量证据发布，无原始载荷。

主要发现：GOAL的Coreset几何覆盖更好，但Task6末步监督明显偏向中后段正夹爪命令。相同偏移在OBJECT小得多。GOAL总体AUC下降主要由Task6保持损失贡献，不能解释为新IFA配对、损坏缓存或普遍漏采夹爪转换。尚不能证明频率偏移是唯一原因，也没有验证修复方案。

在本目录运行（Python标准库，无服务器/模型依赖）：

```sh
python -X utf8 analyze.py
python -X utf8 build_report.py
python -X utf8 independent_verify.py
```

三个脚本只读取归档证据并重建本目录派生结果，不访问模型或环境。`collect_readonly.py`、`inspect_joint_rng.py`、`verify_buffer_readonly.py`为真实服务器只读采集代码，依赖原远端资源路径，归档用于审计，不能误当可在任意电脑直接运行的实验入口。`prepare_local_evidence.py`记录初次本地证据收集来源。

EVIDENCE_MANIFEST追踪历史轻量输入；TRANSFER_MANIFEST核验远端传输字节；ARTIFACT_MANIFEST覆盖发布文件（不含自身）。历史首次OBJECT43保留，repeat1单独作敏感性分析，不改写旧实验报告。
