# LIBERO-GOAL Random / Coreset＋IFA 三 seed 最终结果

**COMPLETED，2026-10-18。** 2,400 epoch、456,600 step、24 个最终 checkpoint、1,200 条正式记录、120 次 vector attempt；服务器独立加载验收及本地轻量证据复算全部通过。

[最终报告与全部矩阵](FINAL_REPORT.md) · [精确复算结果](FINAL_RECOMPUTED.json) · [服务器最终验收](final_evidence/queue/FINAL_QUEUE_ACCEPTANCE.json) · [最终进程快照](FINAL_SNAPSHOT.json)

|Seed|Random AUC|Coreset AUC|C−R（pp）|
|---|---:|---:|---:|
|42|80.9375|78.7500|−2.1875|
|43|79.0625|78.4375|−0.6250|
|44|79.0625|72.8125|−6.2500|
|均值 ± 样本SD|79.6875 ± 1.0825|76.6667 ± 3.3415|−3.0208 ± 2.9036|

三 seed 均未显示 Coreset 总体 AUC 增益，退步共同集中于 Task6 保持。三 seed、原全局动作 RNG 未逐轨迹隔离、IFA 可间接受影响，不能断言普遍因果。官方预训练模态元数据未知、调度补全、3090 与论文 A100 差异均在报告披露。

从仓库根目录只读复算（标准库，不加载权重或环境）：
```sh
python -X utf8 reports/goal_random_coreset_seeds42_43_44/recompute_final.py
```

final_evidence/PACKAGE_MANIFEST.json 保留服务器原包941项哈希；PUBLICATION_MANIFEST.json 覆盖本报告目录所有审阅文件（不含自身）。冻结源码仅供审计，其硬编码远端路径不是本地执行入口。仅发布代码、配置、哈希、结果及尝试账目，没有权重、演示、latent、buffer载荷或归档包。

[历史阶段性报告](HISTORICAL_PARTIAL_README.md) 和 progress/ 下各时间点完整保留；这些 PARTIAL 状态是历史观察，并非最终状态。
