# OBJECT 零响应残差五seed结果与只读问题分析

[完整报告](FINAL_REPORT.md) · [问题分析](FAILURE_ANALYSIS.md) · [精确结果](FINAL_RESULTS.json) · [验收](LOCAL_FINAL_ACCEPTANCE.json)

seed40/41完成；合并已发布42–44，五seed共1000条正式记录。主比较seed43用repeat1，首次敏感性比较完整保留。AUC平均差−3.375pp；没有稳定互补收益。旧报告不覆盖。

从仓库根目录复算（只读取轻量记录；不加载模型/运行环境）：
```sh
python -X utf8 reports/object_zr_coreset_five_seed/final_local_audit.py
python -X utf8 reports/object_zr_coreset_five_seed/analyze_failure.py
```
前者使用已发布的reports/object_zr_coreset_seeds42_43_44中的旧证据，只在本目录更新派生结果；后者仅输出任务贡献。source中冻结训练代码仅归档，勿将它误当本地复算入口。无权重/数据载荷。
