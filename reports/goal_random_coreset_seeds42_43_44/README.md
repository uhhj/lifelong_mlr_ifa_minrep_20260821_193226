# GOAL Random vs Coreset＋IFA：阶段性结果

状态 **PARTIAL**，观察时间2026-10-18 12:32（北京时间）。seed42/43两组各200条正式评估完成并已通过服务器独立验收；seed44两组训练完成，12:29进入Random评估。此发布不是三seed最终验收，不以预计结束时间代替真实结果。

|Seed|方法|AUC↑|FWT↑|NBT↓|
|---|---|---:|---:|---:|
|42|Random|80.9375|82.5000|4.4444|
|42|Coreset|78.7500|81.2500|5.5556|
|43|Random|79.0625|76.2500|−6.6667|
|43|Coreset|78.4375|83.7500|11.6667|

AUC差分别−2.1875pp和−0.625pp；目前不支持宣称GOAL收益，不能据未完成的seed44结果作三seed结论。所有任务矩阵和精确分数见PARTIAL_RESULTS.json。

从仓库根目录运行只读复算：
```sh
python -X utf8 reports/goal_random_coreset_seeds42_43_44/recompute_partial.py
```

该脚本验证146个轻量文件哈希并从800条真实记录复算，不导入训练框架、不加载checkpoint、不运行环境。source/保留三seed执行源，仅用于审计来源，勿把远端硬编码路径当本地执行入口。

三模态启用、GOAL gamma0.7、100完整epoch、batch10、原0.5入库配额、原GMM动作采样。官方预训练模态元数据unknown、逐step调度/warmup与ep99规则为明确补全；全局动作RNG每阶段重置，未逐轨迹隔离。详见PROTOCOL.md/spec.yaml。历史state_v2仅作参考，未冒充本轮共同实现Random基线。

仅代码、配置、哈希、逐episode结果与尝试账目；没有权重、演示、latent或buffer载荷。后续监控将更新进度与最终验收；历史观察时间保留。
