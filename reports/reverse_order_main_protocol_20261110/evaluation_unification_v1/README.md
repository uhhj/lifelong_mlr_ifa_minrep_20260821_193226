# 当前逆序实验共同评估修订

用户授权：2026-11-11，“MLR+IFA同样采用完整方案协议”。修订在任何正式回合前完成，不新增run、训练更新或环境预算。

- 六个MLR case沿用各自原训练、Random、BC+0.1IFA、100轮／任务、官方起点和实际模型；未加入DEBR组件，也未改写训练。
- 正式rollout复制完整DEBR的evaluate.py，仅更换runtime适配器与arm标签；共享缓存、Normal、done／环境生命周期、cuDNN benchmark=False、20初态／600步／5zero等行为。
- 原root合同／入口保存于original。新root合同的父子关系、逐文件SHA和用户修订由根EVALUATION_PROTOCOL_AMENDMENT.json绑定；12份训练case合同保持不变。
- 六case CPU配置／导入和架构AST一致通过。已完成两个MLR run的8CP实际strict100键载入通过；16次CPU前向另列。未来未产生的MLR检查点没有提前验收。
- 首次CPU探针缺shape_meta而失败，源与真实失败保留；没有训练或环境消耗。修复检查后实际通过，不以静态审查代替运行证据。
- 每个MLR evaluator的长度1–8 GPU缓存准入在首个正式回合前自行执行；未通过不得进入环境，正常负结果不重试。
- 原定正式总数2400，训练采集240不变。最终真实CPU及本地验收需核两方法共同初态／noise／env_seed／attempt，报告配对改善／退化与完整成本。
- 原顺序历史MLR仍为旧评估，不能把新旧顺序的MLR差全部解释为顺序效应。本次没有追加原顺序重评。

当前状态和证据：LOCAL_AMENDMENT_AUDIT.json、根APPLIED记录及最新RUN_STATUS。不是训练／闭环最终接受报告。
