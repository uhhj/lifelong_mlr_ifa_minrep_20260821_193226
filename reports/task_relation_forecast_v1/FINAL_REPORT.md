# 任务关系信号前瞻预测实验 v1：最终报告

完成12个源模型、36个有序seed×任务对、22800次诊断更新和144×20=2880次有效闭环。正式lifelong训练未恢复。

## 冻结的主比较

|目标|候选−基线|OOF MSE差|区间|4任务折同向数|判定|
|---|---|---:|---|---:|---|
|TRANSFER_300|M_A−M_IFA|-0.00648507|0.975 -0.0693445, 0.0486034|2/4|INCONCLUSIVE|
|TRANSFER_300|M_LA−M_A|0.0285613|0.975 -0.151216, 0.117644|0/4|INCONCLUSIVE|
|TRANSFER_300|M_GA−M_LA|0.00182336|0.95 -0.0880104, 0.138286|2/4|INCONCLUSIVE|
|DAMAGE_300|M_GA−M_LA|-0.0416476|0.95 -0.166995, -0.0324356|4/4|INCONCLUSIVE|

## 全模型的严格留任务结果

|目标|模型|MSE|MAE|Spearman|
|---|---|---:|---:|---:|
|TRANSFER_300|M0|0.183642|0.331276|0.1752849947130735|
|TRANSFER_300|M_IFA|0.195105|0.356483|0.1582845396784477|
|TRANSFER_300|M_A|0.18862|0.330628|0.13832188414536437|
|TRANSFER_300|M_LA|0.217181|0.374156|0.0808809527404924|
|TRANSFER_300|M_GA|0.219004|0.37087|0.020735403489202664|
|TRANSFER_300|M_probe|0.183642|0.331276|0.1752849947130735|
|DAMAGE_300|M0|0.223842|0.368673|0.29288525703948803|
|DAMAGE_300|M_IFA|0.214454|0.380925|0.37971863941286926|
|DAMAGE_300|M_A|0.150028|0.295104|0.6322192773099081|
|DAMAGE_300|M_LA|0.194711|0.337179|0.483067137601693|
|DAMAGE_300|M_GA|0.153064|0.310215|0.6264131819060416|
|DAMAGE_300|M_probe|0.223842|0.368673|0.29288525703948803|
|TRANSFER_0|M0|0.0245361|0.111103|-0.3153839096704965|
|TRANSFER_0|M_IFA|0.0190554|0.0996675|-0.0016937911367910662|
|TRANSFER_0|M_A|0.0190201|0.0903511|0.022019284778283858|
|TRANSFER_0|M_LA|0.0242242|0.112918|-0.11822662134801643|
|TRANSFER_0|M_GA|0.027556|0.11454|-0.1866557832743755|
|TRANSFER_0|M_probe|0.0245361|0.111103|-0.3153839096704965|
|ACQUISITION_DIFF_300|M0|0.15767|0.310808|0.35767656224044464|
|ACQUISITION_DIFF_300|M_IFA|0.193316|0.360798|-0.00978877159894627|
|ACQUISITION_DIFF_300|M_A|0.162713|0.314025|0.2020866136677197|
|ACQUISITION_DIFF_300|M_LA|0.198659|0.364092|0.13420921060660546|
|ACQUISITION_DIFF_300|M_GA|0.191568|0.36294|0.10381249879935124|
|ACQUISITION_DIFF_300|M_probe|0.15767|0.310808|0.35767656224044464|

## 结论与边界

```json
{
  "protocol_status": "COMPLETE",
  "MEASUREMENT_VALIDITY": "PASS_WITH_DISCLOSED_PROVENANCE_LIMITATIONS",
  "TARGET_VARIATION": {
    "TRANSFER_300": {
      "min": -0.8,
      "max": 0.9,
      "std": 0.43033820678882584,
      "unique": 27
    },
    "DAMAGE_300": {
      "min": -0.85,
      "max": 0.95,
      "std": 0.4721629864808011,
      "unique": 21
    },
    "TRANSFER_0": {
      "min": 0.0,
      "max": 0.5,
      "std": 0.13559283412727482,
      "unique": 6
    },
    "ACQUISITION_DIFF_300": {
      "min": -0.8,
      "max": 0.85,
      "std": 0.41407527496016144,
      "unique": 26
    }
  },
  "ACTION_AFFINITY_PREDICTIVE_VALUE": "INCONCLUSIVE",
  "STRUCTURE_PLUS_ACTION_INCREMENT": "INCONCLUSIVE",
  "GN_INCREMENT_OVER_CHEAP_STRUCTURE": "INCONCLUSIVE",
  "ONE_STEP_PROBE_REFERENCE_VALUE": "UNINFORMATIVE_ZERO_FIRST_STEP_LR",
  "TRANSFER_FORECAST": {
    "M_IFA": "INCONCLUSIVE",
    "M_A": "INCONCLUSIVE",
    "M_LA": "INCONCLUSIVE",
    "M_GA": "INCONCLUSIVE",
    "M_probe": "INCONCLUSIVE"
  },
  "OLD_DAMAGE_FORECAST": {
    "M_IFA": "INCONCLUSIVE",
    "M_A": "INCONCLUSIVE",
    "M_LA": "INCONCLUSIVE",
    "M_GA": "INCONCLUSIVE",
    "M_probe": "INCONCLUSIVE"
  },
  "GENERALIZATION_SCOPE": "Four fixed tasks, 12 ordered pairs, 3 optimization seeds sharing one pretrained base; no population-level or native-MLR controller conclusion",
  "RESOURCE_COST": {
    "sources": 358.96811051200166,
    "transfers": 328.39667870300764,
    "signals": 1285.6035921559806,
    "valid_behavior": 63129.543846962006,
    "analysis": 7.229250613992917
  },
  "diagnostic_training_performed": true,
  "formal_training_resumed": false
}
```

M_probe的R_old/R_new全部为零：保留原生第一步LR=0，没有换后续步制造信号，因此本轮不能判断非零影子适应是否有价值。

主区间由5000次seed整块＋共享初态配对重采样、重新拟合产生；两个TRANSFER主比较各用97.5%区间。其他比较95%且属次级描述，不作多重比较后胜者搜索。

原官方基座配套训练配方与检查点来源一致，但原作者训练job日志不可得，暴露核验是来源证据而非直接原日志验证。

行为配对曾因非策略输入的辅助四元数字段微小变化误拦；保留失败证据，仅修正工程检查范围。有效结果要求初始状态、原生策略输入和RNG配对检查通过。无阈值或试验预算调节。

有效rollout每模型任务仅20次；单个trial差异不自动构成正负迁移证据。36行只有12种有序关系，三个seed不是独立预训练种子。

本轮无回放、1000步源预算/300步目标预算，不代表完整MLR或终身训练收益。GN未显示相对廉价结构增量时，不扩展rank/Fisher挽救；无可靠关系增量时不扩建控制器。

逐次结果见ROLLOUTS.csv，绝对成功率/条件迁移/源损伤见TARGETS.csv；离线逐轨迹均值见OFFLINE_NLL.csv；逐窗口原始值保留在offline/*/WINDOW_NLL.csv。

统计正确性：独立增广最小二乘重算全部OOF预测，最大误差 2.22e-15；原始成功数组独立重算标签；固定bootstrap前三次与冻结实现一致。

formal_training_resumed=false

## 研究结论的直接解释

1. 固定目标预算的迁移有充分观测变动（-0.80至+0.90），不是所有结局落在同一地板。源任务退化为-0.85至+0.95；不能合并为同一个迁移标签。
2. 迁移预测的最简单难度/掌握度基线M0，OOF MSE=0.183642；IFA=0.195105；动作相容性A=0.188620；latent+A=0.217181；GN+A=0.219004。关系信号没有展示可靠的基线外增量。
3. A相对IFA的MSE差为-0.006485，97.5%区间[-0.069344,0.048603]；仅2/4任务折同向。结论INCONCLUSIVE，不声称A优于IFA。
4. latent+A相对A的MSE差为+0.028561，四个留任务折均变差，但区间跨0。按协议仍记录INCONCLUSIVE；这不构成结构增量支持。GN+A相对latent+A也没有建立迁移预测增益。
5. 源任务损伤预测有描述性正面趋势：A的MSE=0.150028、Spearman=0.6322，M0为0.223842、0.2929。完整区间和seed稳定性尚不足以判SUPPORTED。不要把这个趋势混入迁移结论。
6. M_probe与M0等效，因为首步LR=0，R_old/R_new均为0。这里只能判该冻结探针无信息，不能推断非零适应反馈无效。
7. 删除一个任务后的敏感性分析中，IFA训练相似度近常数而非精确常数，导致极端外推（最大预测绝对值约403447）。未裁剪预测、未调整零方差门，见TASK_DELETION_SENSITIVITY.csv。这是预测设计的局限，不是实际成功率。
8. 不允许据此恢复正式训练、扩建门控或用新的rank/Fisher变体挽救本轮。若继续研究，需要独立协议；本轮主结论是前瞻关系预测增量未建立。

## 独立原始数据核验

直接重算184320条离线窗口记录（包含共享端点在各对中的重复评价），逐轨迹汇总误差为0；不把这些窗口计作独立关系样本。
逐条检查36分支×300步的batch、LR、优化器和scheduler计数与共享目标参考一致。直接从逐trial数组重算四种标签，最大误差0。
主OOF独立增广最小二乘误差2.22e-15。任务删除的极端预测导致绝对差1.16e-10，但相对差仅1.45e-14；原判据/数据未改变。

源模型自身成功率见SOURCE_MASTERY.json；0/20的源任务原样保留，不把无下降空间解释为保护。

已完成全部测量、5000次配对重采样和独立核验。formal_training_resumed=false。
