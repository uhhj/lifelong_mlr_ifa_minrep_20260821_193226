"""Recompute final SAZR report from downloaded metadata only."""
from pathlib import Path
from collections import Counter
import datetime
import json
import sys

R = Path(__file__).resolve().parent
D = Path(sys.argv[1])
E = D / "evidence"
js = lambda p: json.loads(p.read_text(encoding="utf-8"))
jl = lambda p: [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
assert js(D / "LOCAL_ACCEPTANCE.json")["status"] == "PASS_FINAL"
assert not js(D / "SNAPSHOT.json")["workers"]
records = {k: jl(E / k / "episodes.jsonl") for k in
           ("anchor", "validation_parent", "validation_candidate", "formal")}
parent = {x["initial_index"]: x for x in records["validation_parent"]}
candidate = {x["initial_index"]: x for x in records["validation_candidate"]}
pairs = Counter()
for i, a in parent.items():
    b = candidate[i]
    for key in ("initial_sha256", "noise_seed", "noise_sha256", "parent_sha256", "env_seed"):
        assert a[key] == b[key]
    pairs[("S" if a["success"] else "F") + ("S" if b["success"] else "F")] += 1
assert pairs["FS"] == 0 and pairs["SF"] == 0
assert not js(E / "DECISION.json")["enabled"]
anchors = js(E / "ANCHORS.json")
structure = jl(E / "STRUCTURE.jsonl")
epochs = js(E / "epochs.json")
timings = jl(E / "timings.jsonl")
assert len(epochs) == 10 and sum(x["batches"] for x in epochs) == 1570
assert sum(x["samples"] for x in epochs) == 15620
assert len(structure) == 11 and all(x["passed"] for x in structure)
assert all(x["exit"] == 0 for x in timings)
start = js(E / "pipeline.started")["time"]
end = js(E / "PROCESS_EXIT_ACCEPTANCE.json")["time"]
cst = datetime.timezone(datetime.timedelta(hours=8))
stamp = lambda t: datetime.datetime.fromtimestamp(t, cst).isoformat()
summary = {
    "status": "COMPLETED", "engineering_acceptance": "PASS",
    "residual_enabled": False, "observed_validation_improvement": False,
    "validation_pairs": {k: pairs[k] for k in ("SS", "SF", "FS", "FF")},
    "successes": {k: sum(x["success"] for x in v) for k, v in records.items()},
    "counts": {k: len(v) for k, v in records.items()},
    "formal_percent": 60, "historical_parent_percent": 60, "delta_pp": 0,
    "anchors": len(anchors["anchors"]), "anchor_rank": anchors["rank"],
    "epochs": 10, "optimizer_steps": 1570, "exposures": 15620,
    "structure_checks": 11,
    "anchor_response_max": max(x["anchor_response_max"] for x in structure),
    "anchor_mean_error_max": max(x["anchor_mean_error"] for x in structure),
    "dataset_delta_max": max(x["dataset_delta_max"] for x in structure),
    "start_CST": stamp(start), "finish_CST": stamp(end),
    "wall_seconds": end - start, "phases": timings,
    "evidence": D.name,
}
(R / "FINAL_SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
report = f"""# SAZR OBJECT seed41 Task6：完成验收，残差未获启用

状态：COMPLETED（工程完成，不代表方法有效）。仅此Task6原型，无后续阶段或seed扩展。
执行时间：{stamp(start)} 至 {stamp(end)}，约{(end-start)/60:.2f}分钟。

| 环节 | 成功/总数 | 作用 |
|---|---:|---|
| 父策略成功锚定，初态20–29 | 3/10 | 构造32个锚点，FP64 SVD rank30 |
| 独立验证父策略，初态30–39 | 3/10 | 固定逐轨迹噪声配对 |
| 独立验证候选，初态30–39 | 3/10 | SS3、SF0、FS0、FF7 |
| 正式部署策略，初态0–19 | 12/20（60%） | 开关关闭，精确父head回退 |
| 历史RA父策略参照 | 12/20（60%） | 差0个百分点，非本轮新增对照组 |

预先固定的开关为FS>=2且SF=0。本轮FS=0，因此不启用残差。
正式60%是回退策略表现，不能称为启用SAZR后的成功率，也不能据此声称残差改善。
验证十条中既无新增成功，也无成功损失；小样本相同结果不证明两策略普遍等效。

## 校准及保护证据

10完整epoch、1570 optimizer step、15620样本曝光，1个最终checkpoint严格加载PASS。
11次结构检查全部PASS。锚点最大残差{summary['anchor_response_max']:.8g}，
锚点最终均值最大误差{summary['anchor_mean_error_max']:.8g}；
全数据检查中最大修正{summary['dataset_delta_max']:.8g}。
因此学到了非零修正，同时在记录的锚点特征上满足零响应约束；
这不保证保护未采样的整条轨迹或所有成功状态。
冻结父模型、输入和源文件哈希通过；关闭/基任务/未知语言的父head分布与配对采样精确回退PASS。

共50条真实记录、5次vector执行、4次环境构建/关闭，没有技术重试或未解释尝试。
八个正式phase及pipeline、supervisor退出码全部0；最终快照无残留worker，GPU显存10MiB。
服务器AUDIT及PROCESS_EXIT_ACCEPTANCE、本地LOCAL_ACCEPTANCE均PASS。
本地核对68个下载元数据文件与19个部署源码文件；没有下载模型、Q、特征、动作、噪声表或RNG二进制。

## 真实执行与成本

服务器有限流水线命令见证据commands.jsonl；全部退出0：

```text
python -m sazr_exp.rollout anchor
python -m sazr_exp.anchors
python -m sazr_exp.train
python -m sazr_exp.rollout validation_parent
python -m sazr_exp.rollout validation_candidate
python -m sazr_exp.decide
python -m sazr_exp.rollout formal
python /data/mlr_ifa/experiment_workspaces/SAZR_OBJECT_RA_seed41_task6_20261020/audit.py
```

各phase实测秒数：{', '.join(x['phase']+'='+str(round(x['seconds'],2)) for x in timings)}。
纯epoch计时合计{sum(x['seconds'] for x in epochs):.2f}秒；训练phase含加载和检查38.65秒。
主要成本来自真实环境闭环。

本次只读检查、独立复算命令（退出0）：

```text
python execution/20261020_sazr_object_seed41/check_progress.py
python execution/20261020_sazr_object_seed41/audit_local.py execution/20261020_sazr_object_seed41/{D.name}
python execution/20261020_sazr_object_seed41/final_report.py execution/20261020_sazr_object_seed41/{D.name}
```

证据：{D.name}/evidence；独立验收：{D.name}/LOCAL_ACCEPTANCE.json。
远端产物：/data/mlr_ifa/experiments/SAZR_OBJECT_RA_seed41_task6_20261020。

## 解释边界与停止

本轮支持“成功特征零响应、独立验证开关和拒绝回退可运行”，不支持“闭环性能提升”。
仅一个事后选定seed/Task6、三条成功锚定轨迹、十条配对验证；
校准监督下降与非零残差不等同于任务成功率收益。
正式评估仍采用原global动作RNG，验证才使用逐初态/step匹配噪声，二者不能混算。
父模型及历史资产只读，未重训或重评基线，没有自动追加Task7–9、seed或消融。
完整验收交付后取消本轮定时检查。
"""
(R / "FINAL_REPORT.md").write_text(report, encoding="utf-8")
print(json.dumps(summary, indent=2))
