"""Read-only final report from verified episode evidence; no model/environment calls."""
import json
import sys
from pathlib import Path
from fractions import Fraction as F
from audit_download import main as audit
from flow_metrics import metrics

R = Path(__file__).resolve().parent
D = Path(sys.argv[1])
a = audit(D)
assert a["status"] == "PASS_FINAL"
assert set(a["variants"]) == {"v3"} and not a["variants"]["v3"]["gate"]
E = D / "evidence"
groups = [
    ("RA＋IFA父基线", E / "controls/RA", "ra_coreset"),
    ("RA＋ZR-v2", E / "controls/ZR_v2", "ra_coreset_ifa_zrv2"),
    ("RA＋ZR-v3", E / "v3/evaluation", "ra_coreset_ifa_zrv3"),
]
spec = json.loads((E / "v3/RUN_SPEC.json").read_text())
groups[-1] = (groups[-1][0], groups[-1][1], spec["arm"])
computed = [(name, *metrics(p, 41, arm)[:2]) for name, p, arm in groups]
v = a["variants"]["v3"]
timings = [json.loads(l) for l in (E / "v3/timings.jsonl").read_text().splitlines()]
task_delta = {
    str(t): sum(computed[2][2][s, t] - computed[0][2][s, t] for s in range(t, 10)) / (10-t) / 4
    for t in range(6, 10)
}
assert sum(task_delta.values()) == F(-17, 960)
out = [
    "# ZR-v3 OBJECT seed41最终验收",
    "",
    "状态：COMPLETED（工程通过，效果门槛未通过）。200条正式闭环已完成，服务器controller/supervisor均退出0且无残留worker。按预定规则停止，未创建或执行v3-A；不扩seed或套件。",
    "",
    "本报告由write_final_report.py对逐episode记录进行精确分数复算，并重新调用audit_download.py；两份历史控制组只读复用，没有重训或重评。最终证据目录：" + D.name + "。",
    "",
    "| 方法 | AUC % | FWT % | NBT % | 最终旧任务均值 % | 最终全部任务均值 % |",
    "|---|---:|---:|---:|---:|---:|",
]
for name, m, _ in computed:
    out.append("| " + name + " | " + " | ".join(f"{float(m[k]*100):.6f}" for k in ("AUC","FWT","NBT","OldFinal","FinalAll")) + " |")
out += [
    "",
    "ZR-v3相对RA：AUC −1.770833pp、FWT −1.25pp、OldFinal −5pp；三项推进条件全部不满足。相对旧ZR-v2，AUC +2.8125pp、FWT +7.5pp，但最终旧任务/全部任务均值没有提高。NBT按原定义报告，较小表示遗忘较少；较低起始成绩也可能使NBT看起来较好，不能单独据此判断模型更优。",
    "",
    "| 阶段 | 任务 | RA＋IFA % | RA＋ZR-v2 % | RA＋ZR-v3 % | v3−RA pp |",
    "|---:|---:|---:|---:|---:|---:|",
]
for s in range(6,10):
    for t in range(6,s+1):
        values = [x[2][s,t] for x in computed]
        out.append(f"| {s} | {t} | " + " | ".join(f"{float(x*100):g}" for x in values) + f" | {float((values[2]-values[0])*100):+g} |")
out += [
    "",
    "按任务分解AUC差值（每任务历阶段均值差÷4）：",
    "",
    "| 任务 | 对总AUC差的贡献 pp |",
    "|---:|---:|",
]
for t, d in task_delta.items():
    out.append(f"| {t} | {float(d*100):+.6f} |")
out += [
    "",
    "主要退步集中在Task6：最终45%→25%，阶段7的85%→65%；阶段8则35%→55%，不是每阶段均下降。Task7阶段8下降10pp；Task8最终提高5pp；Task9最终均为45%。这些是结果定位，不能仅凭成功/失败记录断言某段控制或某类动作是因果来源。",
    "",
    "校准新增40完整epoch、10,550次optimizer调用、105,240次样本曝光；4个最终checkpoint严格加载通过；16次完整实际dataset结构检查通过；200条正式记录和20次完整vector attempt，0技术异常。预检恰好4次隔离更新、0环境执行，未混入正式预算。父模型、缓存成员/顺序/载荷与冻结输入哈希通过；117个轻量包文件逐文件SHA256验证通过。",
    "",
    f"实际缓存KL最大值 {v['cache_max_KL']:.17g} nat；闭环在线KL最大值 {v['online_max_KL']:.17g} nat，均满足0.5+1e-12。在线投影 {v['projected_online_states']}/{v['online_states']} 个lane状态（{100*v['projected_online_states']/v['online_states']:.4f}%），包含已结束lane的继续计算，不能当独立episode数或失败概率。零残差还原、非选中均值保护与门结构全部通过。",
    "",
    "投影确实约束了执行分布，但逐状态分布接近并不保证闭环成功率改善。相对v2同时改变了训练目标与执行分量，且条件v3-A没有启动，因此无法隔离投影本身的净收益。",
    "",
    "| 阶段命令 | 退出码 | 实测耗时（秒） |",
    "|---|---:|---:|",
]
for t in timings:
    out.append(f"| {t['phase']} | {t['exit']} | {t['seconds']:.3f} |")
out += [
    "",
    "校准约4.27分钟、评估约79.25分钟、服务器审计约0.60分钟。完整真实命令在evidence/v3/commands.jsonl；监控和轻量打包命令及退出码在COMMANDS.json。pipeline、calibrate、evaluate、audit、controller、supervisor均退出0；远端PROCESS_EXIT_ACCEPTANCE与本地LOCAL_ACCEPTANCE=PASS_FINAL。",
    "",
    "复算命令：",
    "```powershell",
    f'python execution/20261020_zrv3_object_seed41/write_final_report.py "{D.as_posix()}"',
    "```",
    "",
    "解释边界：seed41是在看到旧ZR-v2负结果后选定的单seed诊断；动作仍使用每阶段重置的global RNG，没有独立轨迹随机流；历史控制没有重新配对评估；新增残差校准有额外参数与计算预算；官方预训练state元数据unknown、父训练调度补全等原缺口保留。结果支持“本次未通过预定筛选”，不支持统计显著、跨seed普遍无效或投影的独立因果结论。",
    "",
    "没有下载或发布权重、latent、buffer载荷、原动作、RNG二进制或凭据。代码、报告与轻量记录按既有授权发布MLR；发布SHA以本地PUBLICATION_RECEIPT为准。",
]
(R / "FINAL_REPORT.md").write_text("\n".join(out) + "\n", encoding="utf-8")
(D / "FINAL_TASK_DECOMPOSITION.json").write_text(json.dumps({
    "status": "PASS", "auc_delta_fraction": "-17/960",
    "task_contributions": {t: {"fraction": str(d), "pp": float(d*100)} for t,d in task_delta.items()},
    "new_training_or_rollouts": False
}, indent=2) + "\n")
print("FINAL_REPORT and task decomposition written; read-only acceptance PASS_FINAL")
