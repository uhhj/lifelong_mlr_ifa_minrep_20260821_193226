"""Independent package/history checks and reproducible report. No model operations."""
from pathlib import Path
import json,hashlib,datetime,statistics
from collections import defaultdict
from analyze import enrich_rows,analyze_root
R=Path(__file__).resolve().parent;E=R/"evidence";ROOT=R.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
def fmt(x):return "—" if x is None else "%.4f"%x
pkg=json.loads((E/"FINAL_PROCESS_AND_PACKAGE.json").read_text())
assert pkg["status"]=="PASS" and not pkg["workers"]
for name,m in pkg["files"].items():
    p=E/name;assert p.stat().st_size==m["bytes"] and sha(p)==m["sha256"],name
for name,h in pkg["scripts"].items():assert sha(R/name)==h,name
assert sha(R/"PROTOCOL.md")==pkg["protocol_sha256"]
result=analyze_root(E)
history={};history_inputs={};history_rows=[]
roots={
 "GOAL42":ROOT/"execution/20261018_ra_coreset_goal_seed42/final_evidence/ra_coreset/evaluation",
 "OBJECT41":ROOT/"execution/20261019_ra_object_conditional/completion_20261020/download/evidence/OBJECT_RA_seed41/ra_coreset/evaluation",
}
for name,folder in roots.items():
    hist={}
    for task in (6,7,8):
        rates=[]
        for stage in range(task,10):
            p=folder/("trials_stage%d_task%d.jsonl"%(stage,task));history_inputs[str(p.relative_to(ROOT))]=sha(p)
            rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            assert len(rows)==20 and all(r["task"]==task and r["stage"]==stage for r in rows)
            successes=sum(bool(r["success"]) for r in rows);rates.append(successes)
            history_rows.append(dict(suite=name,task=task,stage=stage,successes=successes,total=20))
        hist[str(task)]=rates
    history[name]=hist
write(E/"HISTORICAL_OUTCOMES.json",dict(counts=history,rows=history_rows,input_sha256=history_inputs))
detail={}
for s in result["suites"]:
    name=s["input_subdir"]
    rows=enrich_rows([json.loads(x) for x in (E/name/"pairs.jsonl").read_text().splitlines()])
    eligible=[r for r in rows if r["view"]=="native" and r["selected"] and r["split"]=="validation" and r["cross_eligible"]]
    cache={ (r["task"],r["stage"],r["sample_id"]):r for r in rows if r["view"]=="cache"}
    shared=[r for r in eligible if r["damage"] and cache[(r["task"],r["stage"],r["sample_id"])]["damage"]]
    # Reduced diagnostics contain identities and scalar errors, not actions/latents.
    stats={}
    for view in ("cache","native"):
        v=eligible if view=="native" else [cache[(r["task"],r["stage"],r["sample_id"])] for r in eligible]
        values=sorted(r["motion_delta"] for r in v)
        targets=defaultdict(list)
        for r in v:targets[(r["task"],r["demo_id"],r["target"])].append(r["motion_delta"])
        biggest=max(targets,key=lambda k:statistics.mean(targets[k])) if targets else None
        remaining=[r["motion_delta"] for r in v if (r["task"],r["demo_id"],r["target"])!=biggest]
        stats[view]=dict(median_delta=statistics.median(values),max_delta=max(values),
            mean_delta=statistics.mean(values),
            largest_mean_target=list(biggest) if biggest else None,
            mean_delta_excluding_largest_target=statistics.mean(remaining) if remaining else None)
    detail[name]=dict(eligible_comparisons=len(eligible),eligible_unique_targets=len({(r["task"],r["demo_id"],r["target"]) for r in eligible}),
        shared_damage=[{k:r[k] for k in ("task","stage","sample_id","demo_id","target","teacher_error","student_error")} for r in shared],
        shared_damage_unique_targets=len({(r["task"],r["demo_id"],r["target"]) for r in shared}),outlier_checks=stats)
    detail[name]["all_cache_by_membership"]={}
    for member in (False,True):
        v=[r for r in rows if r["view"]=="cache" and r["selected"]==member]
        detail[name]["all_cache_by_membership"][str(member)]=dict(rows=len(v),
            teacher_error=statistics.mean(r["teacher_error"] for r in v),
            student_error=statistics.mean(r["student_error"] for r in v),
            harm=sum(r["damage"] for r in v),benefit=sum(r["improvement"] for r in v))
write(E/"SUPPLEMENTARY_CHECKS.json",detail)
audits={n:json.loads((E/n/"FINAL_AUDIT.json").read_text()) for n in roots}
assert all(a["status"]=="PASS" and len(a["checkpoints"])==4 for a in audits.values())
assert all(a[k]==0 for a in audits.values() for k in ("optimizer_calls","optimizer_constructs","environment_constructs","environment_steps","policy_action_samples"))
assert all(sha(ROOT/p)==h for p,h in history_inputs.items())
status=dict(status="STAGE_A_COMPLETED",stage_B="NOT_STARTED",recommend_B={s["input_subdir"]:s["gate"]["recommend_B"] for s in result["suites"]},
    time=datetime.datetime.now().astimezone().isoformat(),engineering="PASS",new_training_steps=0,new_closed_loop_episodes=0)
write(R/"RUN_STATUS.json",status)
write(R/"LOCAL_ACCEPTANCE.json",dict(status="PASS",package_files=len(pkg["files"]),input_transmission_hashes_unchanged=True,
      six_transition_cells_each_suite=True,teacher_fixed_across_stages=True,paired_native_cache_identity=True,
      original_history_hashes_unchanged=True,software_tests=18,software_test_exit=0,analysis_exit=0,
      model_checkpoints=8,optimizer_calls=0,environment_calls=0,stage_B_started=False))
text=[
"# EPR 阶段A：零更新诊断结论",
"",
"**结论：两套数据都存在原生演示输入上的行为退化，但当前 EPR 方案均不建议进入阶段B。**",
"阻碍不是“没有遗忘”，而是缓存输入上要保护的变化与原生输入上的退化位置不一致。固定准入门槛均未通过；本轮停止，不训练、不追加闭环。",
"",
"## 执行与验收",
"",
"- GOAL seed42、OBJECT seed41 各四个已完成RA最终checkpoint，全部严格加载100个下游张量；每套前端跨stage一致。",
"- Task6/7/8全部原缓存，共7,392窗口；每任务160条原生演示窗口，共960条，按demo×时间×是否入库预选，不按错误或闭环成败挑样本。",
"- 六种旧任务转移每套完整；共16,799条窗口—后续阶段—输入视图比较。这不是16,799次环境执行，也不是独立样本。",
"- 原生复核包括同历史完整8×7动作绑定、padding、语言和state。仅取实际argmax Gaussian参数，没有动作采样。",
"- 优化器构造/更新、环境构造/step、追加闭环全部为0；源文件/输入/权重前后hash及模型推理前后张量不变，进程均已退出0。",
"- 18项合成汇总程序测试PASS，仅验证统计软件。服务器推理及完整hash验收时间如下，非训练开销测量。",
"",
"|套件|缓存窗口|原生窗口|比较行|含hash总秒|原生编码秒|时间解码/头秒|",
"|---|---:|---:|---:|---:|---:|---:|",
]
for n,a in audits.items():
    text.append("|%s|%d|%d|%d|%.2f|%.2f|%.2f|"%(n,sum(a["cache_windows"].values()),sum(a["native_windows"].values()),a["pair_rows"],a["total_seconds"],a["native_seconds"],a["decode_seconds"]))
text+=["","## 关键结果：相同合格样本的两条输入路径","",
"以下只取预留诊断演示5–9、真实RA入库窗口，教师在两种输入上均通过专家检查，且跨输入差异合格。",
"误差是前六维按原任务固定robust scale标准化后的RMS；不是成功率。重大退化/改善阈值分别为±0.1，夹爪SF/FS另计。",
"“比较数”可含同一真实时刻在不同后续stage的重复；不同真实监督时刻另列，它们仍不等于统计独立样本。",
"",
"|套件|合格比较/真实时刻|缓存退化/改善|原生退化/改善|原生退化真实时刻|退化集合Jaccard|结论|",
"|---|---:|---:|---:|---:|---:|---|"]
for s in result["suites"]:
    gate=s["gate"];n=s["input_subdir"]
    g={v:next(x for x in s["groups"] if x["level"]=="suite" and x["view"]==v and x["selected"] and x["split"]=="validation" and x["population"]=="cross_eligible") for v in ("cache","native")}
    text.append("|%s|%d / %d|%d / %d|%d / %d|%d|%.2f%%|%s|"%(n,g["native"]["rows"],g["native"]["unique_targets"],g["cache"]["damage_rows"],g["cache"]["improvement_rows"],g["native"]["damage_rows"],g["native"]["improvement_rows"],gate["native_damage_unique_targets"],100*gate["damage_cache_native_jaccard"],gate["decision"]))
text+=["",
"两套原生退化均涉及5个不同演示编号与全部6个任务阶段单元，不能解释成只在一个演示里自证；但GOAL两视图的退化集合交集为0，OBJECT交集仅3条阶段比较，来自**两个真实监督时刻**。这不支持把缓存输出保留直接当作原生行为保护。",
"",
"|套件|视图|教师→学生平均误差|变化中位数|最大变化|去掉最大均值目标后的平均变化|",
"|---|---|---:|---:|---:|---:|"]
for s in result["suites"]:
    n=s["input_subdir"]
    for v in ("cache","native"):
        g=next(x for x in s["groups"] if x["level"]=="suite" and x["view"]==v and x["selected"] and x["split"]=="validation" and x["population"]=="cross_eligible")
        z=detail[n]["outlier_checks"][v]
        text.append("|%s|%s|%.4f→%.4f|%.4f|%.4f|%s|"%(n,v,g["mean_teacher_error"],g["mean_student_error"],z["median_delta"],z["max_delta"],fmt(z["mean_delta_excluding_largest_target"])))
text+=["",
"OBJECT最大目标为Task6/demo6/85：stage7、8缓存误差分别约14.43、13.95，stage9恢复；另一个共有退化目标为Task6/demo9/129。这里14是固定动作尺度标准化后的RMS，绝非14单位实际动作。前一目标stage7/8执行赢家不是最接近旧Gaussian的分量，stage9又恢复；这只是路由变化线索，不能证明路由是唯一原因。",
"",
"平均值需结合中位数与逐目标结果阅读，不能把OBJECT少数大变化外推为普遍大幅恶化。EPR教师资格本身也是有限的：每套验证原生入库样本120个真实时刻，双视图合格分别28和38个。",
"",
"## 六种转移都报告",
"",
"|套件|任务→阶段|合格原生比较|缓存退化/改善|原生退化/改善|原生平均误差变化|",
"|---|---|---:|---:|---:|---:|"]
for s in result["suites"]:
    for t in (6,7,8):
        for stage in range(t+1,10):
            g={v:next(x for x in s["groups"] if x["level"]=="task_stage" and x["task"]==t and x["stage"]==stage and x["view"]==v and x["selected"] and x["split"]=="validation" and x["population"]=="cross_eligible") for v in ("cache","native")}
            text.append("|%s|%d→%d|%d|%d / %d|%d / %d|%s|"%(s["input_subdir"],t,stage,g["native"]["rows"],g["cache"]["damage_rows"],g["cache"]["improvement_rows"],g["native"]["damage_rows"],g["native"]["improvement_rows"],fmt(g["native"]["mean_motion_delta"])))
text+=["",
"## 为什么现在不应进入B",
"",
"1. **原生问题存在，但缓存保护信号不能可靠定位它。** GOAL合格样本缓存误差总体改善，原生输入却退化。OBJECT的共有退化仅两个真实时刻。当前方案只在缓存上施加输出保持；阶段A没有证明它能影响这些原生退化点。",
"2. **按发生频次，夹爪符号或分量选错并不普遍。** 合格原生验证集合两套夹爪SF均为0，最近教师分量仍是执行赢家的比例分别98.41%和93.85%。多数可观测变化在连续动作均值；但OBJECT少数路由变化对应大误差，不能因其数量少就断言不重要。",
"3. **存在需要保留的有益迁移。** 全部验证入库缓存比较，GOAL重大改善218次、退化33次；OBJECT改善608次、退化207次。它们包含教师本来不够准确的样本，不应被无条件蒸馏锁死。资格筛选减少这一风险，但并未解决原生迁移断层。",
"4. **泛化证据有边界。** 诊断演示5–9未用于本次规则拟合，但历史RA曾训练过全部10条演示；这里只支持跨演示一致性检查，不是未见演示/未见物理状态泛化。原生样本是预先固定的分层样本，不代表所有原生状态。",
"5. **不能单因果归因于增广。** 原缓存来自正式导出的既有训练模式/增广，而本次原生路径为eval；两条路径同时包含视图与模块模式差别。未额外做分解实验，因此不能声称已确定增广是唯一原因。",
"",
"## 已入库与未入库：保护位置是否匹配",
"",
"这里使用全部缓存、全部演示，包含教师不合格与未知身份窗口。计数为相关的阶段—窗口比较，不是独立统计试验。",
"",
"|套件|缓存成员|比较数|教师→学生平均误差|重大退化|重大改善|",
"|---|---|---:|---:|---:|---:|"]
for n,z in detail.items():
    for member in (True,False):
        q=z["all_cache_by_membership"][str(member)]
        text.append("|%s|%s|%d|%.4f→%.4f|%d|%d|"%(n,"入库" if member else "未入库",q["rows"],q["teacher_error"],q["student_error"],q["harm"],q["benefit"]))
text+=["",
"两套数据都显示：入库缓存的总体误差下降，未入库缓存总体上升。EPR不访问未入库旧样本且仅约束已存缓存，正好最缺乏对退化区域的直接约束。不能据此偷偷加入未入库数据或改变原训练输入。",
"",
"## 与已有闭环证据的关系",
"",
"|套件|Task6 成功数|Task7 成功数|Task8 成功数|",
"|---|---|---|---|"]
for n,h in history.items():
    text.append("|%s|%s|%s|%s|"%(n,"→".join(map(str,h["6"])),"→".join(map(str,h["7"])),"→".join(map(str,h["8"]))))
text+=["",
"每个数字的分母为20，顺序为刚学完到stage9。仅复算原记录，未新增评估。Task6并非单调遗忘；动作空间误差和这些闭环涨跌不能一一因果对应，原全局轨迹动作RNG也未隔离。",
"",
"## 决策与剩余未知",
"",
"**不进入当前EPR阶段B，不启动任何更新。** 这是对“可信缓存输出保护能转化为原生保护”这条机制的否定准入，不是证明所有功能回放都无效。",
"若后续另行重设计，首先需要解释缓存约束如何覆盖原生退化方向，而不是直接调大蒸馏权重或放松本次门槛。本轮不提出自动下一组实验。",
"",
"阶段A未测试更新后的保旧/保新效果、辅助梯度作用、实际训练额外开销、闭环增益或统计显著性；不能把这些未知写成通过。",
"",
"## 真实命令与产物",
"",
"服务器命令逐字保存在 evidence/execution/*_COMMAND.json；两套 run_diagnostic、有限run_all和final_snapshot均退出0。",
"",
"```text",
"python execution/20261020_epr_stageA/test_metrics.py",
"python execution/20261020_epr_stageA/analyze.py execution/20261020_epr_stageA/evidence",
"python execution/20261020_epr_stageA/finalize_local.py",
"```",
"",
"上述本地测试、复算与验收退出0。最初SSH密钥认证失败后，经用户提供凭据仅内存认证恢复；凭据未写文件。一次本地命令JSON转义错误发生于任何远端诊断启动前，后改用有限Python runner；没有重复推理或失败环境尝试。",
"",
"- `evidence/analysis/ANALYSIS.json`：完整判定、所有分层和输入hash。",
"- `evidence/analysis/GROUP_SUMMARIES.csv`：全部demo、task-stage、入库/未入库、两视图表。",
"- `evidence/{GOAL42,OBJECT41}/pairs.jsonl`：仅身份与标量误差，不含原动作或特征。",
"- `evidence/FINAL_PROCESS_AND_PACKAGE.json`：输出传输hash、真实进程退出证据。",
"- `evidence/SUPPLEMENTARY_CHECKS.json`：共同退化身份及极端目标敏感性。",
"",
"按长期授权推送代码、报告与轻量证据；不上传checkpoint、数据、latent、buffer、原动作或凭据。",
]
(R/"REPORT.md").write_text("\n".join(text)+"\n",encoding="utf8")
print(json.dumps(status,ensure_ascii=False))
