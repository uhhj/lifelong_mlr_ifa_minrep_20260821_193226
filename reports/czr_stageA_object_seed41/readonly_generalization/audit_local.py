"""Recompute counts/frontiers from downloaded metadata; do not load features."""
from pathlib import Path
import json,hashlib,csv,datetime,statistics
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text(encoding="utf-8"))
def write(n,v):(R/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
receipt=read("DOWNLOAD_ACCEPTANCE.json")
for n,h in receipt["files"].items():assert hashlib.sha256((R/n).read_bytes()).hexdigest()==h
a=read("DIAGNOSTIC.json");q=read("QUERY_METADATA.json");centers=read("CENTER_NEIGHBORS.json")
assert a["original_files_unchanged"] and a["original_radius_unchanged"]
assert a["new_model_forwards"]==a["optimizer_calls"]==a["environment_calls"]==0
assert hashlib.sha256((R/"diagnose_v2.py").read_bytes()).hexdigest()==a["script_sha256"]
original=json.loads((R.parent/"NEIGHBOR_AUDIT.json").read_text())
for kind,oldkey in (("holdout","heldout_rows"),("cross_demo","cross_demo_rows")):
    old={r["sample_id"]:r for r in original[oldkey]}
    for r in q["centered_unit__all_correct"][kind]:
        assert abs(r["ratio"]-old[r["sample_id"]]["min_distance_to_radius_ratio"])<1e-9
table=[];maxnet={}
for mode,parts in q.items():
    maxnet[mode]={}
    for split,rows in parts.items():
        assert len(rows)==(303 if split=="holdout" else 1255)
        er=[r for r in rows if not r["parent_correct"]];co=[r for r in rows if r["parent_correct"]]
        m=a["analyses"][mode][split]
        assert len(er)==m["parent_errors"] and len(co)==m["correct"]
        for c in m["fixed_counterfactual_curve"]:
            fs=sum(r["ratio"] is not None and r["ratio"]<c["radius_multiplier"] for r in er)
            sf=sum(r["ratio"] is not None and r["ratio"]<c["radius_multiplier"] for r in co)
            assert (fs,sf,fs-sf)==(c["FS"],c["SF"],c["net"])
            table.append(dict(mode=mode,split=split,**c))
        firsterror=min(r["ratio"] for r in er if r["ratio"] is not None)
        firstharm=min(r["ratio"] for r in co if r["ratio"] is not None)
        assert firsterror==m["first_error_threshold"] and firstharm==m["first_harm_threshold"]
        assert sum(r["ratio"]<firstharm for r in er if r["ratio"] is not None)==m["maximum_repairs_before_any_harm"]
        auc=sum((e["ratio"]<c["ratio"])+.5*(e["ratio"]==c["ratio"]) for e in er for c in co)/(len(er)*len(co))
        assert abs(auc-m["error_ranking_auc"])<1e-12
        scores={}
        for r in rows:
            if r["ratio"] is not None:scores[r["ratio"]]=scores.get(r["ratio"],0)+(1 if not r["parent_correct"] else -1)
        running=0;peak=0
        for score in sorted(scores):
            running+=scores[score];peak=max(peak,running)
        maxnet[mode][split]=peak
with (R/"RADIUS_COUNTERFACTUALS.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
counts=dict(error_centers=len(centers),nearest_correct_same_demo=sum(r["nearest_correct_same_demo"] for r in centers),
            nearest_correct_same_parent=sum(r["nearest_correct_parent_class_matches"] for r in centers),
            time_gaps=[r["nearest_correct_time_gap"] for r in centers],
            median_matched_radius_increase=statistics.median(r["matched_protection_radius_ratio"] for r in centers),
            max_matched_radius_increase=max(r["matched_protection_radius_ratio"] for r in centers))
assert len(centers)==23 and counts["nearest_correct_same_demo"]==23
status=dict(status="READONLY_AUDIT_COMPLETED",local_recompute="PASS",counts=counts,
    maximum_retrospective_net_over_all_uniform_radius_thresholds=maxnet,
    inputs_unchanged=True,original_exit=1,repaired_exit=0,
    model_forwards=0,training_steps=0,environment_calls=0,
    time=datetime.datetime.now().astimezone().isoformat())
write("LOCAL_ACCEPTANCE.json",status);write("RUN_STATUS.json",status)
base=a["analyses"]["centered_unit__all_correct"]
lines=["# CZR跨演示泛化失败：只读证据审计","",
       "结论：严格局部球的确很保守，但现有数据不支持“只把半径放大即可修复”。"
       "直接原因是当前距离下同演示相邻状态很近，而跨演示错误没有形成与正确状态分离的局部簇。"
       "这说明当前特征距离与高精度动作纠错机制不匹配，不证明隐藏表示完全没有纠错信息。","",
       "## 1. 构造集修复主要来自自匹配","",
       "- 23个训练错误中，删除查询自身作为中心但保留其他中心、均值和半径后，只有2个仍落入另一错误中心的球内。"
       "这是中心自排除检查，不是重新拟合的独立验证。",
       "- 原构造集23/23修复因此主要反映精确记忆，不应解释成学到跨演示规律。",
       "- 8次逐示范留出均无触发、无修复；两条留出演示也无触发。","",
       "## 2. 保护半径由同演示的邻近正确时刻决定","",
       f"- 23/23错误中心的最近正确样本都在同一演示；真实监督时刻仅相隔1–2步。",
       "- 1255个训练窗口的自排除最近邻中，1251个来自同一演示（99.68%），时间差中位数1步。",
       "这与窗口重叠、相邻时刻输入相似相符，但没有隔离视觉背景、操作进度和动作转换各自贡献，"
       "不能直接声称表示只编码了示范身份。",
       "- 半径中位数约0.0703，而留出错误到匹配错误中心的最短距离约0.787–0.909。"
       "不同演示的错误在这个几何尺度上并不相邻。",
       "",
       "## 3. 放宽半径：先伤害，再纠正","",
       "下表是已保存特征上的事后反事实，不改变原0.45系数，不执行任何动作或选择部署阈值。",
       "",
       "| 原半径倍率 | 跨演示FS | 跨演示SF | 留出FS | 留出SF |",
       "|---:|---:|---:|---:|---:|"]
for x,y in zip(base["cross_demo"]["fixed_counterfactual_curve"],base["holdout"]["fixed_counterfactual_curve"]):
    lines.append(f"| {x['radius_multiplier']} | {x['FS']} | {x['SF']} | {y['FS']} | {y['SF']} |")
lines+=["",
    f"- 跨演示：首个伤害阈值{base['cross_demo']['first_harm_threshold']:.3f}倍，首个修复{base['cross_demo']['first_error_threshold']:.3f}倍；首个修复前已触及7个正确窗口。",
    f"- 留出：首个伤害{base['holdout']['first_harm_threshold']:.3f}倍，首个修复{base['holdout']['first_error_threshold']:.3f}倍；首个修复前已触及6个正确窗口。",
    "- 对原统一半径缩放，零伤害条件下可修复数两种划分均为0；这不是只检查几个离散倍率得到的结论，首个阈值次序已能排除。",
    f"- 对全部已观察分数断点的事后计数，原机制最大净修复为：{maxnet['centered_unit__all_correct']}。"
    "这是描述性上限，不是经过独立确认的超参数选择。",
    "",
    "## 4. 原保护集存在额外保守性，但放松它仍不解决问题","",
    "- 23个中心中有12个的最近正确邻居，其父夹爪类别与该中心不同。"
    "这类查询原本会被父类别门拒绝，却仍参与缩小原球半径，因此原定义有额外保守性。",
    f"- 改用相同父类别正确样本计算半径的离线对照，中位半径增大{counts['median_matched_radius_increase']:.3f}倍，最大{counts['max_matched_radius_increase']:.3f}倍。",
    "- 但固定0.45系数下，跨演示和留出仍无修复。留出首次修复前仍触及7个正确窗口；放大至该对照半径6倍时，修复4个、损害102个。",
    "因此“保护了不必要的正确案例”是次要限制，不能解释全部失败。",
    "",
    "## 5. 有弱排序信息，不等于有可靠的动作替换边界","",
    f"- 原距离/半径分数识别标签错误的描述性ROC-AUC：跨演示{base['cross_demo']['error_ranking_auc']:.4f}，留出{base['holdout']['error_ranking_auc']:.4f}。此AUC不是原任务成功率AUC。",
    "- 这说明错误相对很多正确样本确实更接近错误中心；不能说特征毫无信息。",
    "- 但留出只有4个错误、299个正确样本。8倍半径检出4个错误时同时误触27个正确样本，干预精确率仅4/31=12.90%。"
    "错误稀少时，较高ROC-AUC仍可能无法提供高精度纠错。",
    "- 四个留出错误的最近同父类别正确样本均比最近错误中心更近。简单最近专家标签检索也未修复这4个错误，反而损害11个正确窗口。",
    "- 跨演示最近专家标签检索修复6个、损害64个；不能把这项事后检索对照当作训练通过的新模型。",
    "",
    "## 6. 不是单纯中心化/归一化错误","",
    "| 距离/保护 | 留出首次伤害倍率 | 首次修复倍率 | 8倍半径FS/SF |",
    "|---|---:|---:|---:|"]
for mode,d in a["analyses"].items():
    m=d["holdout"];eight=next(x for x in m["fixed_counterfactual_curve"] if x["radius_multiplier"]==8)
    lines.append(f"| {mode} | {m['first_harm_threshold']:.3f} | {m['first_error_threshold']:.3f} | {eight['FS']}/{eight['SF']} |")
lines+=["",
    "未中心化单位向量和原始欧氏距离都保留了同样的失败模式。"
    "这只排除了两个简单距离替换作为明显解释，不排除更合适的时序条件、纠错监督或表示中存在未被当前距离利用的信息。",
    "",
    "## 判断与边界","",
    "当前证据支持：构造集自匹配＋演示内近邻主导＋罕见错误的极高精度要求，"
    "共同使各向同性局部球缺乏可迁移的纠错范围。0.45是直接拒绝原因，但单纯放大或删除额外保护并不能满足原零伤害门槛。",
    "不建议根据这些留出结果调大半径；这会把诊断集变成调参集。"
    "也不能据此断言整个父表示无用，或证明这些标签差异会造成物理失败。",
    "四个留出错误全部来自demo8，demo9没有父标签错误；窗口彼此相关。"
    "父模型以前见过全部10条示范，这里只有纠错记忆层面的留出。",
    "未来若继续，优先需要能对应同类纠错需求的跨演示监督或可信同状态纠正，"
    "而不是增加残差参数量或继续扩大无条件邻域。本次不启动任何后续实验。",
    "",
    "## 执行、恢复与只读保护","",
    "- 仅CPU加载既有native_features.pt；新模型前向0、optimizer调用0、环境调用0。",
    "- 原CZR源码、记忆、特征、父模型与输入文件前后哈希一致；下载JSON哈希与远端一致，本地独立计数及原分数一致性通过。",
    "- 首次统计脚本exit1：NumPy int64 JSON序列化错误。原脚本/日志/退出码保留；v2仅增加NumPy标量序列化并使用新输出目录，exit0。",
    f"- 修复后只读计算耗时{a['seconds']:.2f}秒。",
    "- 命令：`/data/mlr_ifa/env/bin/python /data/mlr_ifa/analyses/CZR_A_OBJECT_seed41_20261020_generalization_v2.py`。",
    "- 本地复算：`python execution/20261020_czr_stageA/readonly_generalization/audit_local.py`，退出0。",
    "- 远端结果目录：`/data/mlr_ifa/analyses/CZR_A_OBJECT_seed41_20261020_generalization_v2`。",
    "- 发布仅代码、报告、哈希、距离及正确性元数据；不包含特征、动作标签、权重或纠错记忆载荷。"]
(R/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps(status,ensure_ascii=False,indent=2))
