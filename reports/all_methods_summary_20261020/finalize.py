"""Verify exported workbook and original source protection; build readable report."""
from pathlib import Path
import json,hashlib,zipfile,xml.etree.ElementTree as ET,datetime,csv
R=Path(__file__).resolve().parent;root=R.parents[1]
d=json.loads((R/"ALL_RESULTS.json").read_text(encoding="utf8"))
xlsx=root/"outputs/all_methods_20261020/Coreset全方法指标总表.xlsx"
ns={"x":"http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
with zipfile.ZipFile(xlsx) as z:
    assert not any("vba" in n.lower() or "externallink" in n.lower() for n in z.namelist())
    sheets=[n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
    assert len(sheets)==7
    for n in sheets:
        t=ET.fromstring(z.read(n))
        assert not t.findall(".//x:c[@t='e']",ns),n
    t=ET.fromstring(z.read("xl/worksheets/sheet2.xml"))
    cells={c.attrib["r"]:c for c in t.findall(".//x:sheetData/x:row/x:c",ns)}
    for i,r in enumerate(d["runs"],6):
        for col,k in zip("FGHIJ",["AUC","FWT","NBT","OldFinal","FinalAll"]):
            c=cells[f"{col}{i}"]
            assert c.find("x:f",ns) is not None
            assert abs(float(c.find("x:v",ns).text)-r["metrics"][k])<1e-8
for s in d["sources"]:
    assert hashlib.sha256(Path(s["path"]).read_bytes()).hexdigest()==s["sha256"],s["id"]
def f(v):return "—" if v is None else f"{v:.2f}"
lines=["# Coreset 路线：全方法、全指标汇总","",
f"数据快照：{d['created']}。共51个完整四阶段实验、510个有效阶段/任务单元、10,200条原协议正式记录；18个方法/套件/重跑口径汇总组。另列原型、离线诊断和取消分支。无新训练或环境执行。","",
"Excel提供筛选、公式和来源；JSON/CSV保留数值精度。Excel中单次实验的五指标由10个矩阵单元计算，方法均值/样本SD引用单次实验；导出后缓存值再次核对。","",
"## 主要指标总览","",
"单位：均值为百分数，SD为百分点；格式“均值 ± 样本SD”。n=1不报告SD。**各行seed组合不同，不按跨行均值直接排名。**","",
"|套件|方法|版本|seeds|n|AUC|FWT|NBT|OldFinal|FinalAll|",
"|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
for x in d["summaries"]:
    vals=[f(x[k])+(f" ± {x[k+'_SD']:.2f}" if x[k+"_SD"] is not None else "") for k in ["AUC","FWT","NBT","OldFinal","FinalAll"]]
    lines.append("|"+"|".join([x["suite"],x["method"],x["version"],x["seeds"],str(x["n"]),*vals])+"|")
lines+=["","## 完整逐实验表","",
"|ID|套件|方法|seed|版本|AUC|FWT|NBT|OldFinal|FinalAll|基线ID|",
"|---|---|---|---:|---|---:|---:|---:|---:|---:|---|"]
for x in d["runs"]:
    lines.append("|"+"|".join([x["id"],x["suite"],x["method"],str(x["seed"]),x["version"],*[f(x["metrics"][k]) for k in ["AUC","FWT","NBT","OldFinal","FinalAll"]],x["baseline_id"]])+"|")
lines+=["","## 指标与比较口径","",
"- AUC：每个任务从刚学完到最后阶段的成功率平均，再对4任务平均；不是10格简单平均。",
"- FWT：四个对角线成功率的平均，沿用现有项目/论文口径，不当作其他文献的零样本迁移定义。",
"- NBT：三个旧任务各自相对刚学完的后续平均下降，再跨任务平均；越低越好，可为负数。其他四项越高越好。",
"- OldFinal：最后阶段Task6–8平均；FinalAll：最后阶段Task6–9平均。",
"- 全部阶段/任务单元均为20次正式记录。表内百分数和pp均存为数值，不将缺失值写成零。",
"- 首次seed43和repeat1各自保留。ZR-v1的seed43主比较按用户后来要求使用repeat1，同时保留首次口径敏感性差值；这是事后报告口径改变。",
"- RA OBJECT seed43使用repeat1 Random作为门槛比较；ZR-v2(Coreset)43绑定repeat1父模型；RA+ZR-v2与本轮RA比较。",
"- 五seed汇总不把seed43重复运行算成第六个seed。GOAL总览包含后来新增的seed41，故显示41–44；原42–44可在逐次表筛选，未覆盖历史三seed报告。",
"- 冻结父模型后的残差校准只记录新增40epoch/10,550step；完整训练每arm400epoch，OBJECT105,500step，GOAL76,100step。耗时仅填已有核验值，部分为epoch累计，不能视为统一端到端成本。",
"- 三seed或五seed为探索性重复；硬件、global动作RNG未按轨迹隔离、额外残差参数/校准预算、预训练state元数据unknown、调度补全和IFA间接变化，限制纯模块因果解释。","",
"## 不应混入完整四阶段排名的结果","",
"- TCC、新旧任务更新协调：只有现有数据/checkpoint诊断，未启动正式训练或闭环；专项页保留原始指标字段。",
"- ZR门控原型：200次原型更新、0闭环；正式ZR-v1为另外的五seed完整实验。",
"- SAZR：Task6单任务原型；锚定3/10、验证父/候选均3/10，FS=0、SF=0，候选未启用；正式12/20为回退父策略结果，不是SAZR候选成功率。",
"- 夹爪修复Stage A：离线训练与留出验证，0环境执行；未补造阶段B/C或完整策略指标。",
"- 取消的9,000条追加复评只保留11条正式与20条A/A，不并入原协议；后续重新授权的原协议IFA关闭三seed已经独立完成并纳入。",
"- v3-A、未触发的GOAL残差条件分支、未通过门槛的SC扩展均为NOT_STARTED，不能写成0成功率。",
"- Coreset路线之前的E1/E1-U/E1-O/R1及GOAL旧state_v2/E2/E3不在这次范围，不能因名称相近混入同协议表。","",
"## 复算与产物","",
"- collect.py：读取25份原始轻量证据，Fraction精确复算51组矩阵与原已报告指标，记录前后SHA256。",
"- build.mjs：用Artifact Tool构建7页可筛选Excel；五指标及均值/样本SD有Excel公式。",
"- finalize.py：检查导出公式缓存值、无公式错误、无宏/外链、原始证据哈希不变。",
"- ALL_RESULTS.json / single_runs.csv / comparisons.csv / special_metrics.csv：可机器读取的结果。",
"- sources/：原始轻量证据副本，来源ID与哈希见Excel末页。",
"- 本次只读汇总的完成不代表任何未启动分支完成。",""]
(R/"REPORT.md").write_text("\n".join(lines),encoding="utf8")
with (R/"method_summary.csv").open("w",encoding="utf-8-sig",newline="") as fp:
    rows=[{k:v for k,v in s.items() if k!="run_ids"} for s in d["summaries"]]
    w=csv.DictWriter(fp,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(R/"deliverables").mkdir(exist_ok=True)
(R/"deliverables"/xlsx.name).write_bytes(xlsx.read_bytes())
check=dict(status="COMPLETED_READONLY_CONSOLIDATION",time=datetime.datetime.now().astimezone().isoformat(),
    runs=51,summary_groups=18,full_run_records=10200,matrix_cells=510,special_metrics=len(d["special"]),
    workbook_formula_values_checked=255,workbook_sheets=7,source_files=25,source_hashes_unchanged=True,
    workbook_sha256=hashlib.sha256(xlsx.read_bytes()).hexdigest(),no_new_training_or_rollouts=True)
(R/"RUN_STATUS.json").write_text(json.dumps(check,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
print(json.dumps(check,indent=2))
