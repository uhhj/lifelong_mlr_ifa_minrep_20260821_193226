"""Read-only recomputation of the original ZR-v1 five-seed results."""
from pathlib import Path
from fractions import Fraction as F
import json, hashlib, statistics
R=Path(__file__).resolve().parent
p=R/"SOURCE_SUMMARY.json";s=json.loads(p.read_text())
def metrics(table):
    r={tuple(map(int,k.split(","))):F(str(v))/100 for k,v in table.items()}
    return dict(AUC=sum(sum(r[k,t] for k in range(t,10))/(10-t) for t in range(6,10))/4,
                FWT=sum(r[t,t] for t in range(6,10))/4,
                OldFinal=sum(r[9,t] for t in range(6,9))/3,
                FinalAll=sum(r[9,t] for t in range(6,10))/4)
rows=[]
for row in s["primary_rows"]:
    seed=row["seed"];t=s["tables"][str(seed)]
    base=metrics(t["repeat1_baseline" if seed==43 else "first_baseline"])
    new=metrics(t["new"])
    for k in base:
        assert base[k]==F(row["baseline"][k])
        assert new[k]==F(row["new"][k])
    rows.append(dict(seed=seed,baseline_AUC=float(base["AUC"]*100),
        ZR_AUC=float(new["AUC"]*100),delta_pp=float((new["AUC"]-base["AUC"])*100)))
assert sum(r["delta_pp"]>0 for r in rows)==2
mean=statistics.mean(r["delta_pp"] for r in rows)
assert abs(mean+3.375)<1e-10
out=dict(status="PASS",method="ZR-v1 Coreset+IFA+instruction-zero-response residual",
    rows=rows,positive_seeds=[r["seed"] for r in rows if r["delta_pp"]>0],
    mean_delta_pp=mean,paired_sample_sd=statistics.stdev(r["delta_pp"] for r in rows),
    source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),no_new_training_or_rollouts=True)
(R/"RECOMPUTED.json").write_text(json.dumps(out,indent=2)+"\n")
tab="\n".join(f"|{r['seed']}|{r['baseline_AUC']:.4f}|{r['ZR_AUC']:.4f}|{r['delta_pp']:+.4f}|" for r in rows)
report=f"""# 最初ZR-v1是否产生过正向效果

有真实观测到的正向结果，尤其seed42；但五seed整体没有稳定收益。
此处“最初”指2026-10-17的Coreset＋IFA＋指令零响应残差：
共享时间网络/策略头继续训练，当前残差训练，旧残差冻结。
它不同于后续冻结父网络、追加校准的ZR-v2/v3/SAZR。

|Seed|Coreset＋IFA AUC%|ZR-v1 AUC%|差值pp|
|---|---:|---:|---:|
{tab}

seed42：AUC+11.0417pp，FWT+6.25pp，OldFinal+11.6667pp，FinalAll+16.25pp。
Task6阶段8由20%至45%、阶段9由30%至55%；Task9由45%至75%。
该seed满足当时全部预定推进门槛，因此扩展有实际正向数据依据。
seed44也出现AUC+2.3958pp，但FWT下降1.25pp，未满足全部原门槛。

五seed仅2/5 AUC上升，3/5下降。主比较均值76.8542%→73.4792%，
配对差均值{mean:.4f}pp、样本SD{out['paired_sample_sd']:.4f}pp。
OldFinal均值仅+0.6667pp，FinalAll均值-2pp。

seed43按用户指定采用第二次repeat1基线，为看到结果后的报告口径调整。
首次43基线也保留：该seed差-20.9375pp，全五seed平均差-6.5208pp，
无论采用哪一口径，都不支持跨seed稳定增益。

观察到的提升不能直接证明“零响应约束”本身的因果收益：
共享网络/策略头仍被训练，额外参数、优化轨迹与模块作用混合，
评估global动作RNG未逐轨迹隔离，部分基线存在硬件差异；
尚无等参数普通残差对照。此前SAZR饱和/投影诊断针对冻结父策略的校准结构，
不能直接套用来否定ZR-v1曾出现的正向结果。

本次从已有成功率矩阵按精确分数复算AUC/FWT/OldFinal/FinalAll，
逐项与原聚合值核对，无新训练、GPU调用或闭环。原最终验收另存
SOURCE_FINAL_ACCEPTANCE.json；SOURCE_SUMMARY中的pending标签是历史聚合时点，
不替代之后完成的验收。不覆盖历史报告。
"""
(R/"REPORT.md").write_text(report,encoding="utf-8")
(R/"RUN_STATUS.json").write_text(json.dumps(dict(status="COMPLETED",kind="read_only_retrospective"))+"\n")
print(json.dumps(out,indent=2))
