"""Recompute causal-boundary report from existing episode and inference metadata."""
from pathlib import Path
import csv
import hashlib
import json

R=Path(__file__).resolve().parent
E=R.parent/"progress_20261020_160301/evidence"
js=lambda p:json.loads(p.read_text(encoding="utf-8"))
jl=lambda p:[json.loads(x) for x in p.read_text(encoding="utf-8").splitlines()]
d=js(R/"gripper_followup/OFFLINE_DIAGNOSTIC.json")
c=d["corpus"]
assert d["optimizer_calls"]==d["environment_calls"]==0
assert d["artifact_hashes_before"]==d["artifact_hashes_after"]
assert d["source_contract_unchanged"]
a={x["initial_index"]:x for x in jl(E/"validation_parent/episodes.jsonl")}
b={x["initial_index"]:x for x in jl(E/"validation_candidate/episodes.jsonl")}
pairs=[]
for i in sorted(a):
    x,y=a[i],b[i]
    for k in ("initial_sha256","noise_seed","noise_sha256","env_seed","parent_sha256"):
        assert x[k]==y[k]
    assert y["enabled"] and not x["enabled"]
    pairs.append(dict(initial=i,parent_success=x["success"],candidate_success=y["success"],
        parent_step=x["first_success_policy_step"],candidate_step=y["first_success_policy_step"],
        outcome=("S" if x["success"] else "F")+("S" if y["success"] else "F")))
with (R/"VALIDATION_PAIRS.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
assert c["metrics"]["route"]["min"]==1
grip=c["by_dimension"]["6"]
assert grip["zero_tanh_slope_count"]==1562 and grip["max_abs_shift"]==0
assert grip["sign_mismatch_count"]==10
traces=d["successful_parent_features"]
n_trace=sum(x["steps"] for x in traces.values())
n_nonzero=sum(x["nonzero_mean_rows"] for x in traces.values())
assert n_trace==448 and n_nonzero==416
mean_shift=c["metrics"]["mean_change"]
rows="\n".join(f"|{x['initial']}|{'成功' if x['parent_success'] else '失败'}|"
              f"{'成功' if x['candidate_success'] else '失败'}|{x['parent_step']}|{x['candidate_step']}|"
              for x in pairs)
report=f"""# SAZR未产生验证收益：已有资产只读诊断

结论：工程机制通过，但候选主要做了有限的连续动作微调，未修复验证中七个失败初态。
最明确的新发现是夹爪输出路径数值饱和，训练缓存上完全没有可学习的夹爪修正；
其次，成功锚点空间覆盖了绝大部分训练特征能量。二者是实际观测到的限制，
但缺少失败轨迹逐步状态与动作记录，不能把七次失败全部归因于其中任一项。

## 1. 开关没有掩盖已经观察到的改善

验证初态30–39、环境seed、逐初态/step噪声SHA和父权重逐项匹配。
SS=3、SF=0、FS=0、FF=7。候选验证enabled=true，正式才因门槛拒绝关闭。
即使把FS>=2降为FS>=1也不会启用。正式12/20是回退父策略，不能用来证明候选有效或无效。

|初态|父结果|候选结果|父首次成功step|候选首次成功step|
|---|---|---|---:|---:|
{rows}

三个成功初态的首次成功step变化为-9、-1、+4，平均-2。
这表明执行时间有变化，但不是额外成功证据，也不是每条动作都等价。
七个失败均运行到600步；已保存记录不足以区分未抓到、掉落、放置偏差等故障阶段。

## 2. 30维保护空间覆盖约96.53%的训练特征平方范数

只读加载既有父模型、Q与最终UV，对全部1562个原训练窗口做前向复算。
投影后保留的总平方范数比例为{100*c['retained_feature_energy']:.4f}%；
每窗口范数比中位数{c['metrics']['projection_norm_ratio']['quantiles']['0.5']:.4f}。
所以“512维只保护30维，还剩482维”不能解释为保留了94%的实际输入信号。
这些特征在成功锚点张成的方向上高度集中，保护同时削弱了大量训练样本的残差输入。

这是保护与可塑性的实际取舍，但能量不是任务信息量：
没有无投影的同预算训练消融，不能断言它独自造成失败，也不能推断删除投影就会提高成功率。

## 3. 夹爪修正路径在训练数据上完全饱和

第7维（夹爪）选中分量的预激活绝对值范围
{grip['preactivation_abs']['min']:.4f}–{grip['preactivation_abs']['max']:.4f}，
平均{grip['preactivation_abs']['mean']:.4f}。
1562/1562窗口的float32 tanh导数项1−tanh(b)^2为0，
最终夹爪残差和动作均值变化也全部为0。
其中10个窗口父策略夹爪均值符号与专家目标不一致，校准后仍未改变。

当前残差加在tanh之前且幅度上限0.1；它无法把绝对值至少9.89的预激活推过0。
当tanh已经数值饱和时，直接L1经这条路径回传的夹爪学习信号也为0。
因此本轮结构实际上只能学习前六维的局部连续修正，不能据此期待修复夹爪切换错误。
这是明确的可学习性限制；但训练窗口上的10处符号错误不能对应到七条验证失败，
更不能据此宣称失败一定发生在抓取或释放。

## 4. 离线拟合改善有限，而且不普遍

在同一批1562个训练窗口上，冻结父模型与最终候选配对复算：

- L1：{c['parent_L1']:.8f} → {c['candidate_L1']:.8f}，相对降低{100*c['relative_L1_reduction']:.3f}%。
- {c['improved']}窗口改善、{c['worsened']}窗口恶化；这是训练集拟合，非泛化验收。
- 七维合计的平均绝对动作均值变化{mean_shift['mean']:.8f}，
  中位数{mean_shift['quantiles']['0.5']:.8f}，最大{mean_shift['max']:.8f}。
- 最大调整主要在动作第1维，其平均绝对变化0.0030488；夹爪完全不变。

小的绝对修正并不等于分布改动可以忽略：
同状态选中Gaussian的均值平移KL平均{c['metrics']['gaussian_KL']['mean']:.4f}、
最大{c['metrics']['gaussian_KL']['max']:.4f} nat。
本方案没有KL约束，这不是协议违规；也不能把此离线KL当成验证轨迹在线KL。
父模型某些尺度小，绝对微调仍可能相对噪声有明显影响。
因此证据支持“调整方向没有实现任务级改善”，不支持“只是随机噪声淹没了一切”。

## 5. 成功锚点保护既不等于整轨迹保护，也不提供失败恢复监督

3条既有父成功轨迹共448个历史特征，仅32个被用作锚点。
将最终候选在这些已保存特征上做只读前向：416/448个时刻的最大动作均值变化>1e-6；
每条轨迹最大绝对变化约0.0055–0.0061。锚点本身均值误差0。
因此结构保证针对指定特征空间，不能升级为“成功轨迹完全不动”。
这次验证的三次父成功被保留，是十条观测中的事实，不是对其他初态的保证。

校准仍只模仿原专家窗口，不使用失败后状态的恢复目标。
成功锚点告诉残差哪些方向不要改变，却没有告诉它七条失败在哪里、该怎样挽救。
当错误来自状态偏离、模式选择或夹爪切换时，仅微调冻结读出的选中均值可能不足。
这些是与设计和观测相符的解释，具体失败机制尚未被逐步证据定位。

## 判断

优先保留的成果是独立验证拒绝与精确回退：未观察到获益时，它阻止了候选默认部署。
本次结果不足以支持直接扩展完整阶段或更多seed，也不支持仅靠延长训练、增大幅度、
放宽启用门槛解决问题。最明确的结构问题是饱和夹爪路径；最值得警惕的表示问题是
用线性子空间保护少量成功特征时，也排除了大量普遍存在的特征方向。
没有新增实验，未修改冻结实现或门槛。

## 证据与复算范围

两次只读前向分别耗时6.86秒、7.04秒，第二次针对新发现补查逐维饱和。
均0 optimizer调用、0环境调用、无新闭环；父tensor及冻结合同前后通过，
最终checkpoint/Q/原成功特征/DECISION文件SHA前后相同。
只下载聚合统计与哈希；原模型和成功特征保持服务器原位。
验证原始记录来自../progress_20261020_160301/evidence，
新增统计见OFFLINE_DIAGNOSTIC.json和gripper_followup/OFFLINE_DIAGNOSTIC.json。
执行命令及退出0见两份EXECUTION.json；本报告本地复算命令：

```text
python execution/20261020_sazr_object_seed41/readonly_analysis/analyze.py
```

仅一个seed、一个Task6、10配对验证状态。未记录验证失败轨迹的状态、动作、路由计数、
投影残差或在线逐步KL，无法从现有文件重建这些量；不填补或伪造。
不得把训练缓存上的统计直接外推为失败轨迹的因果诊断。
"""
(R/"REPORT.md").write_text(report,encoding="utf-8")
inputs=[R/"OFFLINE_DIAGNOSTIC.json",R/"gripper_followup/OFFLINE_DIAGNOSTIC.json",
        E/"validation_parent/episodes.jsonl",E/"validation_candidate/episodes.jsonl",E/"DECISION.json"]
(R/"ANALYSIS_ACCEPTANCE.json").write_text(json.dumps(dict(status="PASS",
    existing_pairs=10,training_windows=1562,existing_success_feature_steps=n_trace,
    zero_optimizer_calls=True,zero_environment_calls=True,
    input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}),indent=2)+"\n")
print("PASS: existing 10 validation pairs, 1562 windows, 448 saved feature steps; no training or rollout.")
