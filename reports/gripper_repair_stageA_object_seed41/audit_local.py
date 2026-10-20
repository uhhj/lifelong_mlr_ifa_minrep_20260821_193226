"""Recompute Stage A label metrics and native-interface limitations, no torch."""
from pathlib import Path
from collections import Counter
import json,csv,hashlib,datetime
R=Path(__file__).resolve().parent
js=lambda p:json.loads(p.read_text(encoding="utf-8"))
snapshot=js(R/"FINAL_OBSERVATION.json")
assert not snapshot["workers"]
assert snapshot["run_exit"]==snapshot["native_audit_exit"]==0
assert snapshot["source_input_hash_pass"]
full=R/"WINDOW_OUTCOMES.json"
if full.exists():
    for n,m in snapshot["files"].items():
        assert hashlib.sha256((R/n).read_bytes()).hexdigest()==m["sha256"]
    records=js(full)
    for r in records:
        assert r["parent_correct"]==(r["parent_predicted_class"]==r["target_class"])
        assert r["candidate_correct"]==(r["candidate_predicted_class"]==r["target_class"])
    # Publish only identities, payload hashes and boolean outcome metadata.
    light=[{k:v for k,v in r.items() if k not in ("target_class","parent_predicted_class","candidate_predicted_class")} for r in records]
    (R/"WINDOW_OUTCOMES_LIGHT.json").write_text(json.dumps(light,indent=2)+"\n")
else:
    records=js(R/"WINDOW_OUTCOMES_LIGHT.json")
    for n,m in snapshot["files"].items():
        if n=="WINDOW_OUTCOMES.json":continue
        assert hashlib.sha256((R/n).read_bytes()).hexdigest()==m["sha256"]
assert len(records)==1558 and len({r["sample_id"] for r in records})==1558
split=js(R/"SPLIT.json")
assert len(split["excluded"])==4
assert {r["demo_id"] for r in records if r["split"]=="train"}=={f"demo_{i}" for i in range(8)}
assert {r["demo_id"] for r in records if r["split"]=="holdout"}=={"demo_8","demo_9"}
remote=js(R/"FINAL_AUDIT.json")
def metric(rows):
    n=len(rows);p=sum(x["parent_correct"] for x in rows);c=sum(x["candidate_correct"] for x in rows)
    repair=sum(not x["parent_correct"] and x["candidate_correct"] for x in rows)
    harm=sum(x["parent_correct"] and not x["candidate_correct"] for x in rows)
    assert c-p==repair-harm
    return dict(n=n,parent_correct=p,candidate_correct=c,parent_errors=n-p,label_repairs=repair,label_harms=harm,
                parent_accuracy=100*p/n,candidate_accuracy=100*c/n,delta_pp=100*(c-p)/n)
metrics={k:metric([x for x in records if x["split"]==k]) for k in ("train","holdout")}
for k,m in metrics.items():
    for name in ("n","parent_correct","candidate_correct","parent_errors","label_repairs","label_harms"):
        assert m[name]==remote["metrics"][k][name]
demo=[dict(demo_id=f"demo_{i}",**metric([x for x in records if x["demo_id"]==f"demo_{i}"])) for i in range(10)]
with (R/"PER_DEMO.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(demo[0]));w.writeheader();w.writerows(demo)
epochs=js(R/"epochs.json")
assert [x["epoch"] for x in epochs]==list(range(10))
assert sum(x["batches"] for x in epochs)==1260 and sum(x["samples"] for x in epochs)==12550
assert all(x["grad_min"]>0 for x in epochs)
assert js(R/"PREFLIGHT.json")["isolated_optimizer_calls"]==3
assert remote["strict_checkpoint_load"] and remote["environment_steps"]==remote["environment_constructs"]==0
interface=js(R/"NATIVE_INTERFACE.json");assert len(interface["cases"])==24
assert all(x["motion_exact"] and x["classifier_matches_actual"] for x in interface["cases"])
assert all(x["protected_full_exact"] for x in interface["cases"] if x["task"]!=6)
native=js(R/"NATIVE_ERROR_AUDIT.json")
assert native["optimizer_calls"]==native["environment_calls"]==0
assert native["source_inputs_unchanged"] and native["checkpoint_unchanged"]
assert len(native["rows"])==25 and all(x["status"]=="PASS" for x in native["rows"])
def pair(rows):
    c=Counter(("S" if x["raw_image_parent_mean_correct"] else "F")+("S" if x["raw_image_candidate_correct"] else "F") for x in rows)
    return {k:c[k] for k in ("SS","SF","FS","FF")}
native_all=pair(native["rows"])
native_original_errors=pair([x for x in native["rows"] if not x["cache_parent_correct"]])
assert native_original_errors==dict(SS=6,SF=0,FS=0,FF=4)
assert native_all==dict(SS=10,SF=9,FS=0,FF=6)
verdict=dict(status="STAGE_A_ENGINEERING_COMPLETE_BEHAVIORAL_REQUIREMENT_NOT_MET",
             engineering="PASS",saturated_action_can_flip=True,learned_cached_correction=True,
             preserve_parent_correct_windows=False,positive_holdout_net_gain=False,
             native_confirmed_error_repair_observed=False,stage_B="DEFERRED_BY_USER",stage_C="NOT_STARTED",
             formal_training_steps=1260,isolated_steps=3,environment_steps=0,
             metrics=metrics,native_selected25=native_all,native_original_cache_errors10=native_original_errors,
             limitation="Label correctness is not physical action correctness or causal rescue.")
(R/"LOCAL_ACCEPTANCE.json").write_text(json.dumps(dict(status="PASS",verdict=verdict),indent=2)+"\n")
(R/"STAGE_A_VERDICT.json").write_text(json.dumps(verdict,indent=2)+"\n")
table="\n".join(f"|{'训练 demo0–7' if k=='train' else '留出 demo8–9'}|{m['n']}|{m['parent_correct']}（{m['parent_accuracy']:.2f}%）|{m['candidate_correct']}（{m['candidate_accuracy']:.2f}%）|{m['label_repairs']}|{m['label_harms']}|{m['delta_pp']:+.2f}|" for k,m in metrics.items())
report=f"""# 夹爪动作可学习性 Stage A：工程完成，行为准入未通过

本轮按用户明确回复只完成A，B专家来源之后确定。没有环境构建、物理步或闭环rollout；
未启动B/C、未重训父网络或改变旧buffer/IFA。工程执行与独立验收PASS，不等于方法有效。

## 固定实现与数据

RA+IFA OBJECT seed41 Task6 ep99冻结。新增零初始化Linear(512,1)夹爪分类头，
BCE监督真实末端夹爪标签，原前六维动作完整保留。规范Task6才替换实际动作第7维；
其他任务和未知语言精确回退。正负号核对真实PandaGripper.format_action：−1张开，+1闭合。
只在已有专家图像上执行策略前向；强制正负logit的接口测试属于软件能力证据，不是学到的纠错效果。

固定demo0–7训练1255窗口、demo8–9留出303窗口；4个跨演示绑定歧义窗口排除。
父网络曾见过全部10条示范，因此这里是“新头未使用留出演示训练”，不是全模型从未见过。
10完整epoch、1260正式optimizer step、12550曝光；3次隔离恢复预检；
固定最后ep9，不挑checkpoint或追加预算。全部梯度非零且有限，BCE
{epochs[0]['BCE']:.6f} → {epochs[-1]['BCE']:.6f}。

## 训练缓存上的配对标签结果

|集合|窗口数|父正确数|候选正确数|纠正父错|破坏父对|准确率差pp|
|---|---:|---:|---:|---:|---:|---:|
{table}

留出2个父错误都被纠正，但增加15个错误，净少13个正确。
这满足“存在可表达、可学习的翻转”，不满足“保持父策略原本正确动作”或“整体更可靠”。
样本高度相关且只有2条留出演示，不能把303个窗口当303次独立试验。
这些是专家标签一致性，不是物理错误标签，也不是任务救援率。

## 实际get_action复核揭示缓存与原生输入差异

history1/4/8 × 7个规范任务及1条未知自然语言，共24个真实图像接口检查：
前六维配对动作bitwise相同，保护指令全动作bitwise相同，当前夹爪实际输出与分类规则一致。
固定±logit均能在原生接口输出±1，证明绕过原饱和结构；无仿真调用。

随后对所有10个缓存父错误，以及15个缓存留出损害窗口（共25个、非随机抽样）做只读图像前向，
不更新权重。完整动作窗口身份均有唯一start，标签一致；结果如下：

- 原10个缓存父错误在真实图像评估路径上只有4个仍是父错误，另外6个父已正确。
- 这4个原生父错误，新头一个也没有修复（FS=0、FF=4）。
- 全部25个定向诊断窗口：SS10、SF9、FS0、FF6。不得将它当作全留出集准确率或闭环成功率。
- 缓存留出纠正2个中的一个，原生父本来已正确；另一个原生父仍错，新头也仍错。

因此“缓存纠错2/2”不能被写成“实际执行纠错2/2”。同一原生输入下分类到动作接口是一致的；
变化发生在输入特征/模式层面。实际FEATURES清单记录训练模式与原增广，
DataAugGroup源码在eval直接返回图像。原生图像不等于曾经导出的增广缓存表示；
本轮没有分离增广、前端模块模式等因素，不能把全部差异因果归于单一因素。

## 验收与解释

CPU/CUDA各7项软件性质、真实3调用恢复、24个原生接口、25个定向原生诊断均完成。
新头严格加载PASS；父模型tensor、所有冻结输入与源码前后哈希一致。
正式执行run.exit=0，原生只读诊断native.exit=0，最终无相关worker。
工程PASS；科学结论是：结构饱和限制被解除，但当前独立分类头既未保住多数原本正确预测的全部，
也未在检查到的原生父错误上展示真实修复。不要据此自动进入B或完整闭环。

新头从头学习并全面接管当前任务的夹爪，可能丢失父模型已具有的细致判别能力；
父留出标签准确率已99.34%，潜在收益只有2个，误改正确窗口的成本很大。
固定10轮是低成本原型，不能排除其他容量或训练预算的影响，本轮未做搜索。
无成功Q、无运动残差是本轮为隔离夹爪结构而明确登记的选择，不称原SAZR完整升级验证。

B静态核查：get_sim_state仅返回MuJoCo扁平state，不能假定完整包含gripper.current_action、
控制器目标、观测历史、RNG等。尚未验证精确中途恢复，没有可信同状态专家，
不能声称反事实或物理救援。依用户要求B/C暂不执行。

## 命令、证据与发布

```text
python execution/20261020_gripper_repair_stageA/launch.py
# 远端 source .../GripperRepair_A_OBJECT_seed41_20261020/env.sh; python run_stage_a.py
# 远端同env：python - < .../analyses/GripperRepair_A_OBJECT_seed41_20261020/native_error_audit.py
python execution/20261020_gripper_repair_stageA/final_snapshot.py
python execution/20261020_gripper_repair_stageA/audit_local.py
```

以上实际执行退出0。DEPLOYMENT.json、COMMAND_EXIT.json、run.exit、native.exit、
FINAL_OBSERVATION.json、LOCAL_ACCEPTANCE.json保存执行与验收证据。
远端新模型与完整产物：/data/mlr_ifa/experiments/GripperRepair_A_OBJECT_seed41_20261020。
发布仅代码、报告、哈希、布尔结果及示范身份；不发布模型、原动作/标签载荷、演示、latent或RNG。
原WINDOW_OUTCOMES.json不发布，WINDOW_OUTCOMES_LIGHT.json删除动作符号标签和预测值，
保留复算纠正/损害计数所需的布尔结果。
"""
(R/"REPORT.md").write_text(report,encoding="utf-8")
print(json.dumps(verdict,indent=2))
