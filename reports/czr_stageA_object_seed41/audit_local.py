"""Independent lightweight recomputation; no model, training or environment."""
from pathlib import Path
import json,hashlib,csv,datetime,statistics
R=Path(__file__).resolve().parent
def read(n):return json.loads((R/n).read_text(encoding="utf-8"))
def write(n,x):(R/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
def calc(rows,p="parent_correct",c="candidate_correct"):
    ss=sum(r[p] and r[c] for r in rows)
    sf=sum(r[p] and not r[c] for r in rows)
    fs=sum(not r[p] and r[c] for r in rows)
    ff=sum(not r[p] and not r[c] for r in rows)
    assert ss+sf+fs+ff==len(rows)
    return dict(n=len(rows),SS=ss,SF=sf,FS=fs,FF=ff,parent_correct=ss+sf,
                candidate_correct=ss+fs,net=fs-sf)
rows=read("WINDOW_OUTCOMES_LIGHT.json");remote=read("FINAL_AUDIT.json")
obs=read("FINAL_OBSERVATION.json");contract=read("CONTRACT.json")
assert obs["exit"]==0 and not obs["workers"] and obs["frozen_hash_pass"]
for n,v in obs["downloaded"].items():
    assert hashlib.sha256((R/n).read_bytes()).hexdigest()==v["sha256"],n
for n,h in read("DEPLOYMENT.json")["files"].items():
    assert hashlib.sha256((R/n).read_bytes().replace(b"\r\n",b"\n")).hexdigest()==h,n
assert remote["status"]=="PASS" and remote["frozen_inputs_unchanged"]
assert remote["optimizer_calls"]==remote["environment_constructs"]==remote["environment_steps"]==remote["policy_rollouts"]==0
assert len(rows)==1558 and len(set(r["sample_id"] for r in rows))==1558
groups={s:[r for r in rows if r["split"]==s] for s in ("train","holdout")}
assert [len(v) for v in groups.values()]==[1255,303]
assert all(int(r["demo_id"].split("_")[1])<8 for r in groups["train"])
assert all(int(r["demo_id"].split("_")[1])>=8 for r in groups["holdout"])
assert all(r["motion_exact"] and r["native_feature_exact"] and r["rng_exact"] for r in rows)
assert all(r["rejected_action_exact"] for r in rows if not r["trigger"])
assert all(not r["trigger"] for r in groups["train"] if r["parent_correct"])
metrics={s:calc(v) for s,v in groups.items()}
assert metrics==remote["metrics"]
sampled={s:calc(v,"sampled_parent_correct","sampled_candidate_correct") for s,v in groups.items()}
gate=metrics["holdout"]["FS"]>=1 and metrics["holdout"]["SF"]==0
folds=read("TRAINING_CROSS_DEMO.json")
assert len(folds)==8 and sum(f["metrics"]["n"] for f in folds)==1255
assert all(f["normalization_fit_only"] and f["fit_n"]+f["metrics"]["n"]==1255 for f in folds)
cross={k:sum(f["metrics"][k] for f in folds) for k in ("n","SS","SF","FS","FF","net")}
assert cross["n"]==sum(cross[k] for k in ("SS","SF","FS","FF"))
assert gate==remote["prototype_gate"] and (cross["FS"]>0)==remote["cross_demo_support"]
memory=read("MEMORY.json");cases=read("ERROR_CASES_LIGHT.json")
assert len(cases)==memory["error_centers"]==metrics["train"]["FS"]+metrics["train"]["FF"]
assert len(set(r["event_id"] for r in cases))==memory["error_events"]
assert all(abs(r["radius"]-.45*r["nearest_correct_distance"])<1e-12 for r in cases)
assert set(r["sample_id"] for r in cases)==set(r["sample_id"] for r in groups["train"] if not r["parent_correct"])
inter=read("NATIVE_INTERFACE.json")
assert inter["window_checks"]==1558 and inter["max_feature_difference"]==0 and len(inter["extra_cases"])==24
assert read("SOFTWARE_TESTS.json")["status"]=="PASS"
demo=[]
for d in range(10):
    rr=[r for r in rows if r["demo_id"]==f"demo_{d}"]
    demo.append(dict(demo=f"demo_{d}",split="train" if d<8 else "holdout",**calc(rr),
                     triggers=sum(r["trigger"] for r in rr)))
with (R/"PER_DEMO.csv").open("w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=list(demo[0]));w.writeheader();w.writerows(demo)
summary=dict(status="PASS",stage="A_COMPLETED",metrics=metrics,sampled_command_metrics=sampled,
    cross_demo=cross,prototype_gate=gate,cross_demo_support=cross["FS"]>0,
    recommend_closure=gate and cross["FS"]>0,B="NOT_STARTED",optimizer_calls=0,environment_calls=0,
    windows=1558,excluded=4,native_interface_checks=1558,extra_interface_checks=24,
    memory=memory,seconds=remote["seconds"],time=datetime.datetime.now().astimezone().isoformat())
prior_path=R.parent/"20261020_gripper_repair_stageA/WINDOW_OUTCOMES_LIGHT.json"
prior={r["sample_id"]:r for r in json.loads(prior_path.read_text())}
path_comparison={}
for split,rr in groups.items():
    for r in rr:assert prior[r["sample_id"]]["sha256"]==r["sha256"]
    path_comparison[split]=dict(n=len(rr),
        cached_parent_errors=sum(not prior[r["sample_id"]]["parent_correct"] for r in rr),
        native_parent_errors=sum(not r["parent_correct"] for r in rr),
        errors_in_both=sum(not prior[r["sample_id"]]["parent_correct"] and not r["parent_correct"] for r in rr))
write("CACHE_NATIVE_PARENT_COMPARISON.json",dict(comparison=path_comparison,
    previous_light_file=str(prior_path),previous_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
    interpretation="Input-path comparison, not a change in model performance under the same input."))
neighbor=read("NEIGHBOR_AUDIT.json") if (R/"NEIGHBOR_AUDIT.json").exists() else None
if neighbor:
    assert neighbor["features_sha256"]==memory["native_features_sha256"]
    assert neighbor["saved_features_unchanged"] and neighbor["environment_calls"]==neighbor["optimizer_calls"]==0
    summary["neighbor_groups"]=neighbor["groups"]
write("LOCAL_ACCEPTANCE.json",summary)
lines=["# CZR：原生图像局部纠错 Stage A 最终报告","",
       "本轮零训练、零环境执行。工程验收通过；以下标签纠错结果不等于物理任务修复。","",
       "| 数据 | 窗口 | 父策略正确 | CZR正确 | 纠正FS | 损害SF | 净变化 |",
       "|---|---:|---:|---:|---:|---:|---:|"]
for s,m in metrics.items():
    lines.append(f"| {s} | {m['n']} | {m['parent_correct']} ({100*m['parent_correct']/m['n']:.4f}%) | {m['candidate_correct']} ({100*m['candidate_correct']/m['n']:.4f}%) | {m['FS']} | {m['SF']} | {m['net']} |")
lines+=["",f"进入闭环的留出门槛：**{'PASS' if gate else 'FAIL'}**（FS≥1且SF=0）。",
        f"训练演示逐示范留出：SS={cross['SS']}、SF={cross['SF']}、FS={cross['FS']}、FF={cross['FF']}；"
        f"跨演示纠错支持：**{'观察到' if cross['FS']>0 else '证据不足'}**。",
        "半径固定0.45，从未依据留出结果调整；逐示范留出只检查支持，不进行阈值搜索。",
        "",f"记忆包含{memory['error_centers']}个错误窗口中心，组织为{memory['error_events']}个连续目标时刻事件；"
        f"正确保护集{memory['correct_protection']}条。数值载荷{memory['numeric_payload_bytes']}字节，"
        f"实际序列化{memory['file_bytes']}字节（仅服务器保留）。"]
if cases:
    radii=[r["radius"] for r in cases]
    lines.append(f"归一化空间修正半径：最小{min(radii):.8g}，中位数{statistics.median(radii):.8g}，最大{max(radii):.8g}。")
if neighbor:
    for group in ("cross_demo_error","holdout_error"):
        g=neighbor["groups"][group];v=g["ratio"]
        lines.append(f"{group}：{g['n']}个父错误，落入任一匹配修正球{g['inside']}个；"
                     f"最小距离/对应半径之比的统计为{v}（必须严格小于1才有局部支持）。")
lines+=["","原缓存与原生图像的父错误计数并不相同："]
for split,v in path_comparison.items():
    lines.append(f"- {split}：旧缓存{v['cached_parent_errors']}，原生图像{v['native_parent_errors']}，"
                 f"两种路径均错误{v['errors_in_both']}。")
lines.append("因此不把上一轮缓存上的99%准确率直接当作本轮原生图像准确率；本轮没有改变父模型。"
             "原缓存来自train模式及原增广，当前为完整eval路径；未单独分离每个模式因素的贡献。")
lines+=["","## 逐演示结果","",
        "| demo | 划分 | n | 父正确 | CZR正确 | FS | SF | 触发 |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
for d in demo:
    lines.append(f"| {d['demo']} | {d['split']} | {d['n']} | {d['parent_correct']} | {d['candidate_correct']} | {d['FS']} | {d['SF']} | {d['triggers']} |")
lines+=["","## 实际执行接口与输入保护","",
        "- 1558条原始图像窗口逐条检查真实get_action接口；查询特征与构造特征逐元素一致。",
        "- 训练集所有原本正确窗口均无干预；前六维动作、拒绝状态完整动作、CPU/CUDA RNG消耗逐项一致。",
        "- 额外24条history1/4/8与规范/未知指令接口检查通过；6项合成性质测试通过，均不算实验成功结果。",
        "- 父参数张量、原模型/特征/动作文件及冻结代码前后哈希一致；原子任务退出0、无残留worker。",
        f"- 原生提取/诊断总耗时{remote['seconds']:.1f}秒；未创建optimizer、未训练、未构造或执行环境。",
        "",
        "## 解释边界与后续","",
        "训练错误中心上的修复是记忆内性质，不能据此推断泛化。demo8–9仅相对新纠错记忆留出，父模型此前训练时见过全部10条演示。",
        "相邻窗口相关，不是独立错误事件；本报告同时给出事件计数。原生图像离线检查也不等于闭环状态分布。",
        "构造使用最大logits分量均值的夹爪类别；实际Normal采样命令的配对正确性另存LOCAL_ACCEPTANCE，不混为一项指标。",
        "本轮保留前六维，仅检验保守局部例外机制。不能证明夹爪是失败根因，也不能声称已修复完整7维控制。",
        "半径系数0.45、归一化epsilon1e-12及严格小于在执行前固定。有限正确集合保护不保证未知正确状态安全。",
        "不启动闭环B；若门槛通过且获得支持，仍需先固定独立初态、配对噪声与预算。可信同状态专家来源仍未确定。",
        "",
        "## 可复现命令","",
        "```text",
        "python execution/20261020_czr_stageA/remote.py launch",
        "python execution/20261020_czr_stageA/audit_local.py",
        "python execution/20261020_czr_stageA/publish.py",
        "```","",
        "远端真实命令及退出码见DEPLOYMENT.json、COMMAND_EXIT.json、run.exit；发布回执保存在本地。",
        "远端结果：`/data/mlr_ifa/analyses/CZR_A_OBJECT_seed41_20261020`。特征与纠错记忆不下载不发布。"]
(R/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
print(json.dumps(summary,ensure_ascii=False,indent=2))
