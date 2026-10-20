"""Independent recomputation of the 10 paired outcomes from downloaded raw records."""
from pathlib import Path
import json,hashlib,datetime,math
R=Path(__file__).resolve().parent;E=R/"evidence"
def js(p):return json.loads(p.read_text(encoding="utf-8-sig"))
def rows(p):return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
for n,v in js(R/"DOWNLOAD_MANIFEST.json").items():
    raw=(E/n).read_bytes();assert len(raw)==v["bytes"] and hashlib.sha256(raw).hexdigest()==v["sha256"]
    remote_manifest=js(E/"PROCESS_EXIT_ACCEPTANCE.json")["manifest"]
    if v["source"] in remote_manifest:assert v["sha256"]==remote_manifest[v["source"]]["sha256"]
for name in ("freeze","preflight_repair","parent9","teacher7","audit","pipeline","final_process_audit"):
    assert (E/"commands"/(name+".exit")).read_text().strip()=="0"
assert (E/"commands/preflight.exit").read_text().strip()=="1"
assert js(E/"PROCESS_EXIT_ACCEPTANCE.json")["status"]=="PASS"
assert js(E/"PREFLIGHT.json")["status"]=="PASS"
assert js(E/"PREFLIGHT.json")["environment_calls"]==js(E/"PREFLIGHT.json")["optimizer_calls"]==0
pp=rows(E/"parent9/episodes.jsonl");tt=rows(E/"teacher7/episodes.jsonl")
assert len(pp)==len(tt)==10
assert [r["initial_index"] for r in pp]==[r["initial_index"] for r in tt]==list(range(20,30))
categories=dict(SS=0,SF=0,FS=0,FF=0)
for phase,rs in (("parent9",pp),("teacher7",tt)):
    events=rows(E/phase/"attempts.jsonl")
    assert [x["event"] for x in events]==["environment_construct_start","environment_construct_complete","vector_start","vector_complete","environment_closed"]
    steps=rows(E/phase/"step_hashes.jsonl")
    assert [x["step"] for x in steps]==list(range(rs[0]["vector_steps"]))
    assert len(steps)<=600
    assert sum(r["success"] for r in rs)==js(E/phase/"COMPLETE.json")["successes"]
    for r in rs:
        assert r["checkpoint_sha256"]==js(E/"RESOURCE_AUDIT.json")["checkpoints"][str(r["stage"])]["sha256"]
        assert r["success"]==(r["first_success_policy_step"] is not None)
        if r["success"]:assert 0<=r["first_success_policy_step"]<len(steps)
for p,t in zip(pp,tt):
    for key in ("initial_index","initial_sha256","env_seed","noise_seed","noise_sha256"):assert p[key]==t[key]
    cat=("S" if p["success"] else "F")+("S" if t["success"] else "F");categories[cat]+=1
ps=rows(E/"parent9/step_hashes.jsonl");ts=rows(E/"teacher7/step_hashes.jsonl")
assert all(p["noise_sha256"]==t["noise_sha256"] for p,t in zip(ps,ts))
assert categories==js(E/"SCREEN_AUDIT.json")["counts"]
parent=categories["SS"]+categories["SF"];teacher=categories["SS"]+categories["FS"]
assert teacher-parent==categories["FS"]-categories["SF"]
assert teacher==js(E/"SCREEN_AUDIT.json")["teacher_successes"] and parent==js(E/"SCREEN_AUDIT.json")["parent_successes"]
configp=js(E/"parent9/resolved_actual.json");configt=js(E/"teacher7/resolved_actual.json")
for config in (configp,configt):
    assert all(config["policy"][k] is True for k in ("use_state","use_language","use_vision"))
    assert config["seed"]==42 and config["eval"]["max_steps"]==600
differences=[]
def diff(a,b,path=""):
    if isinstance(a,dict) and isinstance(b,dict):
        assert set(a)==set(b)
        for k in a:diff(a[k],b[k],path+"/"+k)
    elif a!=b:differences.append(dict(path=path,parent=a,teacher=b))
diff(configp,configt)
assert all(x["path"]=="/experiment_dir" for x in differences),differences
now=datetime.datetime.now().astimezone().isoformat()
commands=rows(E/"commands/commands.jsonl")
durations={}
for label in ("parent9","teacher7","audit"):
    start=next(r for r in commands if r["label"]==label and "start" in r)
    end=next(r for r in commands if r["label"]==label and "end" in r)
    assert end["exit"]==0
    durations[label]=end["end"]-start["start"]
result=dict(status="PASS",observed_at=now,episodes=20,pairs=10,vector_attempts=2,
    counts=categories,parent_successes=parent,teacher_successes=teacher,difference_pp=(teacher-parent)*10,
    teacher_better_in_this_screen=teacher>parent,config_differences=differences,
    matched_noise_all_common_steps=True,local_manifest_hashes_verified=True,
    no_automatic_distillation=True,command_seconds=durations,
    loop_controls=dict(episodes_per_model=10,lanes=10,zero_steps=5,max_policy_steps=600,
        inherited_eval_n_eval=configp["eval"]["n_eval"],
        note="Inherited n_eval/num_procs fields are not consumed by the fixed screen loop; actual budget comes from frozen screen.py and CONTRACT.json."))
(R/"LOCAL_ACCEPTANCE.json").write_text(json.dumps(result,indent=2)+"\n")
(R/"RUN_STATUS.json").write_text(json.dumps(dict(status="SCREEN_COMPLETED",time=now,episodes=20,full_ovr="NOT_STARTED"),indent=2)+"\n")
table="\n".join("| %d | %s | %s | %s |"%(p["initial_index"],"成功" if p["success"] else "失败","成功" if t["success"] else "失败",("S" if p["success"] else "F")+("S" if t["success"] else "F")) for p,t in zip(pp,tt))
text=f"""# OVR 20条探索性教师筛查

状态：SCREEN_COMPLETED；本地独立复算PASS；{now}。

仅OBJECT seed42 Task6，RA+IFA阶段9父模型和固定阶段7教师各10条，初态20–29，2个10路vector。未训练任何残差，未追加环境smoke、初态或重试。原模型和输入哈希未变。

两组闭环命令用时：父模型{durations['parent9']:.1f}秒，教师{durations['teacher7']:.1f}秒；合计{(durations['parent9']+durations['teacher7'])/60:.2f}分钟（含模型与环境初始化）。

| 策略 | 成功 | 成功率 |
|---|---:|---:|
| 阶段9父模型 | {parent}/10 | {parent*10}% |
| 阶段7教师 | {teacher}/10 | {teacher*10}% |

教师减父模型：{(teacher-parent)*10:+d}个百分点。SS={categories['SS']}、SF={categories['SF']}、FS={categories['FS']}、FF={categories['FF']}，方向为父→教师；净差FS−SF={teacher-parent}。

| 初态 | 父模型 | 教师 | 配对 |
|---|---|---|---|
{table}

## 执行与验证

从既有seed42 env.sh激活目标环境后，在独立工作区依次执行：

```
python screen.py freeze
python preflight_repair.py
python pipeline.py
```

pipeline顺序执行`python screen.py rollout 9`、`python screen.py rollout 7`、`python screen.py audit`。上述退出码全部0；外置`final_process_audit.py`和本地`audit_local.py`通过。原生GPU接口history1/4/8、两模型100项下游参数完整加载、argmax分量均值和标准差、原生随机采样复现检查通过；0 optimizer、预检0环境。首次`python screen.py preflight`退出1：手工测试期望遗漏原头部均值tanh，外置`preflight_repair.py`只修复该测试表达式，原冻结推理/模型/合同不改，失败日志与exit1保留。

每初态独立固定CPU Normal表seed=42000000+初态号，600×7。两策略相同初态/步号的噪声哈希逐步一致，环境seed42；原5零动作、600步上限、8步历史、GMM方差不变。两份resolved config只有输出目录不同。

配置中的原eval.n_eval=20、num_procs=20为继承字段，本轮固定筛查循环不读取这两项；实际每模型10条、10路，由冻结screen.py、CONTRACT.json与attempt账目共同核验，不是每模型20条。三模态字段均为布尔True。GPU为RTX3090，PyTorch1.13.0+cu116；cudnn benchmark=True，TF32 matmul=False/cudnn=True。

## 解释边界与下一步

本轮教师{'优于' if teacher>parent else '未优于'}父模型。这只回答是否观察到可供后续迁移的历史控制能力，不能证明轻量残差能学会该能力，也不能证明教师每个单步动作正确。

本次仅多成功{teacher-parent}个初态（{categories['FS']}条父失败→教师成功、{categories['SF']}条反向损害）；历史80%对10%的巨大优势没有在这批匹配噪声筛查中复现。初态集合和采样随机流都与历史评估不同，不能单独将差距收缩归因于其中某一个因素。现有证据不足以声称教师有稳定的大幅优势，也不足以宣布OVR残差有效；仅有的救援案例不能替代跨初态蒸馏和独立验证。

这10个初态此前已用于其他实验，本轮明确是探索性复用；教师与seed也受历史结果启发。双方共享匹配噪声有助于配对，但仍只有每初态1次，不能分离全部环境和采样波动。历史0–19的16/20与2/20没有混入本次结果，本轮也不更新原AUC/FWT/NBT。

完整蒸馏、留出、独立验收与20初态正式测试的分区和预算尚未授权，故筛查完成后停止。没有自动启动残差训练或扩大seed。

资源审计v1失败是module.参数名前缀解析遗漏；v2失败是目标Python3.8不支持removeprefix；v3解析修复后退出0。均在任何模型前向/环境之前，原权重不改，失败证据保留。
"""
(R/"REPORT.md").write_text(text,encoding="utf-8")
print(json.dumps(result,indent=2,ensure_ascii=False))
