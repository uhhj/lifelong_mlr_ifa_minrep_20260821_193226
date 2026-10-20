"""Independent local repair audit and report; no simulator/model imports."""
from pathlib import Path
import json,hashlib,collections,datetime
R=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text(encoding="utf-8"))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
bundle=js(R/"LIGHT_BUNDLE.json")
for rel,v in bundle.items():
 assert not Path(rel).is_absolute() and ".." not in Path(rel).parts
 raw=v["text"].encode("utf-8")
 assert len(raw)==v["bytes"] and hashlib.sha256(raw).hexdigest()==v["sha256"],rel
 p=R/"evidence"/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
E=R/"evidence/output";a=js(E/"FINAL_REPAIR_ACCEPTANCE.json")
first=js(E/"REPAIR_DIAGNOSTIC.json");second=js(E/"integration_probe/RESULT.json")
assert a["status"]=="BLOCKED_HISTORICAL_FIDELITY"
assert a["thresholds_unchanged"] and a["original_failure_evidence_unchanged"] and a["inputs_unchanged"]
assert a["exits"]==dict(repair=0,integration=0) and not a["workers"]
events=[json.loads(line) for p in (E/"attempts.jsonl",E/"integration_probe/attempts.jsonl")
 for line in p.read_text(encoding="utf-8").splitlines()]
counts=dict(collections.Counter(e["event"] for e in events))
assert counts==a["counts"] and counts["step_complete"]==328
assert a["policy_continuations"]==a["optimizer_calls"]==0 and a["total_engineering_steps"]==644
for x,y,z in zip(a["rows"],first["rows"],second["rows"]):
 assert x["demo_id"]==y["demo_id"]==z["demo_id"]
 assert x["joint_error"]==max(v["obs"]["joint_states"] for v in y["corrected_observations"])==0
 assert x["gripper_error"]==max(v["obs"]["gripper_states"] for v in y["corrected_observations"])==0
 assert x["full_snapshot_repeat_error"]==max(z["snapshot_repeat_error"],z["snapshot_observation_repeat_error"],
  max(v["restore_error"] for v in z["snapshot_trials"]))==0
 assert x["recorded_transition_error"]==z["forced_prefix"][-1]["max_error"]>1e-5
 assert not x["recorded_transition_pass"]
 # Exactly eight correctly indexed frames for each fixed RA sample.
 assert len(y["corrected_observations"])==8
 assert all(v["state_index"]==v["index"]+1 for v in y["corrected_observations"])
now=datetime.datetime.now().astimezone().isoformat()
write(R/"LOCAL_ACCEPTANCE.json",dict(status="PASS_FOR_BLOCKED_REPAIR_REPORT",time=now,
 metadata_hashes_verified=len(bundle),all_counts_recomputed=True,rows=a["rows"],
 new_engineering_steps=328,total_engineering_steps=644,policy_continuations=0,optimizer_calls=0))
write(R/"RUN_STATUS.json",dict(status=a["status"],time=now,
 scope="Engineering repair only; original 54 continuation budget remains unspent",
 fixed=["joint/gripper observation time alignment","same-snapshot repeated transitions at three tested starts"],
 unresolved=["recorded transition fidelity > 1e-5","all original observation fields not yet reproduced within 1e-6"],
 policy_continuations=0,optimizer_calls=0))
lines=["# RCTR状态恢复修复：部分通过，历史轨迹一致性仍阻塞","",
 "**结论：没有放宽阈值。已修复关节/夹爪观察的时序对应和本轮测试下的快照重复一致性，但仍未通过历史轨迹复现门槛，因此未启动原定54条筛查。没有训练、优化器调用或B/C。**","",
 "## 已修复并验证的部分","",
 "1. 以真实数据校验`states[i+1]`与`obs[i]`：三个固定窗口的全部8帧，关节和夹爪误差均为0。原训练数据及RA绑定未重标或修改。",
 "2. XML加载后显式清空MuJoCo运行状态，再写入物理状态；强制更新控制器缓存，避免控制器继续使用reset姿态。按已记录前缀恢复OSC保留目标和夹爪累积命令，未用目标状态反向拟合控制参数。",
 "3. 使用MuJoCo官方`mjSTATE_INTEGRATION`保存/恢复积分状态（含warmstart、ctrl和外力等），并恢复控制器、夹爪、robot近期缓存、observable缓存/时钟及Python/NumPy RNG。在三个测试起点、同一固定专家动作各两次的检验中，状态、图像和低维观察重复误差均为0；通过1e-10标准。这是有限测试的结果，不宣称覆盖任意环境状态。",
 "",
 "官方数据生成程序先执行动作，再记录返回观察，同时按原索引保存states。此代码与实际数据的一步差一致；不能据此认定本地数据一定来自该master版本。参见[LIBERO create_dataset.py](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/scripts/create_dataset.py)。",
 "",
 "## 尚未通过的硬门槛","",
 "| 演示 | 关节/夹爪对齐误差 | 完整快照重复误差 | 历史一步转移最大误差 | 位置坐标最大误差 | 速度坐标最大误差 |",
 "|---|---:|---:|---:|---:|---:|"]
for r in a["rows"]:
 lines.append("| %s | 0 / 0 | %.2g | %.10g | %.10g | %.10g |"%(r["demo_id"],r["full_snapshot_repeat_error"],
  r["recorded_transition_error"],r["recorded_transition_qpos"],r["recorded_transition_qvel"]))
lines += ["",
 "历史一步转移阈值仍为**1e-5**，不计绝对时间；表中max error来自混合单位的MuJoCo坐标向量，不是米或成功率。位置与速度分开报告，但没有仅检查位置、删除静态物体或对角速度放宽标准。",
 "",
 "- demo_0/1最大剩余误差在`wine_bottle_1_joint0`的角速度坐标，约2.0087e-4；奶酪、盘子、碗也有较小差异。",
 "- demo_2最大剩余误差在`akita_black_bowl_1_joint0`的角速度坐标，约0.06033；同时有机器人关节和奶酪坐标偏差。不能当成无关背景物体小漂移忽略。",
 "- 校正索引虽然使关节/夹爪完全一致，末端位置等派生观察仍未达到原1e-6标准；图像误差也完整保留。没有把已通过的部分代替整体准入。",
 "- 普通同步前缀重放、逐记录状态重建控制器、以及携带积分隐状态的前缀检查均已留证。最后一种每步写入记录坐标仅用于定位隐状态差异；它不是已成功的无干预整段重放，更没有被用作54条分支里的作弊恢复。",
 "",
 "## 为什么现在仍不能继续","",
 "“从一个当前快照重复执行能完全一致”和“这个快照完整复现了历史演示”是不同条件。前者已在固定测试上通过，后者没有。继续分支筛查会绕过用户明确要求不放宽的准入门槛。",
 "",
 "现有HDF5只提供flat物理状态和动作/观察，没有保存当年的完整控制器、求解器及observable快照；生成程序还裁去前五步。当前证据不能唯一确定剩余误差来自采集运行时、模型/接触实现、未记录的初始控制状态或其他隐状态。未通过安装多个物理版本、调接触参数或拟合目标状态来挑一个看似通过的结果。",
 "",
 "后续若要保持同一严格验收，需要可核验的原始完整状态快照，或可重建的采集/回放运行时及缺失前缀等来源证据。**不能把同快照重复通过当成历史复现通过，也不能在未通过时自动转入54条。** 本次不启动新实验替代这些条件。",
 "",
 "## 执行、退出与证据","",
 "- `source parent/env.sh; python restoration_repair/repair_probe.py`：退出0；24帧观察恢复、12次直接转移、155步固定前缀，共167工程step。",
 "- `source parent/env.sh; python restoration_repair/integration_probe.py`：退出0；155步带积分状态的诊断前缀＋6次快照重复，共161工程step。",
 "- `python restoration_repair/final_audit.py`：退出0；旧失败文件和输入哈希不变，退出与无残留进程检查通过。",
 "- `python execution/20261020_rctr_goal_seed42_stageA/restoration_repair/audit_local.py`：本地独立复算计数、哈希与阈值。验收通过的是这份如实的BLOCKED报告，不是54条实验。",
 "- 本轮328工程step、2次环境构建并关闭、42次reset，0技术异常。加上一轮316步，共644工程step。策略续跑仍0/54，optimizer仍0。工程step不是episode。",
 "",
 "旧的`probe.exit=1`、资产修复后诊断exit0、原BLOCKED报告与全部失败证据保持不变；新证据单独保存。未改父模型、RA成员、原数据、状态模态、训练协议或原接受阈值。",
 "",
 "远端新产物：`/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020/restoration_repair`。",
 "发布仅含代码、报告、哈希和工程统计；不含权重、动作载荷、潜特征、完整状态或RNG二进制。","",
 "本地审计时间："+now]
(R/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
write(R/"DOWNLOAD_MANIFEST.json",{k:dict(sha256=v["sha256"],bytes=v["bytes"]) for k,v in bundle.items()})
write(R.parent/"LATEST_REPAIR.json",dict(status=a["status"],time=now,report="restoration_repair/REPORT.md",
 original_report_preserved="REPORT.md",policy_continuations=0,optimizer_calls=0))
print(json.dumps(dict(status="PASS_FOR_BLOCKED_REPAIR_REPORT",counts=counts,rows=a["rows"]),indent=2))
