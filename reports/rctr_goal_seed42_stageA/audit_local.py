"""Independent metadata verification and report. Never invokes simulator or model."""
from pathlib import Path
import json,hashlib,collections,datetime
R=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text(encoding="utf-8"))
def write(p,r):p.write_text(json.dumps(r,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
bundle=js(R/"LIGHT_BUNDLE.json")
for rel,v in bundle.items():
 assert not Path(rel).is_absolute() and ".." not in Path(rel).parts
 raw=v["text"].encode("utf-8")
 assert len(raw)==v["bytes"] and hashlib.sha256(raw).hexdigest()==v["sha256"],rel
 p=R/"evidence"/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
E=R/"evidence";a=js(E/"workspace/ASSETS.json")
probe=js(E/"output/recovery_asset_paths/STATE_PROBE.json")
exit_audit=js(E/"output/PROCESS_EXIT_ACCEPTANCE.json")
assert probe["status"]=="BLOCKED_STATE_RESTORATION"
assert exit_audit["status"]=="PASS_BLOCKED_SCIENTIFIC_ADMISSION"
assert exit_audit["original_exit"]==1 and exit_audit["recovery_diagnostic_exit"]==0
assert exit_audit["policy_continuations"]==exit_audit["optimizer_calls"]==0
assert not exit_audit["worker_processes"] and probe["inputs_unchanged"]
events=[]
for p in (E/"output/state_attempts.jsonl",E/"output/recovery_asset_paths/state_attempts.jsonl"):
 events += [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
counts=collections.Counter(r["event"] for r in events)
assert dict(counts)==exit_audit["counts"]
assert counts["engineering_step_start"]==counts["engineering_step_complete"]==316
assert counts["reset_start"]==counts["reset_complete"]==42
assert counts["environment_construct_complete"]==counts["environment_closed"]==2
assert counts["technical_exception"]==1
assert len(a["chosen"])==len(probe["rows"])==3
table=[];obs_rows=[]
for selection,row in zip(a["chosen"],probe["rows"]):
 assert (selection["demo_id"],selection["sample_id"],selection["target"])==(row["demo_id"],row["sample_id"],row["target"])
 assert len(row["direct"])==4 and len(row["prefix"])==2 and len(row["observation_checks"])==8
 assert all(v["next_state_error_without_time"]>1e-5 for v in row["direct"])
 assert all(v["target_state_error_without_time"]>1e-5 for v in row["prefix"])
 table.append(dict(demo=row["demo_id"],sample=row["sample_id"],target=row["target"],
   direct_best=min(v["next_state_error_without_time"] for v in row["direct"]),
   prefix_best=min(v["target_state_error_without_time"] for v in row["prefix"]),
   direct_repeat_worst=max(v.get("repeat_state_error",0) for v in row["direct"])))
 obs_rows.append(dict(demo=row["demo_id"],
   joint_max=max(v["obs"]["joint_states"] for v in row["observation_checks"]),
   gripper_max=max(v["obs"]["gripper_states"] for v in row["observation_checks"]),
   agentview_mae_max=max(v["obs"]["agentview_rgb"]["mae"] for v in row["observation_checks"]),
   eye_mae_max=max(v["obs"]["eye_in_hand_rgb"]["mae"] for v in row["observation_checks"]),
   state_load_max=max(v["state_error"] for v in row["observation_checks"])))
align=js(E/"workspace/ALIGNMENT_DIAGNOSIS.json")
static=js(E/"workspace/STATIC_AUDIT.json")
now=datetime.datetime.now().astimezone().isoformat()
write(R/"LOCAL_ACCEPTANCE.json",dict(status="PASS_FOR_BLOCKED_REPORT_ONLY",time=now,input_manifest_verified=True,
 evidence_file_count=len(bundle),counts=dict(counts),rows=table,observation_rows=obs_rows,
 policy_continuations=0,optimizer_calls=0,not_a_completed_54_episode_screen=True))
write(R/"RUN_STATUS.json",dict(status="BLOCKED_STATE_RESTORATION",time=now,stage="A",
 engineering_steps=316,policy_continuations=0,authorized_max_continuations=54,training_steps=0,
 next_action="STOP. State restoration and action/observation timing require a separately verified recovery implementation before any counterfactual screening. No B/C."))
lines=["# RCTR阶段A：状态恢复准入未通过，已停止","",
 "已执行用户批准的阶段A前置检验。**状态：BLOCKED_STATE_RESTORATION**。没有启动54条策略续跑，没有训练、优化器调用或阶段B/C。该结果否定的是当前恢复链路的可用性，不是RCTR算法的有效性。",
 "", "## 实际执行与退出","",
 "- `python admit.py`：退出0；从原RA有序成员中固定3个唯一、非padding、中段窗口。全过程未用loss/成功率挑样。",
 "- `bash run_probe.sh` → `python state_probe.py`：退出1。历史XML资产前缀`chiliocosm/assets`不存在，首次环境构建后、任何step前停止；原日志与退出码保留。",
 "- `bash run_recovery.sh` → `python recover_asset_paths.py`：退出0。仅用外置适配器重定位已有资产路径，原探针和物理XML数值未改；执行完整固定状态检验。",
 "- `python source_audit.py`、`python diagnose_alignment.py`、`python finalize_remote.py`：退出0，只读诊断和独立退出/输入核验。",
 "- `python audit_local.py`：本地独立复算。2次环境构建/关闭、42次reset、316次工程step，1次已留证资产路径异常；所有step有开始/完成账目，无遗留worker。",
 "- 316步 = 3窗口×2动作索引约定×2重复的12步直接检验 + 2×(46+56+50)=304步专家前缀检验。工程step不是316条episode。",
 "", "## 三个窗口均未复现记录中的物理转移","",
 "| 演示 | RA窗口 | 目标t | 直接一步最佳误差 | 前缀目标最佳误差 | 同起点直接重复最大误差 |",
 "|---|---|---:|---:|---:|---:|"]
for r in table:lines.append("| {demo} | {sample} | {target} | {direct_best:.8g} | {prefix_best:.8g} | {direct_repeat_worst:.8g} |".format(**r))
lines += ["","误差为flat MuJoCo状态去除绝对时间后的最大坐标绝对差，包含不同物理量，**不是米、角度或成功率**。事前阈值1e-5。两种固定动作索引约定均已检查；表中取最佳值仅用于保守描述，即使最佳也未通过，未据此改协议。",
 "", "## 状态能装载，不代表观察和后续控制可恢复","",
 "| 演示 | flat状态装载误差 | 8帧关节观察最大误差 | 夹爪观察最大误差 | agentview图像MAE最大值 | eye图像MAE最大值 |",
 "|---|---:|---:|---:|---:|---:|"]
for r in obs_rows:lines.append("| {demo} | {state_load_max:.8g} | {joint_max:.8g} | {gripper_max:.8g} | {agentview_mae_max:.8g} | {eye_mae_max:.8g} |".format(**r))
lines += ["","图像MAE单位为0–255像素值。上下翻转误差也已留存作诊断，未擅自换映射以宣告通过。",
 "", "只读HDF5索引检查（整个演示内部，去除边界两步）：","",
 "| 演示 | 最佳观察索引偏移 | 最佳关节最大误差 | 同索引关节最大误差 |",
 "|---|---:|---:|---:|"]
for r in align["rows"]:
 best=min(r["shift_checks"],key=lambda x:x["max_joint_error"]);same=next(v for v in r["shift_checks"] if v["observation_index_minus_state_index"]==0)
 lines.append("| %s | %d | %.8g | %.8g |"%(r["demo_id"],best["observation_index_minus_state_index"],best["max_joint_error"],same["max_joint_error"]))
lines += ["",
 "三条演示均发现：`states[i]`中的七个关节位置与`obs/joint_states[i-1]`逐元素完全相等；同索引则不等。因此关节观察存在明确的一步索引差，而不是单纯渲染噪声。这里只证实关节字段关系，尚不能据此宣称全部观察/控制器状态同步或推定完整动作索引。±2步是只读诊断，没有重新标记原训练样本。",
 "",
 "代码中的`set_state`只写MuJoCo flat state，`set_init_state`再forward并刷新observables；并未从HDF5恢复控制器目标、夹爪累积命令、observable缓存/采样时钟和求解器warmstart等全量状态。它们是可能的差异来源，尚未被本轮独立因果分解。当前MuJoCo/robosuite版本为%s/%s；数据未提供可用于证明采集环境完全相同的完整版本清单，不能把差异直接归因于版本。"%(
 static["versions"]["mujoco"],static["versions"]["robosuite"]),
 "",
 "前缀重放也不匹配，因此本轮没有采用“装载后大致相似”作为通过标准。尚未进入完整snapshot恢复、GMM参数一致性、正常/干预状态对、5分量候选或成功/失败认证；没有可发布的恢复成功率，也没有跨演示泛化结论。",
 "",
 "## 决策与保护","",
 "按照用户确认的“恢复不一致就停止”执行。下一项真正需要解决的是受控的状态采集/恢复及动作—观察时序，而不是追加候选次数、降低认证门槛或直接训练RCTR。若继续，应另行固定一个只验证恢复接口的工程方案；在它通过前不投入54条分支预算。",
 "",
 "所有冻结输入/原RA绑定/选样清单/原窗口/HDF5哈希前后相同；父模型没有被加载或修改，未生成latent、sidecar动作监督或训练checkpoint。轻量证据仅含哈希、配置、统计、路径与工程账目。旧实验保持只读。",
 "",
 "远端产物：`/data/mlr_ifa/experiments/RCTR_GOAL_seed42_stageA_20261020`；源码：`/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020`。本报告及复算脚本按长期授权推送MLR，不发布数据或权重。",
 "", "审计时间："+now]
(R/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
(R/"BLOCKED.md").write_text("# BLOCKED_STATE_RESTORATION\n\n"+lines[2]+"\n\n见REPORT.md和evidence/output/recovery_asset_paths/STATE_PROBE.json。不得自动启动分支筛查或B/C。\n",encoding="utf-8")
write(R/"DOWNLOAD_MANIFEST.json",{k:dict(sha256=v["sha256"],bytes=v["bytes"]) for k,v in bundle.items()})
write(R/"COMMANDS.json",dict(environment="/root/mlr_ifa/experiment_workspaces/RA_Coreset_GOAL_seed42_20261018/env.sh",
 remote_workspace="/data/mlr_ifa/experiment_workspaces/RCTR_GOAL_seed42_stageA_20261020",
 commands=[
  dict(command="/data/mlr_ifa/env/bin/python admit.py",exit=0),
  dict(command="nohup bash run_probe.sh; run_probe.sh sources parent env.sh and executes python state_probe.py",exit=1),
  dict(command="nohup bash run_recovery.sh; run_recovery.sh sources parent env.sh and executes python recover_asset_paths.py",exit=0),
  dict(command="source parent/env.sh; python source_audit.py",exit=0),
  dict(command="/data/mlr_ifa/env/bin/python diagnose_alignment.py && /data/mlr_ifa/env/bin/python finalize_remote.py",exit=0),
  dict(command="python execution/20261020_rctr_goal_seed42_stageA/audit_local.py",exit=0)],
 original_failed_attempt_preserved=True,policy_rollouts=0,optimizer_calls=0))
print(json.dumps(dict(status="PASS_FOR_BLOCKED_REPORT_ONLY",rows=table,observation_rows=obs_rows),indent=2))
