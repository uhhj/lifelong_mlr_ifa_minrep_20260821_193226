"""Independent local consistency audit of a downloaded observation, not GPU acceptance."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent
def js(p):return json.loads(p.read_text(encoding="utf-8"))
v=js(R/"LATEST_PROGRESS.json");D=Path(v["directory"])
if "input_checks" in v:assert all(x["status"]=="PASS" for x in v["input_checks"].values())
for n,h in js(D/"MANIFEST.json").items():assert hashlib.sha256((D/n).read_bytes()).hexdigest()==h
old=js(D/"RUN_STATUS.json");assert old["status"]=="BLOCKED"
assert old["counts"]==dict(prefix_steps=135,engineering_steps=48,screening_steps=0,optimizer_calls=0,completed_branches=0)
oldrows=[json.loads(x) for x in (D/"attempts.jsonl").read_text().splitlines()]
assert sum(r["event"]=="branch_start" for r in oldrows)==1
assert not any(r["event"]=="branch_complete" for r in oldrows)
E=D/"engineering_recovery/dedicated_env_v1"
gate=js(E/"ENGINEERING_GATE.json");inventory=js(E/"SNAPSHOT_INVENTORY.json")
assert js(E/"MODEL_ADMISSION.json")["status"]=="PASS"
assert gate["status"]=="PASS" and len(gate["states"])==3
assert all(r["recollected_identity"]=="EXACT_MATCH_ORIGINAL" for r in inventory)
oldinv={r["initial_index"]:r for r in js(D/"SNAPSHOT_INVENTORY.json")}
for r in inventory:
 for k in ("physical_sha256","observation_hashes","prefix_noise_sha256","initial_sha256"):
  assert r[k]==oldinv[r["initial_index"]][k]
for state in gate["states"]:
 assert len(state["repeat_checks"])==8 and len(state["native_interface"])==3
 for check in state["repeat_checks"]:
  assert check["physical"]<=1e-10 and check["observations"]<=1e-10
  assert check["gmm"]<=1e-5 and check["action"]<=1e-5
rows=[json.loads(x) for x in (E/"episodes.jsonl").read_text().splitlines()] if (E/"episodes.jsonl").exists() else []
assert len({(r["initial_index"],r["repeat"],r["candidate"]["id"]) for r in rows})==len(rows)<=333
assert all(0<=r["steps"]<=600 for r in rows)
for r in rows:
 if r["candidate"]["axis"] is not None and r["status"]=="COMMITTED":assert r["intervention_l2"]>1e-7
complete=False
if v["recovery_exit"]==0 and v["supervisor_exit"]==0 and not v["workers"]:
 from analyze import analyze
 result=analyze(E)
 assert result["branches"]==333
 assert js(E/"INPUT_PROTECTION.json")["status"]=="PASS"
 complete=True
out=dict(status="COMPLETED_EXPLORATORY_SCREENING" if complete else "PARTIAL",
 observed_unix=v["unix"],local_manifest="PASS",recollected_identities="PASS",engineering_repeats="PASS",
 original_failure_preserved=True,original_failed_attempts=1,completed_screening_rows=len(rows),
 original_engineering_control_steps=183,recovery_engineering_control_steps=183,
 original_incomplete_first_env_step="May include internal physics substeps; not a completed control step",
 committed_screening_steps=sum(r["steps"] for r in rows),optimizer_calls=0,
 future_training_authorized=False,eta_seconds=v.get("eta_seconds"))
(R/"LOCAL_ACCEPTANCE.json").write_text(json.dumps(out,indent=2)+"\n")
(R/"RUN_STATUS.json").write_text(json.dumps(out,indent=2)+"\n")
when=datetime.datetime.fromtimestamp(v["unix"]).astimezone().isoformat()
text=f"""# RCTR Stage A2 当前状态

状态：**{out['status']}**；观察时间：{when}。

- 用户批准：333个固定计划分支；另行批准首次工程失败后的135采集步+48恢复检查步，最多334次分支尝试。
- 当前已提交：{len(rows)}/333条。尚在运行时不能用此快照声称最终实验通过。
- 原始流程退出1：三个即时重复检查通过，但后续环境硬重置销毁旧模拟器，第一条分支中止；0完整结果。失败日志和合同保留。
- 独立恢复：三个环境各自保留；重新采集的初态、噪声、物理状态与观察哈希逐项等于原采集。所有环境建立后重复检查通过；原生GMM 1/4/8历史接口通过。
- 已投入工程控制步：原183 + 恢复183 =366。另有首次未完成env.step，可能已执行内部物理子步，单独留账。采集中的父策略步属于新增辅助交互，不是原基线重评。
- 已提交分支控制步：{out['committed_screening_steps']}；训练/optimizer：0。
- 20/21/22号初态是探索性复用；固定第40步取状态，不是历史演示精确重建。每次普通动作保持argmax分量后Normal采样，匹配独立噪声用于本轮反事实比较。六方向干预和一次分量均值替换是显式实验因素。
- 所有候选及负结果将完整报告。科学信号仅为描述性筛查，不代表显著性或跨状态泛化；本轮不自动进入训练或独立认证。

证据目录：`{D.name}`。输入hash/原失败/尝试账目见该目录；修复完整diff为`RECOVERY_DIFF.patch`。旧Stage A保持BLOCKED。
"""
if v.get("eta_seconds") is not None:text+=f"\n按最近最多30条续跑实测速度外推剩余约{v['eta_seconds']/3600:.2f}小时；不同状态成功终止时间不同，预计会变化。\n"
(R/"REPORT.md").write_text(text,encoding="utf-8")
print(json.dumps(out,indent=2))
