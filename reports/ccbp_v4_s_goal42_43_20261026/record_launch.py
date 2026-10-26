import json,hashlib,ast,datetime
from pathlib import Path
R=Path(__file__).resolve().parent;js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
deploy=js(R/'deploy_remote_20261026_124949/EVIDENCE.json');snap=R/'check_readonly_20261026_125214';a=js(snap/'LOCAL_AUDIT.json');cap=R/'capacity_readonly_20261026_125121';c=js(cap/'EVIDENCE.json');pre=R/'preflight_readonly_20261026_124346'
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and c['status']=='PASS_NOT_FINAL' and all(x['headroom']>=0 for x in c['capacity'].values())
for p in R.rglob('*.py'):
 if '.session_control' not in p.parts:ast.parse(p.read_text(encoding='utf8'),feature_version=(3,8))
for s in (42,43):
 for n in ['legacy_losses.py','losses.py','memory.py','updates.py','synthetic_checks.py']:
  assert (R/f'seed{s}'/n).read_text(encoding='utf8')==(R.parent/'20261025_ccbp_v4_full'/n).read_text(encoding='utf8')
for dd in [pre,cap]:
 e=js(dd/'EVIDENCE.json');attempt=js(dd/'ATTEMPT.json');assert attempt['exit']==0
 (dd/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_PREDEPLOY_READONLY' if dd==pre else 'PASS_NOT_FINAL',observed=e['observed'],evidence_sha256=sha(dd/'EVIDENCE.json'),attempt_sha256=sha(dd/'ATTEMPT.json'),capacity=e['capacity'],no_new_optimizer_or_environment=True),indent=2)+'\n',encoding='utf8')
obs=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
state=dict(status='PARTIAL_TRAINING',observation=obs,latest_evidence=snap.name,controller=deploy['controller'],root=deploy['root'],contract=deploy['contract'],seeds={s:dict(counts=x['counts']['S'],admissions=x['admissions']) for s,x in a['seeds'].items()},main_updates=a['total_main_updates'],formal_episodes=a['formal_episodes'],training_collection_episodes=a['training_collection_episodes'],next_check_minutes=115,eta_scope='seed42 whole training including collection; later-stage rate unmeasured; not full queue ETA',capacity_evidence=cap.name,scientific_status='PARTIAL')
(R/'RUN_STATUS.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf8');(R/'ACTIVE_ROOT.json').write_text(json.dumps(deploy,indent=2)+'\n',encoding='utf8')
review=dict(status='PASS_SOURCE_REVIEW_AND_REAL_INITIAL_ADMISSION_NOT_FINAL',source_review_sha256=sha(R/'UPLOAD_REVIEW.json'),protocol_sha256=sha(R/'PROTOCOL.md'),deployment_sha256=sha(R/'deploy_remote_20261026_124949/EVIDENCE.json'),progress_sha256=sha(snap/'LOCAL_AUDIT.json'),unchanged_science_files=['legacy_losses.py','losses.py','memory.py','updates.py','synthetic_checks.py'],Python38_static_parsing=True,actual_remote_imports_passed_before_dispatch=True,actual_own_seed42_GPU_admission=4,short_history_extra_autograd=8,short_history_policy_calls=17,seed43_admission='NOT_YET_RUN',final_wrappers='STATIC_ONLY_NOT_FINAL_ACCEPTANCE',local_invocation_correction=dict(error='One audit invocation used request timestamp 125213; actual transport result directory is 125214.',effect='FileNotFound before audit, zero remote commands/updates/environments; corrected path passed.'))
(R/'LAUNCH_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n',encoding='utf8')
report=f'''# CCBP-v4 S GOAL seed42/43 — PARTIAL

实际观察：{obs}，证据{snap.name}。用户指定的两seed已单次成功部署，controller2033/GPU2039健康，33冻结项及两份父源一致。seed42 Task6完整36epoch/3600主更新；自己的4隔离GPU调用与完整下一步恢复、原ALIGN更新准入PASS，短历史1–8实际保持梯度及8帧旧路径精确PASS（额外8autograd/17策略调用）。seed43尚在有限队列，未套用seed42准入。训练采集0、正式0，不计算部分性能。

每seed独立官方初始化，完整四阶段400epoch76100主步；合计800epoch152200主步、80训练采集、400正式回合。成功筛选、固定获得时教师、最多128/task和512/seed、缺额不补采、C专家监督和margin及保持损失均同S41。两seed自身原RA成员/ALIGNmask人口合法数不同，已分别冻结，不复制seed41人口账目。

预检实际3090，按既有授权只记录硬件差异。seed42 Task6最近10轮2.700276秒，余64轮约2.88分钟；辅助阶段尚未实测，借S41约99.45分钟全epoch计算作参考，加采集/准备/保存，下一115分钟检查seed42整组完成附近（约14:50CST）。这不是两seed与全部仿真的完成预测，下一轮按真实吞吐更新。

容量只stat：system普通保留1GiB后余372883456B，data root扣剩余全队列峰值及2GiB后余45169770496B。无需删除或付费。现有S41及所有基线只读复用，不重训旧实验。

最终外置wrappers已静态解析，尚待自然退出后的真实8CPU CP/2完整恢复/8固定缓存与教师重放、80训练400正式真实初态字节/noise/标量全验收。参考结果已SHA绑定，最终报告三seed均值离散程度、完整矩阵、负单元和历史基线差异；不能将额外交互后的收益都归因于筛选或声称同seedC/O消融。

本地一次审计参数把请求时间当结果目录，FileNotFound在读取证据前发生，已按真实返回目录修正并PASS；没有远程重试/重启/改参。发布进度不等于接受。
'''
(R/'REPORT.md').write_text(report,encoding='utf8')
print(json.dumps(state,indent=2))
