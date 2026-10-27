"""Record already-observed launch/admission metadata; never launch remote work."""
from pathlib import Path
import json,datetime,hashlib
R=Path(__file__).resolve().parent;js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
D=R/'check_readonly_20261027_133342';a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');d=js(R/'deploy_remote_20261027_133059/EVIDENCE.json');capdir=R/'capacity_readonly_20261027_133219';cap=js(capdir/'EVIDENCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['admissions']=={'GOAL_42_6':4} and not a['failure'] and v['gpu'].strip()=='2565'
assert d['contract']==a['contract'] and d['controller']==2559 and cap['status']=='PASS_NOT_FINAL'
for x in cap['capacity'].values():assert x['free']-x['remaining']-x['reserve']==x['headroom']>=0
(capdir/'LOCAL_AUDIT.json').write_text(json.dumps(dict(status='PASS_NOT_FINAL',source_sha256=sha(capdir/'EVIDENCE.json'),capacity=cap['capacity'],no_payload_model_environment_or_write=True),indent=2),encoding='utf8')
obs=datetime.datetime.fromtimestamp(a['observation'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
old=js(R.parent/'20261025_ccbp_v3_full/final_readonly_20261025_181226/light/EPOCHS_C.json')
remaining6=(100-a['counts']['GOAL_42']['by_stage']['6'])*a['rates_last10']['GOAL_42']['6']/60
reference=sum(x['seconds'] for x in old if x['stage']>6)/60
status=dict(status='PARTIAL_TRAINING',observation=obs,latest_evidence=D.name,counts=a['counts'],formal_episodes=0,own_admissions=a['admissions'],root=d['root'],controller=d['controller'],contract=d['contract'],hardware=d['hardware'],next_check_minutes=65,eta=dict(task6_remaining_measured_minutes=remaining6,aux_stages_reference_minutes=reference,reference_hardware='historical v3 C GOAL41 4080S; current 3090',reference_total_minutes=remaining6+reference,preparation_saving_extra=True,whole_queue_eta='not measured'),final_acceptance='PENDING_REAL_CPU_AND_COMPLETE_FORMAL_EVALUATION')
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf8')
(R/'ACTIVE_ROOT.json').write_text(json.dumps(dict(status=status['status'],root=d['root'],controller=d['controller'],contract=d['contract'],children=d['children'],frozen_files=d['frozen_files'],observed=obs,latest_evidence=D.name),indent=2),encoding='utf8')
report=f'''# C完整GOAL42/43与OBJECT41/42/43：进行中

状态PARTIAL_TRAINING。真实观察{obs}，{D.name}/LOCAL_AUDIT PASS_PROGRESS_NOT_FINAL。

五组有限队列已一次启动，controller2559/GPU2565正常；68冻结项及父源一致。GOAL42 Task6完整29epoch2900主步，自己的4隔离GPU更新、完整下一步恢复和原ALIGN更新精确/1–8长度/100键21633611参数实际PASS。其余四run未自身准入，正式0/1000；不套旧PASS、不计算部分收益。

五组各从本suite/seed官方初始化，GOAL42→GOAL43→OBJECT41→OBJECT42→OBJECT43全部训练后按同顺序评估。总2000epoch468700主更新，0训练采集，1000正式；C损失/路由/更新代码与已验收原实现逐字一致，GOAL train源码未改，OBJECT只调整总步数断言。新路径、ARMS=C、各seed人口/mask、SEED派生噪声为受审查适配。没有加入S保持项。

首段Task6最近10轮均{a['rates_last10']['GOAL_42']['6']:.6f}秒，余71轮约{remaining6:.6f}min。Task7–9自身未测，借历史v3 C GOAL41对应{reference:.6f}min作不确定参考（旧4080S、当前3090），剩余纯计算参考{remaining6+reference:.6f}min；保存/准备/教师另计。下一65分钟检查GOAL42整组结束附近，不是五组全队列或1000正式完成预测。

预检五父输入身份/RA成员/ALIGN mask和载荷SHA实际通过；system普通留1GiB及data留2GiB、扣剩余6GiB峰值后均正，最近余量system {cap['capacity']['/root']['headroom']} B、data {cap['capacity']['/data']['headroom']} B。无需删除、付费或改预算。

本地首次build以CRLF文件字节与远端LF比较而拒绝；逐文件标准化换行后文本SHA全部精确一致，修正仅本地派生器，之前0远程部署/更新/环境。原断言事实保留在STATIC_REVIEW。27MB只读预检原始元数据本地保留；发布按case拆分的轻量身份/hash绑定及原文件SHA，不发布重复大聚合或部署bundle。

全程最终外置final_readonly→audit_final_local→describe_final目前只通过AST，未提前运行。完成须实际20CPU strict CP、5完整恢复、15教师cache、1000初态真实字节/noise和全部标量独立验收，再完成机制分析和三seed比较。发布进度不等于接受。
'''
(R/'REPORT.md').write_text(report,encoding='utf8')
(R/'IMPLEMENTATION_NOTES.md').write_text('''# 派生实现边界

科学父本为已验收20261025_ccbp_v3_full engineering02 C；其冻结服务端9源文件全部和本地归一LF文本SHA一致。保留losses/legacy_losses/updates及其未启用R/G辅助定义，ARMS仅C，未派发任何R/G。OBJECT和GOAL各父manifest允许合法mask数量随seed改变。

服务端部署一次，实际Python3.8 import与原合成数值检查五case通过后才controller启动。合成检查不是自己的GPU更新准入；只有GOAL42 Task6首4隔离更新当前PASS。最终外置检查不能在活跃训练/评估运行，不得重跑build_extension、prepare_local、deploy_remote或controller。

凭据worker26908通过用户已授权Windows Credential Manager取入内存，.session_control不发布。来源和容量只读预检未增加科学环境。源码冻结后只允许外置检查器留证修复；不得借修复改变目标、seed、预算或重试正常负结果。
''',encoding='utf8')
print(json.dumps(status,indent=2))
