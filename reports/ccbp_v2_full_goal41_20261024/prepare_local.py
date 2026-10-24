"""Predeployment record and reviewed source bundle; not a deployment command."""
from pathlib import Path
import json,hashlib,ast,datetime
R=Path(__file__).resolve().parent;P=R.parent/'20261024_ccbp_v2'
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
snapshot=P/'check_readonly_20261024_234643';d=json.loads((snapshot/'EVIDENCE.json').read_text(encoding='utf8'));audit=json.loads((snapshot/'LOCAL_AUDIT.json').read_text(encoding='utf8'))
old=json.loads((P/'RUN_STATUS.json').read_text(encoding='utf8'));old.update(status='SUPERSEDED_BY_USER_FULL_SEQUENCE_ENGINEERING_FAILURE_PRESERVED',observation_unix=d['observed'],latest_evidence=snapshot.name,counts=audit['counts'],formal_updates=3033,formal_episodes=0,phase='EXIT1_NO_OWN_PROCESSES',next_check_minutes=None,eta=None,automation_deleted=True,replacement_directory=str(R),failure='Gradient diagnostic asserted all p.grad None; existing zero gradient tensors require exact before/after comparison. Old frozen source retained, no restart.',private_recovery_retired_by_explicit_user_authorization=True)
write(P/'RUN_STATUS.json',old);active=json.loads((P/'ACTIVE_ROOT.json').read_text());active.update(automation_id=None,status=old['status']);write(P/'ACTIVE_ROOT.json',active)
notice='# 用户替换为完整序列：本分叉已退出且不恢复\n\n实际只读快照23:46:54核controller.exit1、GPU空闲、无自身进程、32冻结项不变。B保存9完整epoch3033更新，其他0，正式环境0。epoch9诊断断言失败，不能当科学负结果；有状态梯度准入需新增验证。旧监控ccbp-v2-goal41已删除。用户随后要求四组各自从官方初始化跑完整GOAL，并明确授权自动清理废用数据；本分叉恢复权重已按精确SHA清理，日志/源码/失败与哈希保留，不能再声称可恢复此废止分叉。新目录20261024_ccbp_v2_full。\n\n以下历史启动观察保留，不代表现状。\n\n'
(P/'REPORT.md').write_text(notice+(P/'REPORT.md').read_text(encoding='utf8'),encoding='utf8');(P/'MONITOR_HANDOFF.md').write_text(notice,encoding='utf8')
pre=P/'full_sequence_preflight_readonly_20261024_234821/EVIDENCE.json';clean=P/'cleanup_authorized_obsolete_20261024_235135/EVIDENCE.json'
for src,name in [(pre,'SOURCE_AND_CAPACITY_PREFLIGHT.json'),(clean,'AUTHORIZED_CLEANUP_RECEIPT.json')]:
 data=json.loads(src.read_text(encoding='utf8'));data.pop('sources',None);write(R/name,data)
files=['runtime.py','layout.py','losses.py','legacy_losses.py','prototype_loss.py','synthetic_checks.py','updates.py','train.py','evaluate.py','controller.py','PROTOCOL.md','USER_METHOD.txt']
bundle={n:(R/n).read_text(encoding='utf8') for n in files}
for n,s in bundle.items():
 if n.endswith('.py'):ast.parse(s)
write(R/'DEPLOY_TEXT_BUNDLE.json',bundle)
write(R/'STATIC_REVIEW.json',dict(status='AST_ONLY_NOT_GPU_ADMISSION',sources={n:hashlib.sha256(s.encode()).hexdigest() for n,s in bundle.items()},grad_diagnostic_fix='Compare actual preexisting gradients and RNG, test at state after prior update',old_failure_bound=str(snapshot/'LOCAL_AUDIT.json')))
write(R/'RUN_STATUS.json',dict(status='PREREGISTERED_NOT_DEPLOYED',observation=datetime.datetime.now().astimezone().isoformat(),scope='Full independent GOAL Task6-9 from official initialization',epochs=1600,updates=304400,episodes=800,own_admission='PENDING',formal_updates=0,formal_episodes=0,automatic_expansion=False))
(R/'REPORT.md').write_text('# CCBP-v2 GOAL seed41 完整四组\n\n当前预登记，未部署、未正式训练。B/C/N/V各自官方初始化，Task6–9各100轮，每组76100主更新，共304400。闭环四组全三角矩阵800回合。原Task9分叉已废止并保留工程失败。\n\n用户授权清理三份旧中间恢复，精确释放1079091200分配字节，旧最终模型/恢复/原始数据保留。新空间按root1.07GiB/data1.88GiB峰值并保留1/2GiB预检，不减精度预算。静态检查不等于自身GPU准入。\n\n预计训练约3–4小时量级、800回合约数小时；新全序列吞吐未测，正式启动后以当前整组预计结束点更新监控，不能套此前单Task9时间。\n',encoding='utf8')
print('Prepared source and new protocol; old queue superseded, no remote mutation here')
