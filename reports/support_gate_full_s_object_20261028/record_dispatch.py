"""Local evidence-only recording of the user-authorized queue replacement."""
from pathlib import Path
import json,hashlib,datetime,ast
R=Path(__file__).resolve().parent;M=R.parent/'20261027_support_gate_forks'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def put(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
stopdir=M/'deploy_remote_20261028_193702';stop=js(stopdir/'EVIDENCE.json')
assert stop['status']=='USER_STOPPED_PARTIAL_PRESERVED_NOT_SCIENTIFIC_ACCEPTANCE' and not stop['processes_after'] and stop['deleted_files']==0
assert stop['epochs']['OBJECT_41/EPOCHS_S.json']==100 and stop['epochs']['OBJECT_41/EPOCHS_Q.json']==3
tot=acc=phys=0;counts={}
for n,e in stop['epochs'].items():
 case,file=n.split('/');arm=file.split('_')[1].split('.')[0];batch=362 if case=='OBJECT_41' else 185
 counts.setdefault(case,{})[arm]=dict(epochs=e,accepted_updates=e*batch,physical_optimizer_calls=e*batch*(2 if arm=='G' else 1))
 tot+=e;acc+=e*batch;phys+=e*batch*(2 if arm=='G' else 1)
assert (tot,acc,phys)==(1403,295486,350986)
stamp=datetime.datetime.fromtimestamp(stop['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
audit=dict(status='PASS_USER_STOP_PRESERVATION_NOT_SCIENTIFIC_ACCEPTANCE',observation=stamp,counts=counts,epochs=tot,accepted_updates=acc,recorded_complete_epoch_physical_calls=phys,partial_unlogged_cost='UNKNOWN',frozen_files=56,processes=[],deleted_files=0,raw_stop_sha256=sha(stopdir/'EVIDENCE.json'))
put(stopdir/'LOCAL_STOP_AUDIT.json',audit)
prior=js(M/'RUN_STATUS.json');put(M/'USER_STOP_PREVIOUS_STATUS.json',prior)
prior.update(status='USER_STOPPED_PARTIAL_PRESERVED',observation=stamp,counts=counts,epochs=tot,accepted_updates=acc,physical_training_optimizer_calls=phys,processes=[],controller=None,latest_evidence=stopdir.name,mechanism_remaining_queue='CANCELED_BY_USER',mechanism_formal_evaluation='CANCELED_BY_USER',unlogged_partial_epoch_cost='UNKNOWN',followup=dict(root='/data/mlr_ifa/experiment_workspaces/SUPPORT_GATE_FULL_S_OBJECT41_42_43_20261028_v1',directory=str(R),mechanism_acceptance_prerequisite=False))
put(M/'RUN_STATUS.json',prior)
error=M/'deploy_remote_20261028_195051'/'ATTEMPT.json';err=js(error)
assert err['exit']==1 and 'FileExistsError' in err['stderr']
put(R/'LOCAL_DISPATCH_ERROR_AUDIT.json',dict(status='GUARD_STOPPED_REDUNDANT_STOP_NO_SIGNAL_OR_TRAINING',cause='Relative local Copy-Item paths were resolved from transport directory; copies failed, leaving old stop wrapper. Shell continued to queue it.',actual_attempt=str(error),attempt_sha256=sha(error),authenticated=err['authenticated'],remote_commands=err['remote_commands'],guard='user_stop directory mkdir(exist_ok=False) precedes REQUEST writes and all kill actions',new_training_calls=0,process_signals=0,deleted_files=0,correction='Used absolute paths and ErrorActionPreference Stop; unique new dispatch request 20261028_195120_6e9d3cdf.'))
dep=js(R/'deploy_remote_20261028_195120'/'EVIDENCE.json')
assert dep['status']=='DISPATCHED_PENDING_OWN_GPU_ADMISSION' and dep['controller']==3013 and dep['frozen_files']==63
review=js(R/'UPLOAD_REVIEW.json');bundle=js(R/'DEPLOY_TEXT_BUNDLE.json')
for n,s in bundle.items():
 assert hashlib.sha256(s.encode()).hexdigest()==review['files'][n]['sha256']
 if n.endswith('.py'):ast.parse(s,feature_version=(3,8))
put(R/'DEPLOYMENT_LOCAL_AUDIT.json',dict(status='PASS_UNIQUE_DISPATCH_PENDING_FINAL',root=dep['root'],controller=dep['controller'],contract=dep['contract'],children=dep['children'],source_files=dep['frozen_files'],bundle_text_files=len(bundle),deployment_evidence_sha256=sha(R/'deploy_remote_20261028_195120'/'EVIDENCE.json'),mechanism_stop=audit,static_python38=True,private_payloads_uploaded=False))
(R/'IMPLEMENTATION_NOTES.md').write_text('''# 完整 S 实现与来源

build_local.py 在新目录组合已验收原完整 S OBJECT seed41/42/43 与已运行机制门控实现。runtime、初始化、采集、评估、旧缓存读取、losses、synthetic_checks 对各 seed 原文件文本精确不变。逐文件原SHA/新SHA及diff位于UPLOAD_REVIEW和review。三seed各自PARENT_BINDING来自本轮真实只读preflight，不能互套数据mask。

support_memory读取本轮自己的成功采集缓存并派生q，不使用机制旧BINDING。fork_tools的stage由真实Dataset提供；gate_admission每stage指定自己的stage。Task6按原完整S的4更新准入；Task7–9每stage真实29更新准入，完整状态恢复后才训练。新train仅改变支持加权/双候选选择及其记账，阶段采集和正式评估未改变。

每seed195300实际训练调用=105500接受+89800额外C候选。每seed门控9100块（7/8/9各2300/3100/3700），三seed27300块。接受Q时才计其4槽/步；拒绝候选的算力计入physical但其损失不计入接受轨迹totals。每epoch保存完整state，stage9硬链接最后恢复文件。新准入全部273调用/123额外autograd，方向诊断最多108autograd，环境采集120、正式600。不得把内部门控C/Q误当独立重跑机制组。

仅外置监控/验收包装可以版本化修复，不修改63个冻结项；最终独立CPU包装尚须从原完整S适配，本轮只有静态及实际GPU首阶段准入，不能宣称CPU最终通过。final_readonly只可在自然controller.exit0且无自身进程后执行。预检和部署使用现有credential worker，秘密仅内存；新目录不含会话目录。
''',encoding='utf8')
print(json.dumps(dict(status='RECORDED_USER_STOP_AND_NEW_DISPATCH',mechanism_epochs=tot,controller=dep['controller'])))
