"""Persist the user's deferred GOAL queue and bind existing accepted baseline."""
from pathlib import Path
import json,hashlib,datetime,tomllib
R=Path(__file__).resolve().parent.parent;G=R.parent/'20261029_support_gate_full_s_goal';A=Path(__file__).resolve().parent
js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=js(G/'FINAL_ACCEPTANCE_RECEIPT.json');D=G/'final_readonly_20261030_111546';audit=js(D/'LOCAL_FINAL_AUDIT.json')
assert receipt['status']=='COMPLETE_ACCEPTED_INTERPRETED' and audit['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and audit['formal_episodes']==600
for name in ['EVIDENCE.json','LOCAL_FINAL_AUDIT.json']:
 assert sha(D/name)==receipt['files']['final_readonly_20261030_111546/'+name]
assert audit['evidence_sha256']==sha(D/'EVIDENCE.json')
source=dict(status='ACCEPTED_GOAL_010_SOURCE_HASHES_VERIFIED',final_path=str(D),evidence_sha256=sha(D/'EVIDENCE.json'),audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),formal_episodes=600,production_optimizer_calls=audit['physical_training_optimizer_calls'],accepted_updates=audit['main_updates'],own_isolated_calls=audit['isolated_optimizer_updates'],parent_bindings={str(s):sha(G/f'seed{s}/PARENT_BINDING.json') for s in (41,42,43)})
(A/'BASELINE_BINDING.json').write_text(json.dumps(source,indent=2)+'\n',encoding='utf8')
queue=dict(status='AUTHORIZED_WAITING_OBJECT_COMPLETION',authorized_at=datetime.datetime.now().astimezone().isoformat(),user_instruction='OBject跑完后 把goal一起跑了 同样的协议 同样的训练方式',suite='GOAL',lambdas=[.05,.2],seeds=[41,42,43],new_runs=6,baseline_lambda=.1,baseline_read_only=True,max_training_concurrency=2,environment_concurrency=1,protocol=str((A/'PROTOCOL.md').relative_to(R)),baseline_binding=str((A/'BASELINE_BINDING.json').relative_to(R)),planned_directory='execution/20261103_lambda_r_goal_sensitivity',candidate_remote_root='/data/mlr_ifa/experiment_workspaces/S_LAMBDA_R_GOAL41_42_43_20261103_v1',controller=None,contract=None,build='NOT_YET_BUILT',deployment='NOT_DEPLOYED',start_requires=['OBJECT actual natural completion','OBJECT final acceptance and interpretation publication','OBJECT owned hook cleanup under SHA/no-target-process checks','GOAL adapted code and frozen contract reviewed','GOAL capacity and own admission prerequisites passed'],budget=dict(epochs=2400,accepted_updates=456600,production_optimizer_calls=853200,gate_blocks=40200,isolated_optimizer_calls=546,admission_autograd=246,max_direction_autograd=216,training_collection_episodes=240,formal_episodes=1200),monitor='object-s-r-run',close_monitor_only_after='GOAL six runs independently accepted, interpreted, published and owned hook cleanup; OBJECT completion is not final user objective')
(R/'AUTHORIZED_GOAL_FOLLOWUP.json').write_text(json.dumps(queue,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
s=js(R/'RUN_STATUS.json');s['authorized_followup']=queue;(R/'RUN_STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
intro='''## 最新用户授权：OBJECT结束后继续GOAL同协议六run

用户明确“OBject跑完后 把goal一起跑了 同样的协议 同样的训练方式”。详见AUTHORIZED_GOAL_FOLLOWUP.json及goal_followup_20261103/PROTOCOL.md、BASELINE_BINDING.json。只新增GOAL最新S .05/.2各seed41/42/43，已有GOAL .1的600正式已核接受源SHA，只读复用。每阶段100轮、最多两训练并行、采集/正式互斥和优先.05正式/跨组填空槽方式不变。GOAL六run预算456600接受853200生产40200门控、546隔离246准入autograd、最多216方向autograd、240采集1200正式，不能套OBJECT的预算。

当前GOAL仅持久排队协议和基线绑定，尚未构建、冻结或部署；监控必须自主完成后续适配/容量核验/唯一部署和验收。OBJECT六run训练正式结束且独立验收解释发布、自有hook安全清理后才启动GOAL，不与OBJECT重叠。候选独立目录execution/20261103_lambda_r_goal_sensitivity和root S_LAMBDA_R_GOAL41_42_43_20261103_v1，部署前核唯一性。若容量不足不删旧模型或付费，保留实际证据通知需行动。GPU优化须核GOAL适用性与每阶段自身准入，不盲套OBJECT逐位证明。硬件迁移、开发集复用/预训练来源限制继续保留。

沿用同一个object-s-r-run监控并在接续时更新活动目录，不新建重复监控。该授权覆盖此前“OBJECT六run完成即删除监控”的结束条件：必须GOAL六run也实际独立接受、解释发布及自有hook清理后才关闭。OBJECT0.05已接受结果保持，0.2继续，不重复任何旧训练评估。

'''
for n in ['MONITOR_HANDOFF.md','REPORT.md']:
 p=R/n;p.write_text(intro+p.read_text(encoding='utf8'),encoding='utf8')
config=tomllib.loads(Path('C:/Users/Lenovo/.codex/automations/object-s-r-run/automation.toml').read_text(encoding='utf8'))
config['name']='OBJECT→GOAL S λ_R 两档实验'
config['prompt']=intro.replace('\n',' ')+config['prompt'].replace('全部六run独立验收解释发布并清理自有hook后才删除同一监控。','OBJECT验收发布清理后继续已授权GOAL六run，GOAL也完成后才删除同一监控。')
(R/'MONITOR_UPDATE_REVIEW.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=queue['status'],baseline=source['status'],build=queue['build'],budget=queue['budget'])))
