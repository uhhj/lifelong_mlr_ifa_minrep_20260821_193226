"""Record actual startup only after read-only observation and own admission."""
from pathlib import Path
import json,sys,hashlib,datetime
D=Path(__file__).resolve().parent;R=D.parent;S=Path(sys.argv[1])
J=lambda p:json.loads(p.read_text(encoding='utf8'))
e=J(S/'EVIDENCE.json');a=J(S/'LOCAL_AUDIT.json');review=J(D/'REVIEW.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
assert e['contract_sha256']==review['adapted_contract']
L={n:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in e['light'].items()}
assert 'coordinator/FAILURE.json' not in L and 'coordinator/controller.exit' not in L
assert all(not n.endswith('.exit') or str(v).strip()=='0' for n,v in L.items())
assert L['coordinator/STATUS.json']['suite']=='OBJECT'
assert L['coordinator/STATUS.json']['status']=='TRAINING'
assert all(a['counts'][x]['epochs']==0 for x in ['OBJECT_43','GOAL_41','GOAL_42','GOAL_43'])
admissions={}
for case in ['OBJECT_41','OBJECT_42']:
 init=L[case+'/INITIALIZATION_AUDIT.json'];ad=L[case+'/ADMISSION_6.json']
 assert init['status']=='PASS_ACTUAL_INITIALIZATION' and init['initial_exact']
 assert ad['status']=='PASS_OWN_GPU_ALIGN_ADMISSION' and ad['optimizer_calls']==5
 assert all(ad[k] for k in ['disabled_original_random_full_state_exact','valid_mask_nll_exact','empty_mask_differentiable_zero','serialized_full_state_next_step_exact','initial_state_restored'])
 assert a['counts'][case]['epochs']>0
 admissions[case]=dict(initialization=init,task6_admission=ad)
dep=L['coordinator/DEPLOYMENT.json']
assert any(x['pid']==dep['controller'] for x in e['processes'])
obs=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
out=dict(status='PARTIAL_STARTED_OWN_TASK6_ADMISSION_PASS',observed=obs,root='/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261109_v1',contract=e['contract_sha256'],controller=dep['controller'],counts=a['counts'],epochs=a['epochs'],production_updates=a['production_updates'],formal_episodes=a['formal_episodes'],own_admissions=admissions,snapshot=str(S),evidence_sha256=hashlib.sha256((S/'EVIDENCE.json').read_bytes()).hexdigest(),processes=e['processes'],q1_publication=dep['previous_q1_publication'],remaining_stages='NOT_YET_ADMITTED',queue='OBJECT three train then600formal, GOAL three train then600formal; at most2trainers; existing baselines read-only')
(S/'STARTUP_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for n in ['ACTIVE_ROOT.json','RUN_STATUS.json']:(R/n).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
header=f'''## {obs} ALIGN六run队列已唯一启动

q1两套件已实际独立接受、完整比较及FINAL_REPORT发布，启动门槛已核远端提交{dep['previous_q1_publication']}。唯一engineering_completion_gate_20261110_v2/launch_after_q1.py部署成功，controller{dep['controller']}，合同{e['contract_sha256']}。仅runtime的旧q1完成源适配，所有训练评估代码/输入保持原准备清单。禁止重复任何launch/deploy/controller或完成轮。

实际快照{S.name}进度审计及STARTUP_AUDIT通过。OBJECT41/42各自初始化tensor/RNG与Random父源精确，Task6自己的5次隔离ALIGN准入通过；已完成轮分别{a['counts']['OBJECT_41']['epochs']}/{a['counts']['OBJECT_42']['epochs']}，累计{a['production_updates']}生产更新、10隔离、0训练采集、0正式；未来Task7–9准入不得套PASS。其余四run未开始。既有q1与所有S只读。

此后只用engineering_completion_gate_20261110_v2/transport.py check_readonly.py获取实际新合同快照，再用基础目录audit_progress.py NEW PREVIOUS。最终用该工程目录final_readonly.py（已适配新合同）和基础目录audit_final_local.py/compare_baseline.py；仍须真实controller.exit0且无自身进程。没有定时监控或自动聊天验收服务，远端有限controller会按OBJECT三训练/600正式→GOAL三训练/600正式自然运行；失败停止新增运行并保留既有结果。

'''
p=R/'MONITOR_HANDOFF.md';p.write_text(header+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps({k:v for k,v in out.items() if k not in ['processes','own_admissions']},ensure_ascii=False))
