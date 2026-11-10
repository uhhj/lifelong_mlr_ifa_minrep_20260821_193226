"""Record actual stopped evaluation, without relaunch or scientific mutation."""
from pathlib import Path
import json,hashlib,datetime
R=Path(__file__).resolve().parent;V=R/'engineering_startup_20261110_v4'
D=V/'check_readonly_20261110_151130';F=V/'capacity_readonly_20261110_151221'
J=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=J(D/'EVIDENCE.json');a=J(D/'LOCAL_AUDIT.json');f=J(F/'EVIDENCE.json')
L={n:v['text'] for n,v in e['light'].items()}
assert not e['processes'] and not f['processes']
assert L['coordinator/controller.exit'].strip()=='1'
assert L['OBJECT_41/evaluate.exit'].strip()=='1'
assert a['epochs']==1200 and a['production_updates']==316500 and a['formal_episodes']==0
assert all(a['counts']['GOAL_'+str(i)]['epochs']==0 for i in [41,42,43])
for s in [41,42,43]:
 c='OBJECT_'+str(s);assert L[c+'/train.exit'].strip()=='0'
 assert json.loads(L[c+'/random/TRAINING_COMPLETE.json'])['epochs']==400
 for t in range(6,10):
  ad=json.loads(L[c+'/ADMISSION_'+str(t)+'.json'])
  assert ad['status']=='PASS_OWN_GPU_ALIGN_ADMISSION' and ad['optimizer_calls']==5
  assert all(ad[k] for k in ['disabled_original_random_full_state_exact','valid_mask_nll_exact','empty_mask_differentiable_zero','serialized_full_state_next_step_exact','initial_state_restored'])
attempts=[json.loads(x) for x in L['OBJECT_41/random/evaluation/attempts_stage6_task6.jsonl'].splitlines()]
assert [x['event'] for x in attempts]==['environment_construct_start','environment_construct_complete','vector_attempt_start','technical_exception']
obs=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
events=[json.loads(x) for x in L['coordinator/EVENTS.jsonl'].splitlines()]
timefmt=lambda t:datetime.datetime.fromtimestamp(t,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
failure=J(D/'EVIDENCE.json')['light']['coordinator/FAILURE.json']
out=dict(status='PARTIAL_STOPPED_EVALUATION_ASSET_FAILURE',observed=obs,root=J(R/'ACTIVE_ROOT.json')['root'],contract=e['contract_sha256'],controller=None,former_controller=39717,
 counts=a['counts'],epochs=1200,production_updates=316500,formal_episodes=0,own_admission_optimizer=60,
 object_training_complete_time=timefmt(next(x['time'] for x in events if x['kind']=='exit' and x['case']=='OBJECT_43')),
 failure_time=timefmt(json.loads(failure['text'])['observed']),root_exit=1,object41_evaluate_exit=1,
 missing_asset='cases/OBJECT_41/source/libero/libero/assets/scenes/libero_floor_base_style.xml',
 attempt=dict(stage=6,task=6,vector_group=0,initial_indices=list(range(10)),events=[x['event'] for x in attempts],completed_episodes=0,
 note='Construction/first-vector attempt exists. Do not report zero environment attempt cost or retry without an audited recovery.'),
 processes=[],gpu=f['gpu'],evidence_sha256=sha(D/'EVIDENCE.json'),failure_evidence_sha256=sha(F/'EVIDENCE.json'),
 snapshot=str(D),failure_snapshot=str(F),eta='INVALIDATED_PENDING_AUDITED_ENGINEERING_RECOVERY',new_training=0,new_rollouts=0)
(D/'FAILURE_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for n in ['ACTIVE_ROOT.json','RUN_STATUS.json']:(R/n).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
report=f'''# ALIGN进度：OBJECT训练完成，评估工程故障

实际观察{obs}。状态PARTIAL_STOPPED_EVALUATION_ASSET_FAILURE，不能将LOCAL_AUDIT中的预算/前缀PASS_PROGRESS_NOT_FINAL当成运行健康或最终接受。

OBJECT41/42/43均400完整轮、105500生产更新、四阶段各5次隔离ALIGN准入通过，自己的train.exit均0。43于{out['object_training_complete_time']}自然退出。累计1200/2400轮、316500/544800生产、60/120隔离、0训练采集、0/1200完成正式。GOAL三个seed未启动。135548所有已记录epoch/准入前缀保留，冻结合同verify通过。

调度器随即启动OBJECT41评估，于{out['failure_time']}失败并退出1。实际日志FileNotFoundError指向新部署source/libero/libero/assets/scenes/libero_floor_base_style.xml；环境子进程加载场景失败，主进程记录ConnectionResetError。错误属于新目录资产部署遗漏，先前只导入预检未覆盖真实资产加载，不是已定位的GPU故障。当前nvidia-smi正常，4080SUPER利用率0%、显存10MiB，无本轮进程。

原attempt有environment_construct_start/complete、vector_attempt_start（Task6/task6、初态0–9）和technical_exception，0完成回合。异步构造complete日志不代表所有子进程成功；不能将该attempt抹去或声称环境成本为零。原controller.exit1/evaluate.exit1、日志SHA、失败记录、已完成训练和模型全部保留。本轮检查只读，没有重启、重训、重复评估或修改资产。

此前18:10–19:50估时已失效，当前队列已经停止，不能继续按原速度倒计时。恢复前需要版本化核验既有资产来源/路径、失败attempt与RNG边界，保留原失败及实际成本，再从未完成评估接续；OBJECT600正式完成后才允许GOAL，不能跳过该顺序或重新运行原controller。最终完成源也须按真实恢复工程适配，禁止伪造旧exit0。
'''
(R/'LATEST_PROGRESS_REPORT.md').write_text(report,encoding='utf8')
(R/'EVALUATION_FAILURE_REPORT_20261110.md').write_text(report,encoding='utf8')
p=R/'MONITOR_HANDOFF.md';p.write_text('## '+obs+' 评估失败覆盖旧健康状态及ETA\n\n'+report+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps({k:out[k] for k in ['status','observed','object_training_complete_time','failure_time','eta']},ensure_ascii=False))
