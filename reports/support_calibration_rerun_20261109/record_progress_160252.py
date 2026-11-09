"""Recompute observed timing and record explicitly conditional queue estimates."""
from pathlib import Path
import json,hashlib,statistics,datetime
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261109_160252'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=js(D/'EVIDENCE.json');a=js(D/'LOCAL_AUDIT.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL'
history={}
for suite,stamp in [('OBJECT','20261108_223300'),('GOAL','20261109_120106')]:
 p=R.parent/('20261108_'+suite.lower()+'_s_q1_ablation')/('final_readonly_'+stamp)/'EVIDENCE.json';h=js(p)
 events=[json.loads(x) for x in h['light']['CONTROLLER_EVENTS.jsonl']['text'].splitlines()]
 stages={};walls={}
 for seed in [41,42,43]:
  case='Q1_'+str(seed);epochs=json.loads(h['light'][case+'/EPOCHS_S.json']['text'])
  stages[case]={str(st):statistics.mean(x['seconds'] for x in epochs if x['stage']==st) for st in [6,7,8,9]}
  walls[case]=(next(x['time'] for x in events if x['kind']=='train_exit' and x['case']==case)-next(x['time'] for x in events if x['kind']=='train_launch' and x['case']==case))/60
 trainwall=(max(x['time'] for x in events if x['kind']=='train_exit')-min(x['time'] for x in events if x['kind']=='train_launch'))/60
 evalwall=(max(x['time'] for x in events if x['kind']=='evaluate_exit')-min(x['time'] for x in events if x['kind']=='evaluate_launch'))/60
 history[suite]=dict(source=str(p),sha256=sha(p),stage_mean_seconds=stages,run_wall_minutes=walls,whole_training_wall_minutes=trainwall,whole_formal_wall_minutes=evalwall)
measured={};admissions={}
for seed in [41,42]:
 case='Q1_'+str(seed);epochs=json.loads(v['light'][case+'/EPOCHS_S.json']['text'])
 s=[x['seconds'] for x in epochs if x['stage']==7]
 first=statistics.mean(s[:10]);last=statistics.mean(s[-10:]);scale=last/history['OBJECT']['stage_mean_seconds'][case]['7']
 later=sum(history['OBJECT']['stage_mean_seconds'][case][str(st)]*100 for st in [8,9])*scale/60
 measured[case]=dict(stage7_completed=len(s),first10_seconds=first,last10_seconds=last,last_vs_first_percent=(last/first-1)*100,remaining_stage7_pure_minutes=(100-len(s))*last/60,reference_scale=scale,unmeasured_stage8_9_extrapolated_pure_minutes=later,conditional_current_run_pure_minutes=(100-len(s))*last/60+later)
 admit=json.loads(v['light'][case+'/ADMISSION_S_7.json']['text'])
 checks=['roundtrip_next_step_exact','q_one_original_S_exact','lambda_zero_original_C_exact','ten_step_C_replay_exact','accepted_and_rejected_full_state_restore','measurement_state_exact','probe_all_state_exact','q1_actual_memory_uniform_exact','q1_short_history_exact','synthetic_gate_accept_reject']
 assert admit['status']=='PASS_OWN_GPU_ADMISSION' and admit['optimizer_calls']==29 and admit['extra_autograd_calls']==11 and all(admit[k] for k in checks)
 assert all(x['raw_q_point_one_ignored'] and x['effective_q_one_exact'] and x['finite_gradient'] for x in admit['short_history'])
 admissions[case]=dict(stage=7,status=admit['status'],checks={k:admit[k] for k in checks},raw_q_point_one_ignored_lengths_1_to_8=True,optimizer_calls=29,autograd=11,source_sha256=v['light'][case+'/ADMISSION_S_7.json']['sha256'])
ranges=dict(current_OBJECT_parallel_pair=[220,270],OBJECT_seed43=[210,260],OBJECT_600_formal=[60,90],GOAL_three_training_with_two_slots=[340,420],GOAL_600_formal=[60,90])
total=[sum(x[i] for x in ranges.values()) for i in [0,1]]
obs=datetime.datetime.fromtimestamp(v['observed'],datetime.timezone(datetime.timedelta(hours=8)))
timing=dict(status='CONDITIONAL_PLANNING_NOT_COMPLETION_GUARANTEE',observation=obs.isoformat(),snapshot_sha256=sha(D/'EVIDENCE.json'),measured=measured,historical_references=history,planning_ranges_minutes=ranges,total_minutes=total,rounded_user_hours=[15,19],rounded_finish_times=[(obs+datetime.timedelta(hours=x)).isoformat() for x in [15,19]],basis='Current Task7 last10 / previous same-seed Task7 means scales historical later-stage costs. Add preparation/checkpoint/admission/collection/lock and throughput variation allowances. Seed43 single tail and GOAL pair-plus-tail preserved. No assumed third concurrent trainer.',limitations=['Task8/9, seed43 and all GOAL stages not yet measured in this rerun','Different hardware execution and own success-cache composition can change stage cost; ratio is not a guaranteed whole-queue speedup','Formal lengths depend on termination; previous 600-episode wall times are historical only','CPU independent acceptance, interpretation and publication excluded'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(timing,indent=2)+'\n',encoding='utf8')
(D/'Q1_ADMISSION_AUDIT.json').write_text(json.dumps(admissions,indent=2)+'\n',encoding='utf8')
status=js(R/'RUN_STATUS.json');status.update(observation=a['observation'],counts=a['counts'],new_total_epochs=a['epochs'],new_total_physical_optimizer_calls=a['physical_training_optimizer_calls'],new_total_isolated_optimizer_calls=a['own_isolated_optimizer_calls'],new_total_admission_autograd=a['own_extra_autograd_calls'],new_total_collection=a['training_collection_completed_episodes'],new_total_formal=a['formal_episodes'],failures=a['failures'],evidence=str(D),eta=timing['rounded_user_hours'],task7_q1_admission=admissions)
(R/'RUN_STATUS.json').write_text(json.dumps(status,indent=2)+'\n',encoding='utf8')
handoff=R/'MONITOR_HANDOFF.md';old=handoff.read_text(encoding='utf8')
header='''## 2026-11-09 16:02 实际进度与条件估时

check_readonly_20261109_160252与本地进度审计PASS，154009全部epoch/门控/诊断/正式前缀保留，总协调器148冻结文件和两个套件合同保持。OBJECT41 Task6完整100轮、Task7完整23轮；42 Task6完整100轮、Task7完整24轮。各10采集、33隔离/19准入autograd，正式0；43及GOAL未开始。累计247/2400轮、52738/1012500生产、66隔离/38准入autograd、20/240采集、0/1200正式，无失败。两次采集锁完整互斥。

两者Task7固定q1自己的29optimizer/11autograd GPU准入通过：真实缓存均匀保持更新精确、原始q=.1忽略、短历史1–8、10步及保存加载/接受拒绝恢复精确；COPY_EQ/BRANCH_COPY_EQ/CHECK_EQ在0/15700阶段起点通过。Task8/9未来不能套PASS。

Task7近10轮28.8094/31.0168秒，对各自首10轮29.6703/31.3154秒，暂未见持续减速；剩Task7纯计算36.97/39.29分钟。Task8/9等尚未测，以旧同配置各阶段记录及当前相对速度条件外推，完整来源SHA/公式见本快照TIMING_AUDIT。六run训练采集与1200正式规划余约15–19小时，即11月10日07–11时附近；CPU独立验收解释发布另计。当前两run余220–270分钟含准备采集，OBJECT43预计210–260分钟，GOAL三run最多两槽340–420分钟，两组正式各60–90分钟。不可按时钟断言完成，不固定高频检查。

其余部署/协议/最终验收限制继续；当前capacity_readonly.py为20个1秒GPU采样脚本，不再是删除后容量核验。不要再次调用删除或下载入口。用户仅要求本次检查与估时，未创建自动定时任务。

'''
handoff.write_text(header+old,encoding='utf8')
print(json.dumps(dict(measured=measured,total_minutes=total,finish=timing['rounded_finish_times'],historical_formal_minutes={k:x['whole_formal_wall_minutes'] for k,x in history.items()}),indent=2))
