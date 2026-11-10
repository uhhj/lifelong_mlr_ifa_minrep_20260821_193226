"""Record a read-only progress check and explicitly conditional early-stage ETA."""
from pathlib import Path
import json,sys,hashlib,statistics,datetime
R=Path(__file__).resolve().parent
D=Path(sys.argv[1]);old=Path(sys.argv[2])
J=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=J(D/'EVIDENCE.json');a=J(D/'LOCAL_AUDIT.json');prev=J(old/'EVIDENCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
assert e['contract_sha256']=='ce307dca2bce082e36803d266067462cb35a95f421e371f74f7d20d42a5dda8f'
L={n:v['text'] for n,v in e['light'].items()}
assert not any(n.endswith('FAILURE.json') for n in L)
assert all(v.strip()=='0' for n,v in L.items() if n.endswith('.exit'))
assert any(x['pid']==39717 for x in e['processes'])
stage_stats={};admissions={}
for case,c in a['counts'].items():
 rows=json.loads(L.get(case+'/random/epochs.json','[]'))
 stage_stats[case]={}
 for t in sorted({r['task'] for r in rows}):
  rr=[r for r in rows if r['task']==t]
  stage_stats[case][str(t)]=dict(epochs=len(rr),batches=rr[-1]['batches'],mean_seconds=statistics.mean(r['seconds'] for r in rr),last10_seconds=statistics.mean(r['seconds'] for r in rr[-10:]))
 for n,v in L.items():
  if n.startswith(case+'/ADMISSION_'):
   x=json.loads(v)
   assert x['status']=='PASS_OWN_GPU_ALIGN_ADMISSION' and x['optimizer_calls']==5
   assert all(x[k] for k in ['disabled_original_random_full_state_exact','valid_mask_nll_exact','empty_mask_differentiable_zero','serialized_full_state_next_step_exact','initial_state_restored'])
   admissions[n]=x
assert [a['counts'][c]['epochs'] for c in ['OBJECT_41','OBJECT_42']]==[143,144], 'This early-stage forecast is snapshot-specific'
assert all(a['counts'][c]['epochs']==0 for c in ['OBJECT_43','GOAL_41','GOAL_42','GOAL_43'])
est={}
for case in ['OBJECT_41','OBJECT_42']:
 s=stage_stats[case];sec_batch=s['7']['last10_seconds']/227
 est[case]=dict(task7_remaining_minutes=(100-s['7']['epochs'])*s['7']['last10_seconds']/60,
  task8_unmeasured_scaled_seconds=sec_batch*309,task9_unmeasured_scaled_seconds=sec_batch*362,
  remaining_pure_epoch_minutes=((100-s['7']['epochs'])*227+100*(309+362))*sec_batch/60,
  whole_object_pure_minutes=(100*s['6']['mean_seconds']+100*(227+309+362)*sec_batch)/60,
  whole_goal_unmeasured_pure_minutes=76100*sec_batch/60)
tz=datetime.timezone(datetime.timedelta(hours=8));obs=datetime.datetime.fromtimestamp(e['observed'],tz)
planning={'current_object_pair_minutes':[45,65],'object43_minutes':[55,80],
 'object600_formal_minutes':[60,90],'goal_three_training_two_waves_minutes':[100,150],
 'goal600_formal_minutes':[60,90]}
lo=sum(v[0] for v in planning.values());hi=sum(v[1] for v in planning.values())
timing=dict(status='CONDITIONAL_EARLY_STAGE_PLANNING_NOT_COMPLETION',observed=obs.isoformat(),evidence_sha256=sha(D/'EVIDENCE.json'),
 previous_sha256=sha(old/'EVIDENCE.json'),stage_stats=stage_stats,estimates=est,planning_minutes=planning,
 total_remaining_minutes=[lo,hi],finish_window=[(obs+datetime.timedelta(minutes=x)).isoformat() for x in [lo,hi]],
 limits=['Task8/9 and all GOAL stages unmeasured; per-batch scaling is only a planning model.',
 'Epoch seconds exclude checkpoint saves, stage input preparation, admission and dispatch.',
 'Formal evaluation has not started: 60-90 minutes per suite is a planning allowance, not measured throughput.',
 'Two parallel trainers followed by a single remaining seed per suite; no cross-suite overlap.',
 'Final independent CPU acceptance, comparison and publication excluded.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(timing,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
state=dict(status='PARTIAL_TRAINING_OWN_TASK6_TASK7_ADMISSION_PASS',observed=obs.isoformat(),root=J(R/'ACTIVE_ROOT.json')['root'],contract=e['contract_sha256'],controller=39717,
 counts=a['counts'],epochs=a['epochs'],production_updates=a['production_updates'],formal_episodes=a['formal_episodes'],
 admissions=admissions,processes=e['processes'],snapshot=str(D),evidence_sha256=sha(D/'EVIDENCE.json'),prefix_preserved_from=str(old),
 remaining_stages='Task8/9 and other four runs not admitted yet',queue='OBJECT train three then600formal; GOAL same; max2trainers')
for name in ['RUN_STATUS.json','ACTIVE_ROOT.json']:(R/name).write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
report=f'''# ALIGN六run进度与条件估时

观察{obs.isoformat()}，PARTIAL。实际check与独立进度审计通过，上一123348全部完整epoch/诊断前缀保留，新合同与冻结文件verify通过。controller39717与两个GPU训练进程正常，无新失败或重启。

|case|完整轮/400|当前阶段|生产更新|正式/200|
|---|---:|---|---:|---:|
|OBJECT41|143|Task7 43/100|25461|0|
|OBJECT42|144|Task7 44/100|25688|0|
|OBJECT43|0|未开始|0|0|
|GOAL41/42/43|各0|未开始|各0|各0|

合计287/2400轮、51149/544800生产、20/120自身隔离、0训练采集、0/1200正式。OBJECT41/42各自Task6/7准入的禁用ALIGN等价、有效mask、空mask可微零、完整状态保存恢复下一步精确和恢复初态均通过；未来阶段不得套PASS。

Task7近10轮41/42分别{stage_stats['OBJECT_41']['7']['last10_seconds']:.4f}/{stage_stats['OBJECT_42']['7']['last10_seconds']:.4f}秒，当前Task7余约6.49/6.35分钟。按Task7每batch耗时缩放，未测Task8/9约9.3/10.9秒每轮，当前两run剩纯epoch约40分钟、完整OBJECT单run约52分钟、GOAL单run约38分钟；这些不是后阶段实测。

规划当前两run45–65分钟，OBJECT43另55–80分钟，OBJECT600正式60–90分钟；OBJECT整套余160–235分钟（约15:27–16:42）。随后GOAL三训练两波100–150分钟和600正式60–90分钟。六run训练+1200正式合计320–475分钟，用户口径约5.5–8小时（约今日18:10–20:45）。完整CPU验收、比较和发布另计。Task8/9、GOAL和正式吞吐未实测，不作为完成保证；本次未改科学代码、调度、训练或评估。

详细公式和来源SHA见本快照TIMING_AUDIT.json。全部OBJECT闭环完成后才进入GOAL；已有S、q1和旧失败证据只读保留。
'''
(R/'LATEST_PROGRESS_REPORT.md').write_text(report,encoding='utf8')
p=R/'MONITOR_HANDOFF.md';p.write_text('## '+obs.isoformat()+' 最新进度\n\n'+report.split('\n',1)[1]+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(dict(status=state['status'],estimates=est,total_remaining_minutes=[lo,hi],finish_window=timing['finish_window']),ensure_ascii=False))
