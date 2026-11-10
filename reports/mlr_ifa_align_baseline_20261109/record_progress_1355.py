"""Read-only second progress observation; full-stage measured timing references."""
from pathlib import Path
import json,sys,hashlib,statistics,datetime
R=Path(__file__).resolve().parent;D=Path(sys.argv[1]);P=Path(sys.argv[2])
J=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
e=J(D/'EVIDENCE.json');a=J(D/'LOCAL_AUDIT.json');L={n:v['text'] for n,v in e['light'].items()}
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
assert e['contract_sha256']=='ce307dca2bce082e36803d266067462cb35a95f421e371f74f7d20d42a5dda8f'
assert not any(n.endswith('FAILURE.json') for n in L)
assert all(v.strip()=='0' for n,v in L.items() if n.endswith('.exit'))
assert any(p['pid']==39717 for p in e['processes'])
stats={};admissions={}
for case,c in a['counts'].items():
 rows=json.loads(L.get(case+'/random/epochs.json','[]'));stats[case]={}
 for t in sorted({r['task'] for r in rows}):
  rr=[r for r in rows if r['task']==t]
  stats[case][str(t)]=dict(epochs=len(rr),batches=rr[-1]['batches'],mean_seconds=statistics.mean(r['seconds'] for r in rr),last10_seconds=statistics.mean(r['seconds'] for r in rr[-10:]))
 for n,v in L.items():
  if n.startswith(case+'/ADMISSION_'):
   x=json.loads(v);assert x['status']=='PASS_OWN_GPU_ALIGN_ADMISSION' and x['optimizer_calls']==5
   assert all(x[k] for k in ['disabled_original_random_full_state_exact','valid_mask_nll_exact','empty_mask_differentiable_zero','serialized_full_state_next_step_exact','initial_state_restored'])
   admissions[n]=x
 if rows:assert json.loads(L[case+'/INITIALIZATION_AUDIT.json'])['initial_exact']
assert [a['counts'][c]['epochs'] for c in ['OBJECT_41','OBJECT_42','OBJECT_43']]==[400,400,219]
assert all(a['counts'][c]['epochs']==0 for c in ['GOAL_41','GOAL_42','GOAL_43'])
events=[json.loads(s) for s in L['coordinator/EVENTS.jsonl'].splitlines()]
dispatch={x['case']:x['time'] for x in events if x['kind']=='launch' and x['phase']=='train'}
exits={x['case']:x['time'] for x in events if x['kind']=='exit' and x['phase']=='train' and x['code']==0}
for c in ['OBJECT_41','OBJECT_42']:
 assert L[c+'/train.exit'].strip()=='0' and c in exits
 assert json.loads(L[c+'/random/TRAINING_COMPLETE.json'])['epochs']==400
assert 'OBJECT_43/ADMISSION_9.json' not in L
ref8=statistics.mean(stats[c]['8']['mean_seconds'] for c in ['OBJECT_41','OBJECT_42'])
ref9=statistics.mean(stats[c]['9']['mean_seconds'] for c in ['OBJECT_41','OBJECT_42'])
cur8=stats['OBJECT_43']['8']['last10_seconds'];t8remain=81*cur8/60
pure=[t8remain+100*ref9*(cur8/ref8)/60,t8remain+100*ref9/60]
goal_pure={c:sum(stats[c][str(t)]['mean_seconds']/stats[c][str(t)]['batches']*b*100 for t,b in zip(range(6,10),[100,139,185,337]))/60 for c in ['OBJECT_41','OBJECT_42']}
tz=datetime.timezone(datetime.timedelta(hours=8));obs=datetime.datetime.fromtimestamp(e['observed'],tz)
planning=dict(object43_training_minutes=[35,45],object600_formal_minutes=[60,90],goal_three_training_two_waves_minutes=[100,130],goal600_formal_minutes=[60,90])
lo=sum(v[0] for v in planning.values());hi=sum(v[1] for v in planning.values())
timing=dict(status='CONDITIONAL_PLANNING_NOT_COMPLETION',observed=obs.isoformat(),evidence_sha256=sha(D/'EVIDENCE.json'),previous_sha256=sha(P/'EVIDENCE.json'),
 stage_stats=stats,completed_run_dispatch_to_exit_minutes={c:(exits[c]-dispatch[c])/60 for c in exits},object43_dispatch=dispatch['OBJECT_43'],
 object43_task8_remaining_pure_minutes=t8remain,object43_task9_reference_seconds=ref9,object43_remaining_pure_minutes=pure,
 goal_single_run_unmeasured_batch_scaled_pure_minutes=goal_pure,planning=planning,total_remaining_minutes=[lo,hi],finish_window=[(obs+datetime.timedelta(minutes=x)).isoformat() for x in [lo,hi]],
 limitations=['OBJECT43 Task9 not yet measured or admitted; scaled Task8 speed and completed peers supply a conditional range.',
 'GOAL not started: batch-scaled OBJECT stage times are references, not measurements.',
 'Formal evaluation not started: 60-90 minutes per600 is a planning allowance, not actual throughput.',
 'Epoch timings exclude saves/preparation/admission. No overlapping suites. CPU acceptance/report/publication excluded.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(timing,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
state=dict(status='PARTIAL_OBJECT41_42_TRAIN_COMPLETE_OBJECT43_TASK8',observed=obs.isoformat(),root=J(R/'ACTIVE_ROOT.json')['root'],contract=e['contract_sha256'],controller=39717,
 counts=a['counts'],epochs=a['epochs'],production_updates=a['production_updates'],formal_episodes=a['formal_episodes'],admissions=admissions,processes=e['processes'],
 snapshot=str(D),prefix_preserved_from=str(P),evidence_sha256=sha(D/'EVIDENCE.json'),remaining_stages='OBJECT43 Task9 and all GOAL stages not admitted yet',queue='OBJECT train three then600formal; GOAL same; max2trainers')
for name in ['RUN_STATUS.json','ACTIVE_ROOT.json']:(R/name).write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
report=f'''# ALIGN最新进度与估时

观察{obs.isoformat()}，PARTIAL。实际check与audit_progress PASS，124649全部完整epoch/诊断前缀保持，冻结合同verify通过。controller39717和OBJECT43训练39906健康，无新失败或重启。

|case|完整轮/400|当前阶段|Task8准入|Task9准入|正式/200|
|---|---:|---|---|---|---:|
|OBJECT41|400|train.exit0|PASS|PASS|0|
|OBJECT42|400|train.exit0|PASS|PASS|0|
|OBJECT43|219|Task8 19/100|PASS|尚未进入|0|
|GOAL41/42/43|各0|未开始|未执行|未执行|各0|

41/42自己的四阶段五调用ALIGN准入全部通过；43初始化tensor/RNG及Task6/7/8准入通过，每次均核禁用原更新等价、有效mask、空mask可微零、保存恢复下一步精确及恢复初态，不能套未来PASS。合计1019/2400轮、255271/544800生产更新、55/120隔离、0训练采集、0/1200正式。43真实13:32:13启动；41/42完整训练dispatch至exit分别{timing['completed_run_dispatch_to_exit_minutes']['OBJECT_41']:.2f}/{timing['completed_run_dispatch_to_exit_minutes']['OBJECT_42']:.2f}分钟，含准备保存准入。

41/42 Task8均9.2932/9.3721秒、Task9均10.6758/10.7058秒；43当前单训练Task8近10轮{cur8:.4f}秒，余81轮约{t8remain:.2f}分钟纯计算。Task9尚未实测，参考同套件已完成run及43当前速度缩放，43整run剩纯epoch约{pure[0]:.2f}–{pure[1]:.2f}分钟，计保存准备准入规划35–45分钟。

OBJECT600正式尚未开始，暂留60–90分钟；OBJECT整套训练+正式余95–135分钟，即今日15:30–16:10附近。GOAL三训练按两个波次留100–130分钟、600正式60–90分钟。全部六run训练及1200正式合计余255–355分钟，即约4小时15分–6小时，今日18:10–19:50附近。最终独立CPU验收、结果比较和发布另计。GOAL和正式尚未实测，不能将规划当完成保证；详细公式和来源见本快照TIMING_AUDIT.json。

保持OBJECT三训练→600正式→GOAL三训练→600正式，最多两个训练；本次没有改科学代码、调度、训练或评估。已有S/q1及旧失败证据保持。后续仅v4/transport.py check_readonly.py检查，禁止重复launch/controller/完成轮。
'''
(R/'LATEST_PROGRESS_REPORT.md').write_text(report,encoding='utf8')
p=R/'MONITOR_HANDOFF.md';p.write_text('## '+obs.isoformat()+' 最新进度\n\n'+report+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(timing,ensure_ascii=False))
