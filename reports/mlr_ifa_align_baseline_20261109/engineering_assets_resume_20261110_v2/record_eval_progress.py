"""Record actual vector throughput and conditional remaining queue time."""
from pathlib import Path
import json,sys,statistics,datetime,hashlib
R=Path(__file__).resolve().parent;B=R.parent;D=Path(sys.argv[1]);J=lambda p:json.loads(p.read_text(encoding='utf8'))
e=J(D/'EVIDENCE.json');a=J(D/'LOCAL_AUDIT.json');eng=J(D/'ENGINEERING_AUDIT.json');L={n:v['text'] for n,v in e['light'].items()}
assert eng['status']=='PASS_ENGINEERING_PROGRESS_NOT_FINAL' and eng['rng_boundary']['status']=='PASS_EXACT_PRE_CELL_RNG'
assert a['counts']['OBJECT_41']['formal_episodes']==110 and a['epochs']==1200
assert any(p['pid']==40721 for p in e['processes']) and 'engineering/FAILURE.json' not in L
vectors=[]
for n,t in L.items():
 if not n.startswith('OBJECT_41/random/evaluation/attempts_'):continue
 rows=[json.loads(x) for x in t.splitlines()];assert not any(x['event']=='technical_exception' for x in rows)
 starts={x['group']:x['unix'] for x in rows if x['event']=='vector_attempt_start'}
 for x in rows:
  if x['event']=='vector_attempt_complete':vectors.append(dict(stage=x['stage'],task=x['task'],group=x['group'],seconds=x['unix']-starts[x['group']],steps=x['steps'],end=x['unix']))
vectors.sort(key=lambda x:x['end']);assert len(vectors)==11
mean=statistics.mean(x['seconds'] for x in vectors);recent=statistics.mean(x['seconds'] for x in vectors[-5:]);remaining=60-len(vectors)
tz=datetime.timezone(datetime.timedelta(hours=8));obs=datetime.datetime.fromtimestamp(e['observed'],tz)
planning=dict(object_formal_remaining_minutes=[160,220],goal_training_three_seeds_two_waves_minutes=[100,130],goal600_formal_minutes=[190,250])
lo=sum(x[0] for x in planning.values());hi=sum(x[1] for x in planning.values())
timing=dict(status='ACTUAL_OBJECT_VECTOR_TIMING_CONDITIONAL_QUEUE_ETA',observed=obs.isoformat(),evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),vectors=vectors,mean_seconds=mean,last5_seconds=recent,
 remaining_object_vectors=remaining,object_linear_reference_minutes=[remaining*recent/60,remaining*mean/60],planning=planning,total_remaining_minutes=[lo,hi],finish_window=[(obs+datetime.timedelta(minutes=x)).isoformat() for x in [lo,hi]],
 limits=['Only OBJECT41 first11 vectors observed; later checkpoints/seeds may terminate sooner or later.', 'GOAL training/evaluation not measured. GOAL training100-130min uses previous OBJECT full-stage batch-scaled two-wave reference.', 'Evaluation planning includes initialization and I/O; previous60-90min per suite allowance is superseded.', 'Final CPU acceptance/comparison/publication excluded.'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(timing,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
state=J(B/'ACTIVE_ROOT.json');state.update(status='PARTIAL_RECOVERED_OBJECT_EVALUATING',observed=obs.isoformat(),counts=a['counts'],epochs=a['epochs'],production_updates=a['production_updates'],formal_episodes=a['formal_episodes'],snapshot=str(D),evidence_sha256=timing['evidence_sha256'],remaining_total_time_minutes=[lo,hi])
for n in ['ACTIVE_ROOT.json','RUN_STATUS.json']:(B/n).write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
report=f'''# ALIGN恢复后进度与实测评估速度

观察{obs.isoformat()}，PARTIAL。新工程进度审计PASS，152934全部训练/准入/正式前缀精确保留；原科学和工程合同verify通过，三个OBJECT模型及恢复文件SHA不变。controller40721/OBJECT41评估40723与环境子进程正常，无新错误、重启或重复训练。

|case|训练/400|正式/200|状态|
|---|---:|---:|---|
|OBJECT41|400|110|Task8模型的Task8第二批进行中|
|OBJECT42|400|0|等待串行评估|
|OBJECT43|400|0|等待串行评估|
|GOAL41/42/43|各0|各0|等待OBJECT600正式全部完成|

OBJECT三run的Task6–9各自ALIGN准入均已通过。累计1200完整轮、316500生产、60自身隔离、0训练采集、110/1200正式；原失败attempt和exit1保持。首个恢复单元四类RNG精确PASS，环境锁目前由OBJECT41独占，没有新增工程失败。

11个实际完成vector（每批10回合）均{mean:.4f}秒，近5批{recent:.4f}秒；600步批次约223–231秒，两个提前结束批次分别217/210步、82.26/84.66秒。不同任务终止长度直接影响墙钟，不能把所有批次套同一速度。

OBJECT剩49批线性参考{remaining*recent/60:.2f}–{remaining*mean/60:.2f}分钟，加初始化、后任务/seed波动规划160–220分钟，即约18:45–19:45。GOAL三训练两波仍为未测规划100–130分钟，其600正式按当前实现预留190–250分钟。全队列训练+正式余450–600分钟，约7.5–10小时，11月10日23:35至11月11日02:05附近；CPU最终验收、比较发布另计。这覆盖早先60–90分钟/套件及当天18–20时完成的旧估计，不按时钟断言完成。

本轮没有改训练/推理/评估代码，没有重训或重评。后续继续用engineering_assets_resume_20261110_v2的check/audit及最终包装，真实成功源仍仅新工程controller.exit0且无自身进程；旧退出1和失败证据保留。
'''
(B/'LATEST_PROGRESS_REPORT.md').write_text(report,encoding='utf8')
p=B/'MONITOR_HANDOFF.md';p.write_text('## '+obs.isoformat()+' 恢复后正式评估推进\n\n'+report+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(dict(status=state['status'],formal_episodes=a['formal_episodes'],mean_seconds=mean,last5_seconds=recent,total_remaining_minutes=[lo,hi],finish_window=timing['finish_window']),ensure_ascii=False))
