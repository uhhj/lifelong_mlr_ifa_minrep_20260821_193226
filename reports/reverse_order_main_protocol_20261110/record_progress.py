"""Save an audited observation and next whole-run or suite-evaluation checkpoint."""
from pathlib import Path
import datetime,json,math,sys
from audit_progress import decode,read
R=Path(__file__).resolve().parent;d=Path(sys.argv[1]);a=read(d/'LOCAL_AUDIT.json');e=read(d/'EVIDENCE.json');obj=decode(e)
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
stamp=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8)))
spec=read(R/'SPEC.json');state=obj.get('coordinator/STATUS.json',{});eta={}
if a['controller_exit']=='0' and not a['processes']:
    phase='READY_ACTUAL_FINAL_CPU';minutes=0
elif state.get('status')=='PARTIAL_TRAINING':
    phase='CURRENT_WHOLE_RUN_TRAINING_AND_COLLECTION'
    for case in state['active']:
        inp=read(R/'cases'/case/'INPUT.json');method=inp['method'];h=obj.get(case+('/native/epochs.json' if method=='MLR' else '/EPOCHS_S.json'),[])
        observed=[x for x in h[-20:] if x['batches']>0]
        if not observed:eta[case]=dict(minutes=None,reason='No actual complete epoch yet');continue
        rate=sum(x['seconds'] for x in observed)/sum(x['batches']*(2 if method=='DEBR' and x.get('stage',6)>6 else 1) for x in observed)
        remaining=0;stage_details={}
        for stage in range(6,10):
            done=sum(x['task' if method=='MLR' else 'stage']==stage for x in h)
            candidates=[x for x in h if x['task' if method=='MLR' else 'stage']==stage][-10:]
            borrowed=False
            if not candidates:
                for other,entry in spec['cases'].items():
                    if entry['suite']==inp['suite'] and entry['method']==method and other!=case:
                        candidates=[x for x in obj.get(other+('/native/epochs.json' if method=='MLR' else '/EPOCHS_S.json'),[]) if x['task' if method=='MLR' else 'stage']==stage][-10:]
                        if candidates:borrowed=True;break
            seconds=sum(x['seconds'] for x in candidates)/len(candidates) if candidates else rate*inp['batches'][str(stage)]*(2 if method=='DEBR' and stage>6 else 1)
            remaining+=(100-done)*seconds/60
            stage_details[str(stage)]=dict(epochs=done,seconds_per_epoch=seconds,extrapolated=not candidates,borrowed=borrowed)
        # No repeated task-boundary checks. Entire remaining run includes a stated
        # preparation/save/collection planning allowance, not a guaranteed bound.
        allowance=20 if method=='DEBR' else 12
        eta[case]=dict(minutes=remaining+allowance,pure_epoch_minutes=remaining,planning_allowance_minutes=allowance,stages=stage_details)
    measured=[v['minutes'] for v in eta.values() if v['minutes'] is not None]
    minutes=math.ceil(min(measured)) if measured else 120
elif state.get('status')=='PARTIAL_EVALUATING':
    phase='ENTIRE_CURRENT_SUITE_EVALUATION';case=state['case'];suite=spec['cases'][case]['suite'];remaining=0;known=[]
    for name,entry in spec['cases'].items():
        if entry['suite']!=suite or a['cases'][name]['evaluate_exit']=='0':continue
        dispatch=obj.get(name+'/EVAL_DISPATCH.json')
        count=a['cases'][name]['formal']
        if dispatch and count:
            seconds=(e['observed']-dispatch['time'])/count;known.append(seconds)
        else:seconds=None
        # Native full-history MLR and cached DEBR have different measured speeds.
        # Pending methods without evidence retain a conservative planning estimate.
        peers=[]
        for other,meta in spec['cases'].items():
            if meta['method']!=entry['method']:continue
            start=obj.get(other+'/EVAL_DISPATCH.json');n=a['cases'][other]['formal']
            if start and n:
                end=obj.get(other+'/EVALUATION_COMPLETE.json',{}).get('time',e['observed'])
                peers.append((end-start['time'])/n)
        rate=seconds if seconds is not None else (sum(peers)/len(peers) if peers else (45 if entry['method']=='MLR' else 8))
        rem=(200-count)*rate/60+5;remaining+=rem
        eta[name]=dict(remaining_episodes=200-count,seconds_per_episode=rate,planning_only=seconds is None and not peers,minutes=rem)
    minutes=math.ceil(remaining)
else:phase='REVIEW_STATUS';minutes=30
next_time=stamp+datetime.timedelta(minutes=minutes)
status=dict(status='PARTIAL_EXECUTION_PENDING_FINAL_ACCEPTANCE',observed=stamp.isoformat(),evidence_directory=str(d.resolve()),contract=a['contract'],totals=a['totals'],cases=a['cases'],controller_exit=a['controller_exit'],processes=a['processes'],actual_final_cpu_acceptance=False,phase=phase,eta=eta,next_checkpoint_CST=next_time.isoformat(),eta_is_not_a_guaranteed_bound=True,background_automation_created=False)
(R/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rows='\n'.join('|%s|%d|%d|%d|%d|%d|'%(n,v['epochs'],v['accepted'],v['production_physical'],v['training_collection'],v['formal']) for n,v in a['cases'].items())
(R/'REPORT.md').write_text('# 主表协议逆序实验\n\nPARTIAL，实际观察%s。只读审计PASS_PROGRESS_NOT_FINAL，不代表最终接受或局部性能结论。\n\n原基底0–5和官方检查点不变，增量物理任务9→8→7→6。MLR+IFA和完整DEBR分别继承主表训练与原生评估实现。\n\n|case|完整epoch|接受更新|生产物理更新|训练采集|正式|\n|---|---:|---:|---:|---:|---:|\n%s\n\n下一检查%s，范围%s；估计含明确规划余量，不是全队列保证结束时间。最终48CP/12恢复/24缓存与240训练2400正式字节及预算尚待实际验收。\n'%(stamp.isoformat(),rows,next_time.isoformat(),phase),encoding='utf-8')
(R/'MONITOR_HANDOFF.md').write_text('# 当前逆序任务交接\n\n先读PROTOCOL、IMPLEMENTATION_NOTES、ACTIVE_ROOT、RUN_STATUS和最新证据。唯一root见SPEC，禁止重复deploy/build/prepare/controller及完整epoch、采集或正式attempt。\n\n最新实际观察%s，证据%s。下一检查%s，范围%s。transport.py check_readonly.py → audit_progress.py NEW PREVIOUS → record_progress.py NEW → publish.py。最终自然exit0且无自身进程后才final_readonly.py → audit_final_local.py → describe_final.py。所有wrapper错误保留原attempt后外置修复，不重训重评；不自动扩seed调参。\n\n凭据仅Windows Credential Manager/内存；不得回显、写参数日志。公开仅代码报告/元数据SHA/标量/outcomes。\n'%(stamp.isoformat(),d.name,next_time.isoformat(),phase),encoding='utf-8')
print(json.dumps(dict(status=status['status'],phase=phase,observed=stamp.isoformat(),totals=a['totals'],next_checkpoint_CST=next_time.isoformat())))
