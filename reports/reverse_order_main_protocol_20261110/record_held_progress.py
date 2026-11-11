"""Record a read-only observation without clearing the dispatcher review hold."""
from pathlib import Path
import datetime,json,sys
from audit_progress import read,decode
R=Path(__file__).resolve().parent
d=Path(sys.argv[1]);a=read(d/'LOCAL_AUDIT.json');e=read(d/'EVIDENCE.json');o=decode(e)
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
active=read(R/'ACTIVE_ROOT.json');controller=active['controller']
assert any(p['pid']==controller and p['state']=='T' for p in e['processes'])
hold=read(R/'PROTOCOL_REVIEW_HOLD.json')
assert hold['controller']==controller
stamp=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8)))
spec=read(R/'SPEC.json');cases=a['cases'];eta={}
for name,x in cases.items():
    method=spec['cases'][name]['method']
    completed=o.get(name+('/native/TRAINING_COMPLETE.json' if method=='MLR' else '/TRAIN_S.json'))
    calls=o.get(name+'/OPTIMIZER_CALLS.json',{})
    if x['epochs']==400 and completed and calls.get('status')=='ACTUAL_CALLS_COMPLETED':
        inp=read(R/'cases'/name/'INPUT.json')
        assert calls['calls']['production']==inp['physical_budget']
        x['training_complete_record_verified']=True
        x['operational_state']='TRAINING_COMPLETE' if x['train_exit']=='0' else 'TRAINING_COMPLETE_DISPATCHER_EXIT_REAP_PENDING'
    elif x['epochs']:
        x['operational_state']='ACTIVE_TRAINING' if any(p['cwd'].endswith('/cases/'+name) and p['state'] not in ['T','Z'] for p in e['processes']) else 'REVIEW_NO_ACTIVE_CHILD'
    else:x['operational_state']='NOT_DISPATCHED'
    if x['operational_state']!='ACTIVE_TRAINING':continue
    inp=read(R/'cases'/name/'INPUT.json')
    h=o[name+('/native/epochs.json' if method=='MLR' else '/EPOCHS_S.json')]
    stage=h[-1]['task' if method=='MLR' else 'stage'];recent=[r for r in h if r.get('stage',r.get('task'))==stage][-10:]
    seconds=sum(r['seconds'] for r in recent)/len(recent)
    rate=sum(r['seconds'] for r in recent)/sum(r['batches']*(2 if method=='DEBR' and stage>6 else 1) for r in recent)
    remain=0;details=[]
    for st in range(6,10):
        done=sum(r.get('stage',r.get('task'))==st for r in h)
        if done==100:continue
        own=[r for r in h if r.get('stage',r.get('task'))==st][-10:]
        per=sum(r['seconds'] for r in own)/len(own) if own else rate*inp['batches'][str(st)]*(2 if method=='DEBR' and st>6 else 1)
        remain+=(100-done)*per/60
        details.append(dict(stage=st,original_task=inp['order'][st],complete_epochs=done,remaining_epochs=100-done,seconds_per_epoch=per,unmeasured_stage_extrapolation=not bool(own)))
    allowance=20 if method=='DEBR' else 12
    eta[name]=dict(pure_epoch_minutes=remain,planning_allowance_minutes=allowance,minutes=remain+allowance,estimated_CST=(stamp+datetime.timedelta(minutes=remain+allowance)).isoformat(),last_ten_seconds_per_epoch=seconds,remaining_stages=details,not_guaranteed=True,excludes_formal_and_final_acceptance=True)
status=dict(status='PARTIAL_DISPATCH_HELD_PROTOCOL_REVIEW',observed=stamp.isoformat(),contract=a['contract'],evidence_directory=str(d.resolve()),totals=a['totals'],cases=cases,processes=e['processes'],controller_exit=a['controller_exit'],capacity=e['capacity'],gpu=e['gpu'],protocol_review_hold=hold,phase='DISPATCH_HELD_ACTIVE_CHILD_TRAINING',eta_active_children_only=eta,next_checkpoint_CST=None,background_automation_created=False,actual_final_cpu_acceptance=False,read_only_check_added_optimizer=0,read_only_check_added_environments=0,dispatcher_resumed=False)
(R/'RUN_STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
active.update(status=status['status'],observed=stamp.isoformat(),next_checkpoint_CST=None)
(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rows='\n'.join('|%s|%s|%d|%d|%d|%d|'%(n,x['operational_state'],x['epochs'],x['accepted'],x['training_collection'],x['formal']) for n,x in cases.items())
report='# 主表协议逆序实验：派发暂停下的只读进度\n\n实际观察%s，PARTIAL。独立轻量审计PASS_PROGRESS_NOT_FINAL。controller%d实际状态T，未来派发保持暂停；在训子进程未停止，未自动恢复派发。\n\n基底任务与官方预训练不变，增量原物理任务9→8→7→6；两方法各自继承主表原生训练和评估实现，未在本次检查改协议。\n\n|case|实际运行状态|完整epoch|接受更新|训练采集|正式|\n|---|---|---:|---:|---:|---:|\n%s\n\n训练完成记录与optimizer计数完整不等于最终科学验收。MLR42的父派发器退出回收尚待进行，不把未记录的launcher退出码伪称0。正式评估尚未启动。\n\n在训任务ETA（仅当前子进程，未测阶段按最近实际每batch速度外推；含明确规划余量，不含正式评估／最终验收）：\n\n%s\n\nGPU与容量见RUN_STATUS。无新失败记录，冻结源及旧证据前缀核验通过。本次0新增优化器调用／环境／恢复派发。没有设定下一自动查询；不重启本地观察器。\n'%(stamp.isoformat(),controller,rows,json.dumps(eta,ensure_ascii=False,indent=2))
(R/'REPORT.md').write_text(report,encoding='utf-8')
handoff='# 当前逆序任务交接\n\n最新只读观察%s，证据%s。PARTIAL_DISPATCH_HELD_PROTOCOL_REVIEW；controller%d实际T。保留当前子进程继续原运行，不自动CONT派发器、部署、重启、补跑或恢复本地观察器。旧自动查询时间全部失效。此次没有新增自动化。\n\n先读PROTOCOL、IMPLEMENTATION_NOTES、ACTIVE_ROOT、RUN_STATUS、PROTOCOL_REVIEW_HOLD及最新LOCAL_AUDIT。唯一root/合同见ACTIVE_ROOT；不得重复完成epoch、采集或正式attempt。MLR42训练完成但父派发器尚未记录退出码，不能伪称该launcher exit0。实际正式0。\n\n只读查询transport.py check_readonly.py → audit_progress.py NEW PREVIOUS → record_held_progress.py NEW → publish.py。此记录器保持派发暂停状态；不要调用会清除该标记的旧record_progress.py。仅用户明确要求恢复后才另行处理派发。最终自然exit0且无自身进程后再运行真实CPU最终路径，不能以当前进度代替验收。\n\n凭据仅Windows Credential Manager／内存；禁止回显、明文参数日志。仅发布经审核轻量代码、报告、hash、标量和outcomes。\n'%(stamp.isoformat(),d.name,controller)
(R/'MONITOR_HANDOFF.md').write_text(handoff,encoding='utf-8')
print(json.dumps(dict(status=status['status'],observed=stamp.isoformat(),totals=a['totals'],active_eta=eta,dispatcher_resumed=False),ensure_ascii=False))
