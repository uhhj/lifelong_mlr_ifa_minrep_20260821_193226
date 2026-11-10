"""Current-task checkpoint: recompute lightweight integrity and next phase ETA.

No training, environment, recovery, remote write, or scheduler is invoked.
"""
from pathlib import Path
import datetime
import hashlib
import json
import math
import statistics
import sys
import numpy as np

R=Path(__file__).resolve().parent
d=Path(sys.argv[1]).resolve()
J=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
e=J(d/'EVIDENCE.json');a=J(d/'LOCAL_AUDIT.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL'
assert e['new_optimizer']==e['new_environments']==0
objects={}
for k,v in e['light'].items():
    assert hashlib.sha256(v['text'].encode()).hexdigest()==v['sha256']
    objects[k]=[json.loads(x) for x in v['text'].splitlines() if x] if k.endswith('.jsonl') else json.loads(v['text']) if k.endswith('.json') else v['text'].strip()
events=objects.get('coordinator/EVENTS.jsonl',[])
status=objects['coordinator/STATUS.json']
own_epochs={c:objects.get(c+'/EPOCHS_A.json',[]) for c in a['cases']}
checks=dict(admissions=0,stage_records=0,complete_train_records=0,formal_records=0,formal_vectors=0)
pairing={}
for case,count in a['cases'].items():
    inp=J(R/'cases'/case/'INPUT.json')
    for stage in count['admitted_stages']:
        ad=objects[case+'/ADMISSION_A_%d.json'%stage]
        for key in ['direct_bc_ifa_first_update_full_state_exact','direct_bc_ifa_second_update_full_state_exact','roundtrip_exact','next_step_exact','restore_initial_exact']:
            assert ad[key] is True,(case,stage,key)
        checks['admissions']+=1
    for stage in range(6,10):
        key=case+'/STAGE_A_%d.json'%stage
        if key in objects:
            row=objects[key]
            assert len([x for x in own_epochs[case] if x['stage']==stage])==100
            assert len(row['sha256'])==64
            checks['stage_records']+=1
    tr=objects.get(case+'/TRAIN_A.json')
    if tr:
        assert set(tr['checkpoints'])==set(map(str,range(6,10)))
        for stage in range(6,10):
            assert tr['checkpoints'][str(stage)]==objects[case+'/STAGE_A_%d.json'%stage]['sha256']
        checks['complete_train_records']+=1
    out=objects.get(case+'/OUTCOMES.json',[])
    for row in out:
        stage,task,index=row['stage'],row['task'],row['initial_index']
        assert row['attempt']==1 and row['cp_sha256']==objects[case+'/STAGE_A_%d.json'%stage]['sha256']
        assert type(row['success']) is bool and 1<=row['steps']<=600
        assert row['termination']==('success' if row['success'] else 'timeout600')
        assert row['success'] or row['steps']==600
        assert row['env_seed']==inp['seed']+index//10
        eps=np.random.RandomState(20261024+inp['seed']*1000+task*100+index).standard_normal((600,7)).astype(np.float32)
        assert row['noise_sha256']==hashlib.sha256(eps.tobytes()).hexdigest()
        key=(inp['suite'],inp['seed'],task,index)
        identity=(row['initial_file_sha256'],row['initial_sha256'],row['noise_sha256'])
        assert key not in pairing or pairing[key]==identity,('Paired identity mismatch',case,key)
        pairing[key]=identity;checks['formal_records']+=1
    ev=objects.get(case+'/EVAL_EVENTS.jsonl',[])
    starts={};completed=[]
    for row in ev:
        if row['kind']=='start':
            assert row['run'] not in starts;starts[row['run']]=row
        else:
            assert row['kind']=='complete' and row['run'] in starts and row['run'] not in completed
            assert row['seconds']>0;completed.append(row['run'])
    assert len(out) in [len(completed)*10,len(completed)*10+10],('Snapshot advance beyond one vector',case)
    checks['formal_vectors']+=len(completed)

tz=datetime.timezone(datetime.timedelta(hours=8))
observed=datetime.datetime.fromtimestamp(e['observed'],tz)
own_completed=[]
for case,rows in own_epochs.items():
    ex=[x for x in events if x['kind']=='train_exit' and x['case']==case and x['returncode']==0]
    if ex:
        start=next(x['time'] for x in events if x['kind']=='train_dispatch' and x['case']==case)
        wall=(ex[0]['time']-start)/60
        own_completed.append(dict(case=case,wall_minutes=wall,pure_epoch_minutes=sum(x['seconds'] for x in rows)/60))

eta={};ready_final=a['controller_exit']=='0' and not a['processes']
if ready_final:
    phase='READY_FINAL_CPU';next_at=None
elif status['status']=='PARTIAL_TRAINING':
    phase='CURRENT_COMPLETE_TRAINING_WAVE'
    active=list(status['active']);assert active
    overhead=statistics.mean(max(0,x['wall_minutes']-x['pure_epoch_minutes']) for x in own_completed) if own_completed else 8
    for case in active:
        inp=J(R/'cases'/case/'INPUT.json');rows=own_epochs[case];remaining=0;rates={}
        for stage in range(6,10):
            current=[x for x in rows if x['stage']==stage]
            if len(current)==100:continue
            donor=current
            source=case
            if not donor:
                candidates=[(c,[x for x in v if x['stage']==stage]) for c,v in own_epochs.items() if c.split('_')[1]==inp['variant'] and any(x['stage']==stage for x in v)]
                assert candidates,('No throughput reference',case,stage)
                candidates.sort(key=lambda x:(not x[0].startswith(inp['suite']+'_'),-len(x[1])))
                source,donor=candidates[0]
            per_batch=statistics.mean(x['seconds']/x['batches'] for x in donor[-10:])
            man=J(R/'cases'/case/'manifests'/('task%d.json'%stage));batches=(man['population']+9)//10
            stage_minutes=(100-len(current))*batches*per_batch/60
            remaining+=stage_minutes
            rates[str(stage)]=dict(complete=len(current),source=source,seconds_per_batch=per_batch,remaining_pure_minutes=stage_minutes,borrowed=source!=case)
        preparation=overhead*(400-len(rows))/400+2
        eta[case]=dict(stages=rates,pure_minutes=remaining,preparation_save_exit_allowance_minutes=preparation,total_minutes=remaining+preparation)
    wait_minutes=math.ceil(max(x['total_minutes'] for x in eta.values()))
    next_at=observed+datetime.timedelta(minutes=wait_minutes)
elif status['status']=='PARTIAL_EVALUATING':
    phase='ENTIRE_CURRENT_SUITE_FORMAL_EVALUATION'
    suite=status['case'].split('_')[0]
    cs=[c for c in a['cases'] if c.startswith(suite+'_')]
    durations=[x['seconds'] for c in cs for x in objects.get(c+'/EVAL_EVENTS.jsonl',[]) if x['kind']=='complete']
    remaining_vectors=(1200-sum(a['cases'][c]['formal'] for c in cs))/10
    mean=statistics.mean(durations) if durations else 46
    # Whole-suite mean extrapolation plus case setup allowance; not a guaranteed bound.
    pending=sum(a['cases'][c]['formal']<200 for c in cs)
    minutes=(remaining_vectors*mean)/60+pending*1.5
    wait_minutes=math.ceil(minutes*1.1)
    next_at=observed+datetime.timedelta(minutes=wait_minutes)
    eta=dict(suite=suite,completed_vectors=len(durations),own_mean_seconds=mean if durations else None,borrowed_seconds=46 if not durations else None,remaining_vectors=remaining_vectors,linear_minutes_with_setup=minutes,next_checkpoint_minutes=wait_minutes)
else:
    raise RuntimeError(('Unexpected coordinator state; preserve evidence and diagnose',status))

report=dict(status='PARTIAL_CHECKPOINT_INTEGRITY_PASS' if not ready_final else 'READY_FINAL_CPU_NOT_YET_ACCEPTED',observed=observed.isoformat(),checks=checks,recorded_initial_and_noise_pairing=True,initial_physical_bytes_deferred_to_final=True,
            own_complete_training=own_completed,phase=phase,eta=eta,next_checkpoint_CST=next_at.isoformat() if next_at else None,background_automation_created=False,new_optimizer=0,new_environments=0)
(d/'CHECKPOINT_AUDIT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
(R/'TRACKING_STATUS.json').write_text(json.dumps(dict(report,evidence_directory=str(d)),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rows='\n'.join('|%s|%d/400|%d/200|'%(c,x['epochs'],x['formal']) for c,x in a['cases'].items())
body='''# 当前任务检查与下一检查点

实际观察：%s，PARTIAL。进度前缀审计和独立轻量记录复算通过。真实代码合同、训练/评估退出与准入记录已核；工程失败0。当前科学状态尚未最终接受。

|run|完整训练轮|正式回合|
|---|---:|---:|
%s

当前检查范围：%s。下一检查点：%s（Asia/Shanghai）。按当前完整训练波次或整个套件正式评估余时估计，不因内部Task切换短周期检查。自身完成训练墙钟、纯epoch、借用吞吐与保存准备余量见CHECKPOINT_AUDIT.json；未实测部分不视为完成保证。

核验%d份实际阶段准入、%d份阶段CP记录绑定、%d份完整训练账；正式%d条记录及%d个完成vector。初态/噪声身份采用记录交叉比较与噪声独立重生成，真实初态文件字节和实际CP加载待全队列结束后的CPU验收。

证据：%s。跟踪仅在当前会话执行，没有创建后台定时自动化。没有新增优化器更新、环境回合、服务器科学写入或重启已完成工作。每轮发布以PUBLICATION_RECEIPT为准。
'''%(observed.isoformat(),rows,phase,next_at.isoformat() if next_at else '立即进行最终CPU验收',checks['admissions'],checks['stage_records'],checks['complete_train_records'],checks['formal_records'],checks['formal_vectors'],d.name)
(d/'CHECKPOINT_REPORT.md').write_text(body,encoding='utf-8');(R/'REPORT.md').write_text(body,encoding='utf-8')
with (R/'MONITOR_HANDOFF.md').open('a',encoding='utf-8') as f:f.write('\n\n'+body)
s=J(R/'RUN_STATUS.json');s['checkpoint']=report;(R/'RUN_STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,ensure_ascii=False))
