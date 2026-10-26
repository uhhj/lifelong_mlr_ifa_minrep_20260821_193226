"""Recompute the saved seed42-to-43 transition; no remote execution."""
import datetime as dt
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parent
D=R/'check_readonly_20261026_210507'
CAP=R/'capacity_readonly_20261026_210529'
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
v=read(D/'EVIDENCE.json');a=read(D/'LOCAL_AUDIT.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and a['phase']['seed']==43
L={n:json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in v['light'].items()}
old=read(R/'check_readonly_20261026_191436/EVIDENCE.json')
completed={}
for seed in (41,42):
    prefix=f'seed{seed}/';h=L[prefix+'EPOCHS_S.json'];train=L[prefix+'TRAIN_S.json']
    previous=json.loads(old['light'][prefix+'EPOCHS_S.json']['text'])
    assert h[:len(previous)]==previous
    assert L[prefix+'train_S.exit'].strip()=='0'
    assert (train['epochs'],train['steps'],train['gradient_autograd_calls'])==(400,105500,72)
    assert abs(sum(x['seconds'] for x in h)-train['summed_epoch_seconds'])<1e-7
    for t in range(6,10):
        stage=L[prefix+f'STAGE_S_{t}.json']
        assert stage['epochs']==100 and train['checkpoints'][str(t)]==stage['sha256']
        assert L[prefix+f'collect_S_{t}.exit'].strip()=='0'
        assert L[prefix+f'COLLECTION_STATE_PARITY_S_{t}.json']['parent_all_state_exact']
    events=[json.loads(x) for x in L[prefix+'COLLECTION_EVENTS.jsonl'].splitlines()]
    assert [(e['kind'],e['task']) for e in events]==[(k,t) for t in range(6,10) for k in ('start','complete')]
    completed[str(seed)]=dict(epoch_minutes={str(t):sum(x['seconds'] for x in h if x['stage']==t)/60 for t in range(6,10)},
        collection_minutes=sum(e['seconds'] for e in events if e['kind']=='complete')/60,
        checkpoints=train['checkpoints'],previous_prefix_epochs=len(previous),prefix_exact=True,exit_zero=True)
for seed in (41,42,43):
    p=L[f'seed{seed}/TASK6_PARITY_S.json']
    assert p['historical_ALIGN_equal'] and p['actual_sha256']==L[f'seed{seed}/STAGE_S_6.json']['sha256']
assert a['seeds']['43']['counts']['S']['epochs']==100
assert a['seeds']['43']['training_collection_episodes']==0
reference=sum(completed['42']['epoch_minutes'][str(t)] for t in (7,8,9))
eta=dict(scope='seed43 remaining whole training including collection',
         comparable_seed42_task7_9_compute_minutes=reference,
         comparable_seed42_four_collections_minutes=completed['42']['collection_minutes'],
         unmeasured_prepare_save_margin_minutes=10,next_check_minutes=115,
         seed43_auxiliary_throughput_not_measured=True,not_upper_bound=True,whole_queue_eta_known=False)
save(D/'TRAINING_TRANSITION_AUDIT.json',dict(status='PASS_PROGRESS_NOT_FINAL',observation=v['observed'],completed=completed,
    own_seed43_admissions=a['seeds']['43']['admissions'],historical_task6_server_tensor_equal_seeds=[41,42,43],
    seed43_collection_in_progress=True,independent_final_cpu_pending=True,eta=eta,evidence_sha256=sha(D/'EVIDENCE.json')))
c=read(CAP/'EVIDENCE.json')
for mount,row in c['capacity'].items():
    bound=6*1024**3 if mount=='/data' else 0
    assert row['remaining']==max(0,bound-row['allocated_unique'])
    assert row['reserve']==(2 if mount=='/data' else 1)*1024**3
    assert row['headroom']==row['free']-row['remaining']-row['reserve']>=0
assert all(c[k]==0 for k in ['payload_reads','server_writes','forwards','optimizer_updates','environments'])
save(CAP/'LOCAL_AUDIT.json',dict(status='PASS_NOT_FINAL',observation=c['observed'],capacity=c['capacity'],evidence_sha256=sha(CAP/'EVIDENCE.json')))
obs=dt.datetime.fromtimestamp(v['observed'],dt.timezone(dt.timedelta(hours=8))).isoformat()
s=read(R/'RUN_STATUS.json')
s.update(status='PARTIAL_TRAINING',observation=obs,latest_evidence=D.name,
    seeds={seed:dict(counts=x['counts']['S'],admissions=x['admissions'],memory=x['memory']) for seed,x in a['seeds'].items()},
    main_updates=a['total_main_updates'],training_collection_episodes=a['training_collection_episodes'],formal_episodes=a['formal_episodes'],
    isolated_optimizer_updates=a['isolated_optimizer_updates'],next_check_minutes=115,eta=eta,capacity_evidence=CAP.name,phase=a['phase'])
save(R/'RUN_STATUS.json',s)
active=read(R/'ACTIVE_ROOT.json');active.update(last_observation=obs,last_evidence=D.name);save(R/'ACTIVE_ROOT.json',active)
report=f'''# 最新2026-10-26 21:07 CST进度覆盖

实际观察{obs}，{D.name}/LOCAL_AUDIT及TRAINING_TRANSITION_AUDIT实际PASS_PROGRESS_NOT_FINAL。48冻结项和父源不变，controller15770/seed43训练18434及采集18465正常。seed41/42各完整400epoch105500主步、train_S.exit0、40训练采集，八CP记录及缓存源/父状态绑定通过。原快照seed41全部400轮及seed42前134轮逐字段精确保留，无重复训练或采集。seed42纯epoch{sum(completed['42']['epoch_minutes'].values()):.6f}分钟、采集{completed['42']['collection_minutes']:.6f}分钟；保存/准备/准入另计，72额外梯度autograd单列。

seed43已完成Task6全部100epoch15700步，自己的4隔离GPU更新、完整下一步恢复/原ALIGN更新及短历史1–8准入PASS；当前Task6训练采集进行中，尚无完整十回合采集账本，不将启动当完成。旧阶段自己的准入待运行，不能套其他seed。合计900epoch226700主步、80完整训练采集、48自身隔离更新；正式0/600。三个seed Task6服务器tensor与各自历史ALIGN相等记录绑定，最终CPU验收仍待全部结束。

seed42固定缓存13/115/128/115共371窗口，seed41原345窗口保持；成功筛选缺额不补采不重分配。seed43缓存尚未完成，不推断覆盖或性能。没有恢复重启、改参、删除或付费。

seed43后续辅助阶段速度未实测，借刚完成seed42 Task7–9纯计算{reference:.6f}分钟、四次采集{completed['42']['collection_minutes']:.6f}分钟作可比参考；另加10分钟未实测保存/准备余量。按用户整段完成点规则，同一监控下一115分钟检查seed43整组完成附近，暂参考22:55–23:10 CST。此后才600正式，正式及最终验收ETA尚不可靠；不把上述参考当上界，不内部Task短轮询。

{CAP.name}/LOCAL_AUDIT实际PASS_NOT_FINAL：system普通留1GiB后余372510720B；data唯一inode抵扣余新增4750311424B且留2GiB后余41740230656B。只stat、0payload/前向/更新/环境/服务器写入。全部自然controller.exit0且无自身进程后才最终CPU/初态字节/缓存重放验收；外置最终路径仍待真实调用，整体PARTIAL。

本段覆盖下文旧进度和110分钟间隔，科学协议与限制不变。

'''
for name in ('REPORT.md','MONITOR_HANDOFF.md'):
    p=R/name;p.write_text(report+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(dict(completed=completed,eta=eta),indent=2))
