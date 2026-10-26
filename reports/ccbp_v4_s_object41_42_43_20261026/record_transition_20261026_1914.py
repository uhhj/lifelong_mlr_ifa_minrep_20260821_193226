"""Audit the saved seed41-to-42 transition and record this observation only."""
import datetime as dt
import hashlib
import json
from pathlib import Path

R = Path(__file__).resolve().parent
D = R / 'check_readonly_20261026_191436'
CAP = R / 'capacity_readonly_20261026_191459'
read = lambda p: json.loads(p.read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

v = read(D/'EVIDENCE.json')
a = read(D/'LOCAL_AUDIT.json')
assert a['status'] == 'PASS_PROGRESS_NOT_FINAL' and a['phase']['seed'] == 42
L = {n: json.loads(x['text']) if n.endswith('.json') else x['text'] for n,x in v['light'].items()}
assert L['seed41/train_S.exit'].strip() == '0'
h = L['seed41/EPOCHS_S.json']
old = read(R/'check_readonly_20261026_165809/EVIDENCE.json')
prefix = json.loads(old['light']['seed41/EPOCHS_S.json']['text'])
assert h[:len(prefix)] == prefix
train = L['seed41/TRAIN_S.json']
assert (train['epochs'], train['steps'], train['gradient_autograd_calls']) == (400, 105500, 72)
assert abs(sum(x['seconds'] for x in h)-train['summed_epoch_seconds']) < 1e-7
for t in range(6,10):
    stage = L[f'seed41/STAGE_S_{t}.json']
    assert stage['epochs'] == 100 and train['checkpoints'][str(t)] == stage['sha256']
    assert L[f'seed41/collect_S_{t}.exit'].strip() == '0'
    assert L[f'seed41/COLLECTION_STATE_PARITY_S_{t}.json']['parent_all_state_exact']
for seed in (41,42):
    parity = L[f'seed{seed}/TASK6_PARITY_S.json']
    assert parity['actual_sha256'] == L[f'seed{seed}/STAGE_S_6.json']['sha256']
    assert parity['historical_ALIGN_equal'] is True
events = [json.loads(x) for x in L['seed41/COLLECTION_EVENTS.jsonl'].splitlines()]
assert [(e['kind'],e['task']) for e in events] == [(k,t) for t in range(6,10) for k in ('start','complete')]
collection_minutes = sum(e['seconds'] for e in events if e['kind']=='complete')/60
stage_minutes = {str(t): sum(x['seconds'] for x in h if x['stage']==t)/60 for t in range(6,10)}
remaining7 = (100-a['seeds']['42']['counts']['S']['by_stage']['7'])*a['seeds']['42']['rates_last10']['S']['7']/60
remaining_compute = remaining7+stage_minutes['8']+stage_minutes['9']
eta = dict(scope='seed42 whole training including remaining collections',
           own_remaining_task7_minutes=remaining7,
           comparable_seed41_task8_9_minutes=stage_minutes['8']+stage_minutes['9'],
           remaining_compute_reference_minutes=remaining_compute,
           unmeasured_prepare_collection_save_margin_minutes=10,
           next_check_minutes=110, not_upper_bound=True, whole_queue_eta_known=False)
transition = dict(status='PASS_PROGRESS_NOT_FINAL', observation=v['observed'],
                 completed_seed41_epoch_minutes=stage_minutes,
                 completed_seed41_collection_minutes=collection_minutes,
                 inherited_snapshot_prefix_epochs=len(prefix), prefix_exact=True,
                 completed_seed41_exit_zero=True, checkpoints_bound=train['checkpoints'],
                 own_seed42_admissions=a['seeds']['42']['admissions'],
                 historical_task6_server_tensor_equal_seeds=[41,42],
                 independent_final_cpu_pending=True, eta=eta,
                 evidence_sha256=sha(D/'EVIDENCE.json'))
save(D/'TRAINING_TRANSITION_AUDIT.json',transition)
c = read(CAP/'EVIDENCE.json')
for mount,row in c['capacity'].items():
    bound = 6*1024**3 if mount=='/data' else 0
    assert row['remaining']==max(0,bound-row['allocated_unique'])
    assert row['reserve']==(2 if mount=='/data' else 1)*1024**3
    assert row['headroom']==row['free']-row['remaining']-row['reserve']>=0
assert all(c[k]==0 for k in ['payload_reads','server_writes','forwards','optimizer_updates','environments'])
save(CAP/'LOCAL_AUDIT.json',dict(status='PASS_NOT_FINAL',observation=c['observed'],capacity=c['capacity'],evidence_sha256=sha(CAP/'EVIDENCE.json')))
obs = dt.datetime.fromtimestamp(v['observed'],dt.timezone(dt.timedelta(hours=8))).isoformat()
s = read(R/'RUN_STATUS.json')
s.update(status='PARTIAL_TRAINING',observation=obs,latest_evidence=D.name,
         seeds={seed:dict(counts=x['counts']['S'],admissions=x['admissions'],memory=x['memory']) for seed,x in a['seeds'].items()},
         main_updates=a['total_main_updates'],training_collection_episodes=a['training_collection_episodes'],
         formal_episodes=a['formal_episodes'],isolated_optimizer_updates=a['isolated_optimizer_updates'],
         next_check_minutes=110,eta=eta,capacity_evidence=CAP.name,phase=a['phase'])
save(R/'RUN_STATUS.json',s)
active = read(R/'ACTIVE_ROOT.json'); active.update(last_observation=obs,last_evidence=D.name)
save(R/'ACTIVE_ROOT.json',active)
report = f'''# 最新2026-10-26 19:17 CST进度覆盖

实际观察{obs}，{D.name}/LOCAL_AUDIT及TRAINING_TRANSITION_AUDIT实际PASS_PROGRESS_NOT_FINAL。48冻结项和父源不变，controller15770/GPU17115健康，已自然进入seed42 Task7。seed41完整400epoch105500主步、train_S.exit0和40训练采集全部结束，四CP记录、缓存源和四次父完整状态不变已绑定。纯epoch{sum(stage_minutes.values()):.6f}分钟，四采集{collection_minutes:.6f}分钟，保存/准备/准入另计；72正式额外梯度autograd另列。最终独立CPU和正式评估仍待全队列完成。

seed42已134完整epoch23418步（Task6 100、Task7 34），自己的Task6/7共10隔离GPU更新、完整下一步恢复、状态化测量及短历史1–8准入通过；10训练采集完成。seed43尚未开始。总534epoch128918主步、50训练采集、32自身隔离更新；正式0/600。seed41/42 Task6与各自历史ALIGN的服务器tensor相等记录绑定，不能代替最终CPU验收。无重启或改科学参数。

成功筛选缓存严格保留缺额：seed41各任务13/128/128/76，共345窗口；seed42 Task6为13。Task6两seed的10次训练采集各仅1条成功轨迹，所以按冻结quota各保留13窗口，不补采、不重分配。这是训练缓存覆盖事实，不能冒充正式成功率或提前归因为最终遗忘。

seed42 Task7最近10轮12.063894秒，余66轮{remaining7:.6f}分钟；Task8/9自身尚未测，借刚完成seed41对应阶段{stage_minutes['8']+stage_minutes['9']:.6f}分钟，剩余计算参考{remaining_compute:.6f}分钟。另加10分钟未实测采集/准备/保存余量，下一110分钟检查seed42整段结束附近，暂参考21:00–21:15 CST；不是seed43或全部600正式完成时间，也不是保证上界，不内部Task短轮询。

{CAP.name}/LOCAL_AUDIT实际PASS_NOT_FINAL：system普通留1GiB后余372547584B；data按唯一inode抵扣剩余6GiB界且留2GiB后余41741115392B。只stat，无payload/前向/更新/环境/服务器写入，无删除/付费。整体PARTIAL；全部自然退出后才运行最终独立验收，最终wrappers仍待真实调用。

以下历史进度和首次140分钟间隔由本段覆盖，原科学协议与限制不变。

'''
for name in ['REPORT.md','MONITOR_HANDOFF.md']:
    p=R/name; p.write_text(report+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(transition,ensure_ascii=False,indent=2))
