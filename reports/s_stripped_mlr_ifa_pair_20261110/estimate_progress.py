"""Estimate finite remaining work from audited lightweight evidence only."""
from pathlib import Path
import datetime
import hashlib
import json
import statistics
import sys

R = Path(__file__).resolve().parent
d = Path(sys.argv[1]).resolve()
J = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
e = J(d / 'EVIDENCE.json')
a = J(d / 'LOCAL_AUDIT.json')
assert a['status'] == 'PASS_PROGRESS_NOT_FINAL'
assert a['prefix_preserved'] and a['controller_exit'] is None
tz = datetime.timezone(datetime.timedelta(hours=8), 'Asia/Shanghai')
observed = datetime.datetime.fromtimestamp(e['observed'], tz)
L = e['light']
events = [json.loads(x) for x in L['coordinator/EVENTS.jsonl']['text'].splitlines() if x]
measurements = {}
for case in ['OBJECT_ALIGN_41', 'OBJECT_MLR_41']:
    rows = json.loads(L[case + '/EPOCHS_A.json']['text'])
    start = next(x['time'] for x in events if x['kind'] == 'train_dispatch' and x['case'] == case)
    stages = {}
    for stage in range(6, 10):
        items = [x for x in rows if x['stage'] == stage]
        assert items
        recent = items[-10:]
        stages[str(stage)] = dict(
            complete_epochs=len(items),
            batches_per_epoch=items[-1]['batches'],
            epoch_seconds_sum=sum(x['seconds'] for x in items),
            recent_epoch_seconds=[x['seconds'] for x in recent],
            recent_mean_seconds=statistics.mean(x['seconds'] for x in recent))
    remaining = (100 - stages['9']['complete_epochs']) * stages['9']['recent_mean_seconds'] / 60
    # GOAL not yet observed: reuse same stage's seconds/batch, changing only the known batch count.
    goal_batches = {'6': 100, '7': 139, '8': 185, '9': 337}
    goal_pure = sum(100 * stages[s]['recent_mean_seconds'] / stages[s]['batches_per_epoch'] * n
                    for s, n in goal_batches.items()) / 60
    measurements[case] = dict(stages=stages,
                              dispatch_to_observation_minutes=(e['observed'] - start) / 60,
                              completed_pure_epoch_minutes=sum(x['seconds'] for x in rows) / 60,
                              current_pair_own_remaining_pure_minutes=remaining,
                              own_full_OBJECT_pure_projection_minutes=sum(100 * x['recent_mean_seconds'] for x in stages.values()) / 60,
                              unmeasured_GOAL_pure_projection_minutes=goal_pure)

ref = R.parent / '20261109_mlr_ifa_align_baseline/engineering_cached_eval_20261110_v1/check_readonly_20261110_194749/EVIDENCE.json'
ref_raw = ref.read_bytes()
ref_e = json.loads(ref_raw)
ref_events = [json.loads(x) for x in ref_e['light']['cached_engineering/EVENTS.jsonl']['text'].splitlines() if x]
ref_eval = {}
for case in ['OBJECT_43', 'GOAL_41', 'GOAL_42', 'GOAL_43']:
    lo = next(x['time'] for x in ref_events if x['kind'] == 'launch' and x['phase'] == 'evaluate' and x['case'] == case)
    hi = next(x['time'] for x in ref_events if x['kind'] == 'exit' and x['phase'] == 'evaluate' and x['case'] == case)
    ref_eval[case] = (hi - lo) / 60

components = [
    dict(work='Current OBJECT seed41 pair, including remaining save/exit allowance', minutes=[10, 15], basis='own last 10 stage9 epochs; save/exit allowance is planning'),
    dict(work='OBJECT seed42 and seed43 two parallel training waves', minutes=[150, 180], basis='current OBJECT stage rates; stage preparation/save/admission allowance is planning'),
    dict(work='OBJECT 1200 formal episodes', minutes=[90, 120], basis='recent comparable cached evaluations; this run has no own formal throughput yet'),
    dict(work='GOAL three parallel training waves', minutes=[165, 210], basis='own OBJECT seconds/batch scaled by GOAL stage batch counts; unmeasured GOAL and preparation allowance'),
    dict(work='GOAL 1200 formal episodes', minutes=[90, 120], basis='recent comparable cached evaluations; this run has no own formal throughput yet')]
mins = [sum(x['minutes'][i] for x in components) for i in range(2)]
finish = [(observed + datetime.timedelta(minutes=n)).isoformat() for n in mins]
cap = e['capacity']
capacity = dict(root_margin_after_1GiB_plus_64MiB_bytes=cap['root_free']-1024**3-64*1024**2,
                data_margin_after_remaining_peak_and_2GiB_bytes=cap['data_free']-cap['remaining_bound']-2*1024**3)
assert min(capacity.values()) >= 0
result = dict(status='PARTIAL_ETA_NOT_COMPLETION_GUARANTEE', observed=observed.isoformat(),
              evidence_sha256=hashlib.sha256((d/'EVIDENCE.json').read_bytes()).hexdigest(),
              audit='PASS_PROGRESS_NOT_FINAL', own_measurements=measurements,
              comparable_evaluation_reference=dict(path=str(ref), sha256=hashlib.sha256(ref_raw).hexdigest(),
                                                     observed_minutes_per_200_episodes=ref_eval,
                                                     limitation='Similar cached evaluation, different evaluator details/policies; throughput is an uncertain reference, not equivalence.'),
              planning_components=components, remaining_total_minutes=mins,
              approximate_finish_CST=finish, excludes='Final independent CPU acceptance, analysis and publication',
              capacity=capacity, new_optimizer=0, new_environments=0, server_mutations=0)
(d/'TIMING_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
active_rows='\n'.join('|%s|%s/400|%s/100|%s|'%(case,a['cases'][case]['epochs'],measurements[case]['stages']['9']['complete_epochs'],a['cases'][case]['production']) for case in measurements)
body=f'''# 配对实验进度与时间估计

状态：PARTIAL。实际观察时间：{observed.isoformat()}。

本次通过已授权凭据通道完成一次服务器只读检查；未启动训练、环境或恢复操作。`LOCAL_AUDIT.json` 实际通过，上一快照的全部训练前缀保留。冻结代码与合同一致，controller55246、两个训练launcher及GPU worker正常，未记录失败。GPU快照为4080 SUPER、89%利用率、3691MiB占用；该利用率为单次观测，不是时间平均。

|当前运行|完整训练轮|Task9完整轮|生产更新|
|---|---:|---:|---:|
{active_rows}

其余十个run尚未开始；正式评估0/2400。合计完成{a['totals']['epochs']}/4800轮、{a['totals']['production']}/1089600生产更新与40/240隔离准入更新，0训练采集。四阶段自身准入均已实际通过的仅当前两run；后续准入不预称通过。实验尚未最终接受。

当前两run Task9最近10轮平均分别为{measurements['OBJECT_ALIGN_41']['stages']['9']['recent_mean_seconds']:.3f}和{measurements['OBJECT_MLR_41']['stages']['9']['recent_mean_seconds']:.3f}秒，纯训练剩余约{measurements['OBJECT_ALIGN_41']['current_pair_own_remaining_pure_minutes']:.2f}和{measurements['OBJECT_MLR_41']['current_pair_own_remaining_pure_minutes']:.2f}分钟。计保存与自然退出规划，当前这一对预计22:00前后结束。

后续顺序为OBJECT seed42/43训练、OBJECT1200正式、GOAL六run训练、GOAL1200正式。按当前自身吞吐、GOAL已知batch人口缩放和最近可比cached评估实测，规划余量如下；未实测部分不能作为准确ETA或保证上界。

|剩余部分|规划分钟|
|---|---:|
|当前OBJECT seed41两run|10–15|
|OBJECT另外四run，两个并行波次|150–180|
|OBJECT1200正式|90–120|
|GOAL六run，三个并行波次|165–210|
|GOAL1200正式|90–120|

全队列训练和正式评估余约505–645分钟（8小时25分至10小时45分），对应北京时间11月11日06:13–08:33附近；独立CPU验收、分析和发布另计。当前OBJECT全部训练预计还需160–195分钟；不把当前seed41这一对结束与整套件训练或全队列完成混淆。GOAL与本轮正式吞吐未实测，后续应依据真实阶段速度更新估计。

容量按稳定已分配量抵扣新峰值后仍满足system1GiB+64MiB/data2GiB保留门槛，详见TIMING_AUDIT。没有删除资产或修改预算。

证据：`{d.name}/EVIDENCE.json`、`LOCAL_AUDIT.json`、`TIMING_AUDIT.json`。发布状态以`PUBLICATION_RECEIPT.json`为准，发布进度不等于科学接受。
'''
(d/'PROGRESS_ETA_REPORT.md').write_text(body,encoding='utf-8')
(R/'REPORT.md').write_text(body,encoding='utf-8')
s=J(R/'RUN_STATUS.json')
s['eta']=dict(remaining_minutes=mins,approximate_finish_CST=finish,uncertain=True,final_acceptance_excluded=True,timing_audit=str(d/'TIMING_AUDIT.json'))
(R/'RUN_STATUS.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
with (R/'MONITOR_HANDOFF.md').open('a',encoding='utf-8') as f:
    f.write('\n## 本次估时\n\n详见REPORT.md与'+d.name+'/TIMING_AUDIT.json。全队列训练及2400正式剩余约8小时25分–10小时45分，参考11月11日06:13–08:33 CST；未实测GOAL及正式用不确定参考，CPU最终验收另计。当前seed41两run约22:00完成，不能当成全队列结束。\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
