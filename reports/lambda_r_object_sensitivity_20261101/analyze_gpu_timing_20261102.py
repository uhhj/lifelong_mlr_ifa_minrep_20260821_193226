"""Read-only local recomputation of measured GPU window and conditional queue ETA."""
from pathlib import Path
import json, hashlib, statistics, datetime

R = Path(__file__).resolve().parent
D = R / 'check_readonly_20261102_140853'
G = R / 'capacity_readonly_20261102_140929'
H = R.parent / '20261028_support_gate_full_s_object/final_readonly_20261029_201235/light'
read = lambda p: json.loads(p.read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf8')

a = read(D/'LOCAL_AUDIT.json')
assert a['status'] == 'PASS_PROGRESS_NOT_FINAL' and not a['failures']
assert a['snapshot_sha256'] == sha(D/'EVIDENCE.json')
g = read(G/'EVIDENCE.json')
assert g['status']=='PASS_READONLY_RESOURCE_WINDOW'
assert all(g[k]==0 for k in ['optimizer_calls','environments','signals','production_writes'])
assert len(g['samples'])==30 and all(s['exit']==0 for s in g['samples'])
rows = [[float(y.strip()) for y in s['stdout'].strip().split(',')[:-1]] for s in g['samples']]
stats = {k:dict(mean=statistics.mean(x[i] for x in rows),
                 median=statistics.median(x[i] for x in rows),
                 minimum=min(x[i] for x in rows),maximum=max(x[i] for x in rows))
         for i,k in enumerate(g['fields'][:-1])}
before = next(p for p in g['processes_before'] if p['pid']==2566)
after = next(p for p in g['processes_after'] if p['pid']==2566)
assert before['start']==after['start']
seconds = g['observed_end']-g['observed_start']
cpu_pct = 100*(after['cpu_ticks']-before['cpu_ticks'])/g['cpu_clock_ticks']/seconds
gpu_audit = dict(status='PASS_READONLY_GPU_WINDOW_NOT_HARDWARE_CERTIFICATION',
    evidence_sha256=sha(G/'EVIDENCE.json'),sample_count=30,window_seconds=seconds,
    stats=stats,training_cpu_percent_one_core=cpu_pct,
    phase_before=g['training_before'],phase_after=g['training_after'],
    limitations=['30 one-second samples, not an all-run average.',
      'End-of-window nvidia-smi reports no active thermal/power/hardware slowdown; does not prove absence of all hardware faults.',
      'About one CPU core occupied plus low GPU utilization is consistent with serial host work/synchronization/small kernels, not causal localization.',
      'No training changes, benchmarks, optimizer calls, environments, or signals.'])
dump(G/'LOCAL_AUDIT.json',gpu_audit)

epath=D/'light/L005_41/EPOCHS_S.json'; epochs=read(epath)
stage8=[x['seconds'] for x in epochs if x['stage']==8 and x['epoch']>=60]
stage9=[x['seconds'] for x in epochs if x['stage']==9]
assert len(stage8)==40 and len(stage9)==39
t8=statistics.mean(stage8);t9=statistics.mean(stage9[-10:])
# Conditional batch-count model. Task6 has only one candidate, with no old-task retention.
# Its estimate is deliberately labeled extrapolated, not a measured 3090 duration.
t6=t8*157/(2*309);t7=t8*227/309
remaining_current=61*t9/3600
future_run=100*(t6+t7+t8+t9)/3600
pure=remaining_current+5*future_run
hist=[]
for seed in [41,42,43]:
    p=H/f'seed{seed}/EVAL_EVENTS.jsonl'
    events=[json.loads(x) for x in p.read_text().splitlines()]
    complete=[x for x in events if x['kind']=='complete']
    assert len(complete)==20 and complete[-1]['episodes']==200
    hist.append(dict(seed=seed,source=str(p),sha256=sha(p),vector_seconds=sum(x['seconds'] for x in complete)))
historical600=sum(x['vector_seconds'] for x in hist)/60
observed=datetime.datetime.fromisoformat(a['observation'])
timing=dict(status='PASS_CONDITIONAL_PLANNING_NOT_COMPLETION',observation=a['observation'],
    snapshot_sha256=sha(D/'EVIDENCE.json'),epochs_sha256=sha(epath),
    current_run='L005_41',audited_epochs=339,resource_window_status_epochs=340,
    other_full_runs_remaining=5,formal_remaining=1200,
    stage8_post_recovery=dict(n=40,mean_seconds=t8,recent10_seconds=statistics.mean(stage8[-10:])),
    stage9=dict(n=39,mean_seconds=statistics.mean(stage9),recent10_seconds=t9,
                recent10_min=min(stage9[-10:]),recent10_max=max(stage9[-10:])),
    unmeasured_future_stage6_seconds=t6,unmeasured_future_stage7_seconds=t7,
    extrapolation_formula='Task6=t8*157/(2*309); Task7=t8*227/309; future run=100*(t6+t7+t8+t9). No GPU-utilization multiplier.',
    current_run_pure_hours=remaining_current,future_run_pure_hours=future_run,
    all_remaining_pure_training_hours=pure,
    current_run_training_and_collection_plan_minutes=[75,95],
    training_including_preparation_saving_210_remaining_collection_plan_hours=[27,33],
    historical_evaluation_sources=hist,historical_600_vector_minutes=historical600,
    historical_1200_vector_hours=historical600*2/60,
    new_hardware_1200_formal_plan_hours=[2,4],
    total_training_and_formal_plan_hours=[29,37],
    planned_completion_window=[(observed+datetime.timedelta(hours=h)).isoformat() for h in [29,37]],
    next_check_minutes=75,
    assumptions=['Future five runs have not started; Task6/7 on 3090 are unmeasured. Batch-count extrapolation ignores changing old-cache/probe costs.',
      '27–33h training allowance rounds the conditional pure-training estimate upward for preparation/saving/210 remaining collection episodes and roughly 25% resource/cache variation; this is a planning interval, not a confidence interval.',
      'The .2 group has not been measured; its own successful caches may differ. Fixed dual-candidate structure does not make all stage runtimes equal.',
      'Formal evaluation on current 3090 is not measured. Historical 600 episodes reused only as timing reference; 1200 formal need 2–4h provisionally.',
      'The 29–37h window assumes current throughput persists without further interruption. CPU acceptance/report/publication additional.',
      'Recovery changed host/GPU and snapshot path together; no causal GPU-only speedup claim; original partial cost 140 recorded plus UNKNOWN remains.'])
dump(D/'TIMING_AUDIT.json',timing)
report=f'''# 2026-11-02 14:09 GPU利用率与六run余时

当前仍PARTIAL。14:08:53完整审计L005_41为339/400轮，Task9为39/100，其他五run未开始；30秒资源采样末训练状态到340轮，此为状态观察，完整逐轮审计仍以339轮快照为准。采集30/240、正式0/1200。

GPU连续30个1秒采样：利用率均{stats['utilization.gpu']['mean']:.1f}%、范围0–43%，显存2872/24576MiB；温度56–62℃，功率均{stats['power.draw']['mean']:.1f}W、范围141.87–216.95W（上限350W），SM频率1695–1950MHz。训练进程占用约{cpu_pct:.1f}%单CPU核心；采样末无active thermal/power/hardware slowdown。GPU未持续满载；串行CPU调度/同步/小算子可能相关，仅凭此窗口不能确定具体占比，也不能诊断或排除全部硬件故障。没有新增训练、环境、信号或参数修改。

3090恢复后Task8的40轮平均{t8:.2f}秒；Task9的39轮平均{statistics.mean(stage9):.2f}秒、最近10轮{t9:.2f}秒（{min(stage9[-10:]):.2f}–{max(stage9[-10:]):.2f}）。当前run余61轮纯计算{remaining_current*60:.1f}分钟，连保存和最后10采集规划75–95分钟。

未来五run各400轮。未测Task6按单候选批次数比例估{t6:.2f}秒/轮，Task7按Task8批次数比例估{t7:.2f}秒/轮；Task8/9使用上述实测。条件外推每完整run纯计算{future_run:.2f}小时，总余纯训练{pure:.2f}小时。这不是未来五run的实测吞吐，旧任务缓存、探针、.2组及资源差异可能改变结果。

计准备/保存/剩余210次采集及波动，训练规划还需27–33小时；1200正式闭环另留2–4小时。历史同实现OBJECT600正式vector合计{historical600:.2f}分钟，仅作参考，新3090正式尚未测得。合计规划29–37小时，即11月3日19:10至11月4日03:10附近；CPU独立验收、解释报告和发布另计。不是置信区间或完成保证，后续首个完整新run实测后更新。下一75分钟在当前run结束较早侧复查。

135科学/5旧工程/8恢复工程项与原前缀精确保留，Task9自身准入及工程复制核验通过，controller2560/GPU2566正常。工程诊断8137optimizer/47autograd、原中断140已记账候选+UNKNOWN单列，硬件变更授权及科学可比性限制保留。详见当前快照TIMING_AUDIT.json与资源窗口LOCAL_AUDIT.json。
'''
(R/'GPU_TIMING_REPORT_20261102_1409.md').write_text(report,encoding='utf8')
print(json.dumps(dict(gpu_mean=stats['utilization.gpu']['mean'],cpu_percent=cpu_pct,pure_training_hours=pure,historical600_minutes=historical600,total_plan=[29,37]),ensure_ascii=False))
