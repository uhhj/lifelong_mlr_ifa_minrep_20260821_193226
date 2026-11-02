"""Read-only production timing recomputation and explicit queue ETA assumptions."""
import json,statistics as st,hashlib,datetime,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parent;E=R/'engineering_live_parallel_3090_20261102_v1';D=R/'check_readonly_20261102_183513';O=R/'check_readonly_20261102_172140';G=R/'capacity_readonly_20261102_183609'
js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
v=js(D/'EVIDENCE.json');old=js(O/'EVIDENCE.json');a=js(D/'LOCAL_AUDIT.json');gpu=js(G/'EVIDENCE.json');assert a['status']=='PASS_PROGRESS_NOT_FINAL'
h=lambda case:json.loads(v['light'][case+'/EPOCHS_S.json']['text'])
h42=h('L005_42');h43=h('L005_43');h41=h('L005_41')
pre=[r['seconds'] for r in h42 if r['stage']==8 and 43<=r['epoch']<63];post=[r['seconds'] for r in h42 if r['stage']==8 and r['epoch']>=65]
rates={6:st.mean(r['seconds'] for r in h42 if r['stage']==6),7:st.mean(r['seconds'] for r in h42 if r['stage']==7),8:st.mean(pre),9:st.mean(r['seconds'] for r in h41 if r['stage']==9)}
work={k:0 for k in rates}
for c in ['L005_42','L005_43']:
 before=json.loads(old['light'][c+'/EPOCHS_S.json']['text']);assert h(c)[:len(before)]==before
 for r in h(c)[len(before):]:work[r['stage']]+=1
wall=v['observed']-old['observed'];reference=sum(work[k]*rates[k] for k in rates)
current={6:st.mean(r['seconds'] for r in h43 if r['stage']==6),7:st.mean(r['seconds'] for r in h43 if r['stage']==7),8:st.mean(post),9:st.mean([r['seconds'] for r in h42 if r['stage']==9][-10:])}
remaining42=66*current[9]/60;remaining43=(99*current[8]+100*current[9])/60
parallel_full=sum(current.values())*100/60;solo_full=sum(rates.values())*100/60
# Three jobs on a two-slot pool have a final one-job tail, not 3/2 of one run.
future_compute=(parallel_full+solo_full)/60
now=datetime.datetime.fromtimestamp(v['observed'],datetime.timezone(datetime.timedelta(hours=8)))
out=dict(status='PROVISIONAL_PRODUCTION_COMPARISON_NOT_CONTROLLED_AB',observation=a['observation'],sources={str(p.relative_to(R)):sha(p) for p in [D/'EVIDENCE.json',O/'EVIDENCE.json',G/'EVIDENCE.json',R/'GPU_TIMING_REPORT_20261102_1409.md']},same_seed_same_stage=dict(case='L005_42',stage=8,pre_epochs_one_based=[44,63],pre_count=len(pre),pre_seconds_mean=st.mean(pre),post_epochs_one_based=[66,100],post_count=len(post),post_seconds_mean=st.mean(post),per_epoch_slowdown_fraction=st.mean(post)/st.mean(pre)-1),production_interval=dict(wall_minutes=wall/60,complete_epochs_by_stage=work,reference_stage_seconds=rates,reference_serial_compute_minutes=reference/60,reference_work_per_wall_ratio=reference/wall,implied_time_reduction_fraction=1-wall/reference,meaning='Approximate stage-weighted observed throughput, not controlled serial/concurrent causal speedup'),current_stage_seconds=current,remaining_compute_minutes=dict(L00542=remaining42,L00543=remaining43),future_full_run_compute_minutes=dict(two_active_each=parallel_full,last_solo=solo_full,three_run_two_slot_makespan_hours=future_compute),planning_hours=dict(remaining_L005_training_collection=[3,4],L020_three_run_training_collection_two_slots=[9,11],both_groups_1200_formal=[2,4],total_training_collection_formal=[14,19]),planning_completion_range=[(now+datetime.timedelta(hours=t)).isoformat() for t in [14,19]],CPU_acceptance_report_publication='ADDITIONAL',next_check_minutes=90,limits=['Task7 reference includes earlier isolated validation contention; Task9 reference is another seed with older checks wrapper/different cache. Ratio1.71 is a rough normalization, not exact counterfactual.','Completed-epoch boundaries omit current partial work in both snapshots; preparation and three collections occur within the observed wall interval.','Current Task8/9 pairing differs from earlier Task8/Task6 and Task8/Task7. No uniform 1.71x multiplier is applied to all future work.','L020 has not started; own caches and trajectories may change phase costs. Third run runs after one of first pair completes, leaving a single-job tail.','Formal evaluation on this3090 and these lambdas has not started. Historical OBJECT600 vector compute56.60min is only a reference; total2–4hours is a planning allowance.','This is a planning range, not a statistical confidence interval.'],gpu_window={k:gpu[k] for k in ['mean_gpu_utilization','min_gpu_utilization','max_gpu_utilization','peak_used_MiB']},production_collection_mutex=a['parallel_collection_lock_audit'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
backup=D/'prior_local_metadata';backup.mkdir(exist_ok=True)
for name in ['MONITOR_HANDOFF.md','REPORT.md','RUN_STATUS.json','ACTIVE_ROOT.json']:
 if not (backup/name).exists():shutil.copy2(R/name,backup/name)
old_header=(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8').split('# OBJECT 最新S固定λ_R六run：当前交接')[0]
subprocess.run([sys.executable,'-X','utf8',str(E/'record_progress.py'),str(D),'90'],check=True)
header='''## 2026-11-02 18:35 用户请求并行实际速度和全队列余时

check_readonly_20261102_183513审计PASS，172140全部epoch/门控/诊断前缀精确保留；135科学及全部工程冻结项不变，无失败/重训/新科学运行。L00541完整400轮40采集；L00542为334/400轮(Task9 34)、81608接受147516生产91隔离41准入autograd30采集；L00543为201/400轮(Task8 1)、38709接受61718生产62隔离30准入autograd20采集。新增阶段自己的准入/BRANCH_COPY_EQ/CHECK_EQ均通过，未来不得套PASS。累计935轮225817接受404534生产244自身隔离112准入autograd90采集0正式；L020三run尚未开始。controller8994、原4694及新9000 GPU正常，4347/8608/8609仍T，HANDOFF_COMPLETE尚未发生，禁止恢复旧等待器。

已实际观察3次采集启动锁完整acquired/released：43Task6/7和42Task8，互不重叠、各自10采集且退出成功，非仅模拟验证；其余15次待观察。全部成本边界及旧140+UNKNOWN不变。

同seed42同Task8、并行前20完整轮均45.85秒→并行后35完整轮均53.85秒，单轮慢17.45%。实际17:21–18:35共73.56分钟，两run合计新增252完整轮(Task6/7/8/9为82/100/36/34)。按此前阶段速度折算约125.56分钟串行纯计算工作，参考合计吞吐约1.71倍、同量耗时约少41%；这不是严格同工作量A/B：Task7参照受此前诊断竞争、Task9参照跨seed/缓存/检查版本，完整轮边界和采集亦有影响，不能当全部提速由并行因果贡献。30样本实际GPU均88.77%、80–94%、峰6051MiB，两个真实生产进程，无新增optimizer或环境诊断。

当前42Task9近10轮71.34秒，余66轮约78.48分钟纯计算，保存最后10采集另计，规划85–105分钟。43Task8首轮不作为稳定测速，参考42并行Task8均53.85秒与Task9当前均速，余纯计算207.76分钟，保存准备最后20采集另计。L005整组余训练采集规划3–4小时。L020三run按最多两并发、后一个单任务尾段模型，纯计算约8.78小时，计缓存/准备/采集/波动规划9–11小时；不能直接三run总工时除以二。两组1200正式未开始，历史参考预留2–4小时。全部训练采集+正式粗略再14–19小时，即11月3日08:35–13:35附近，CPU独立验收/解释发布另计，非完成保证。TIMING_AUDIT保存来源SHA/公式/限制。

下一90分钟在42整run预计结束附近复查真实自然根锁交接/采集互斥及剩余队列，按新事实调整，覆盖旧150分钟。健康无实质变化静默。同一监控保留至最终独立验收解释发布，旧科学协议和下面工程交接要求不变。

'''
for name in ['MONITOR_HANDOFF.md','REPORT.md']:
 p=R/name;p.write_text(header+old_header+p.read_text(encoding='utf8'),encoding='utf8')
(R/'PARALLEL_SPEED_ETA_20261102_1835.md').write_text('# 并行后的生产速度与剩余时间\n\n'+header,encoding='utf8')
state=js(R/'ENGINEERING_LIVE_PARALLEL_STATUS.json');state.update(observed=a['observation'],snapshot=D.name,snapshot_sha256=sha(D/'EVIDENCE.json'),production_collection_locks_observed=3,next_check_minutes=90,latest_timing_audit=str((D/'TIMING_AUDIT.json').relative_to(R)),GPU_window=dict(source=G.name,samples=30,mean_utilization=gpu['mean_gpu_utilization'],min_utilization=gpu['min_gpu_utilization'],max_utilization=gpu['max_gpu_utilization'],peak_used_MiB=gpu['peak_used_MiB'],scope='Actual Task9/Task8 production; not controlled A/B'))
(R/'ENGINEERING_LIVE_PARALLEL_STATUS.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(out,ensure_ascii=False))
