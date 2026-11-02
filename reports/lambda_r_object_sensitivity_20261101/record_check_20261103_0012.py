"""Bookkeep an audited snapshot and natural suite transition; no scientific calls."""
from pathlib import Path
import json, statistics, subprocess, sys
R=Path(__file__).resolve().parent; E=R/'engineering_cross_lambda_20261102_v1'
D=R/'check_readonly_20261103_001226'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
def put(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');l=v['light'];interval=105
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and not a['failures']
launches=[json.loads(x) for x in l[E.name+'/LAUNCHES.jsonl']['text'].splitlines()]
exits={c:json.loads(l[E.name+'/'+c+'_evaluate_EXIT.json']['text']) for c in ('L005_41','L005_42','L005_43')}
for c,x in exits.items():
 assert x['returncode']==0 and l[c+'/evaluate.exit']['text'].strip()=='0'
 assert a['counts'][c]['formal_episodes']==200
 assert not any('/'+c+'/evaluate.py' in p['cmd'] for p in v['processes'])
 assert json.loads(l[c+'/EVAL_ADMISSION.json']['text'])['status']=='PASS'
start42=next(x for x in launches if x['case']=='L020_42' and x['phase']=='train')
assert start42['time']>max(x['time'] for x in exits.values())
assert a['environment_mutex_audit']['nonoverlap']
task6=json.loads(l['engineering_rendezvous_recovery_20261102_v1/L020_42_TASK6_GPU_COMPARISON.json']['text'])
assert task6['all_tensors_exact']
for c,s in [('L020_41',9),('L020_42',7)]:
 assert json.loads(l[f'{c}/ADMISSION_S_{s}.json']['text'])['status']=='PASS_OWN_GPU_ADMISSION'
suite_minutes=(max(x['time'] for x in exits.values())-next(x['time'] for x in launches if x['phase']=='evaluate'))/60
timing=dict(status='MEASURED_L005_EVALUATION_COMPLETE_L020_PARTIAL',observation=a['observation'],snapshot_sha256=a['snapshot_sha256'],L005_600_evaluation_wall_minutes=suite_minutes,L02041_task9_recent10_seconds=a['counts']['L020_41']['recent_epoch_mean_seconds'],L02041_remaining_compute_minutes=a['counts']['L020_41']['current_stage_compute_minutes_estimate'],L02041_training_collection_planning_minutes=[105,125],L02042_task7_recent10_seconds=a['counts']['L020_42']['recent_epoch_mean_seconds'],L02042_task7_remaining_compute_minutes=a['counts']['L020_42']['current_stage_compute_minutes_estimate'],next_check_minutes=interval,global_completion_estimate='NOT_REISSUED',limits=['L02042 Task8/9 and L02043 are not measured; no fixed completion guarantee.','Current-run estimate excludes preparation, saving and last acquisition; planning includes margin.','L005 evaluation duration is measured under prior training competition and is not a guaranteed L020 evaluation duration.','Final independent CPU acceptance, interpretation and publication remain pending.'])
put(D/'TIMING_AUDIT.json',timing)
put(D/'TRANSITION_AUDIT.json',dict(status='PASS_L005_600_COMPLETE_BEFORE_L02042_LAUNCH',snapshot_sha256=a['snapshot_sha256'],evaluation_exits=exits,next_training_launch=start42,environment_mutex=a['environment_mutex_audit'],L02042_task6_gpu_comparison=task6,scientific_acceptance='PENDING_FULL_SIX_RUN_CPU_ACCEPTANCE'))
history=(R/'MONITOR_HANDOFF.md').read_text(encoding='utf8').split('# OBJECT 最新S固定λ_R六run：当前交接')[0]
subprocess.run([sys.executable,'-X','utf8',str(E/'record_progress.py'),str(D),str(interval)],check=True)
intro=f'''## 2026-11-03 00:12 实际复查：0.05全部600正式完成，0.2两seed继续并行

{D.name}及cross工程audit_progress实际PASS，230720全部epoch/门控/诊断/正式前缀精确保留。L005三个seed各400轮40采集200正式，三个evaluate.exit0、各自评估准入PASS且无L005进程；43前恢复L005_43_resume.exit0及原失败SHA保持。实际L00543评估退出早于L02042启动，符合用户优先完成0.05正式的顺序。该组结果已完整保存，尚待本轮最终独立CPU验收，不能提前称科学接受。

L02041 Task9完整16/100，共316轮75092接受134484生产91隔离41准入autograd30采集；Task9自身准入及BRANCH_COPY_EQ/CHECK_EQ通过。L02042 Task6完整100轮10采集、Task7完整12轮，共112轮18424接受21148生产33隔离19准入autograd；自己的Task6/7准入、初态/COPY_EQ/BRANCH_COPY_EQ/CHECK_EQ通过，Task6与已有S全tensor这次实际相等。43尚未开始，不套未来准入。累计1628轮410016接受741532完整轮生产397隔离183准入autograd160采集600正式。controller11852及GPU11860/28055健康，实际独立端口42689/44167，135科学及全部工程项不变，无新失败或重训。

合并环境锁实际10/18采集及3/6正式均完整acquired/released，时间线互斥通过、当前无持锁环境；未来8采集3正式待核。自有pth/HOOK_ENABLED继续保留，全部接受后才按原SHA清理。根锁交接已完成，旧等待器不恢复；最终成功源仍仅cross/controller.exit0且无自身进程。

L005600正式从首个dispatch至最后自然退出实际{suite_minutes:.2f}分钟。L02041 Task9近10轮68.5263秒，余84轮95.94分钟纯计算，计保存及最后10采集规划105–125分钟；42 Task7近10轮34.1320秒，余88轮50.06分钟纯计算，后两阶段及43尚未测得。下一105分钟在当前41整run完成较早侧复查，依新事实调整，不固定高频、不按时间推断完成。工程35418/94、已记放弃候选540+UNKNOWN、硬件及泄露边界均保持；全六run仍PARTIAL，独立CPU验收解释发布另计。下方旧进度和间隔以本段覆盖。

'''
for name in ('MONITOR_HANDOFF.md','REPORT.md'):
 p=R/name;p.write_text(intro+history+p.read_text(encoding='utf8'),encoding='utf8')
s=js(R/'RUN_STATUS.json');s.update(status='PARTIAL_L005_EVALUATION_COMPLETE_L020_TRAINING',timing_audit=str(D/'TIMING_AUDIT.json'),transition_audit=str(D/'TRANSITION_AUDIT.json'));put(R/'RUN_STATUS.json',s)
x=js(R/'ACTIVE_ROOT.json');x.update(engineering_state='L005_EVALUATION_COMPLETE_L020_TRAINING',active_training_pids={'L020_41':11860,'L020_42':28055},active_evaluation_pid=None);put(R/'ACTIVE_ROOT.json',x)
x=js(R/'ENGINEERING_CROSS_LAMBDA_STATUS.json');x.update(latest_observation=a['observation'],latest_evidence=D.name,latest_progress_status=a['status'],next_check_minutes=interval,L005_evaluation_completion='ACTUAL_600_AND_THREE_EXIT0_NOT_CPU_ACCEPTANCE');put(R/'ENGINEERING_CROSS_LAMBDA_STATUS.json',x)
print(json.dumps(timing,ensure_ascii=False,indent=2))
