"""Persist an audited progress snapshot and static final-wrapper preparation."""
from pathlib import Path
import json,datetime,tomllib,hashlib
R=Path(__file__).resolve().parent;D=R/'check_readonly_20261103_020151';js=lambda p:json.loads(p.read_text(encoding='utf8'))
a=js(D/'LOCAL_AUDIT.json');v=js(D/'EVIDENCE.json');accept=js(R/'L005_ACCEPTANCE.json')
assert a['status']=='PASS_PROGRESS_NOT_FINAL' and all(v['light'][n]['sha256']==h for n,h in accept['group_hashes'].items())
assert v['light']['L020_41/train_S.exit']['text']=='0'
hist={case:json.loads(v['light'][case+'/EPOCHS_S.json']['text']) for case in ('L020_41','L020_42','L020_43')}
means={str(t):sum(x['seconds'] for x in hist['L020_41'] if x['stage']==t)/100 for t in range(6,10)}
remaining42=a['counts']['L020_42']['current_stage_compute_minutes_estimate']+100*means['9']/60
reference43=sum(means[str(t)]*100 for t in (7,8,9))/60
timing=dict(status='PARTIAL_CONDITIONAL_TIMING',observation=a['observation'],snapshot_sha256=a['snapshot_sha256'],L02041_completed=True,L02041_stage_epoch_means=means,L02042_remaining_pure_minutes=remaining42,L02042_planning_minutes=[160,190],L02043_unmeasured_retention_reference_minutes=reference43,L02043_assumptions='Same stage throughput as completed L02041, own cache/resource differences unknown; Task6 collection ongoing',next_check_minutes=160,limits=['Preparation/save/collection and environment-lock waits extra','No time-based completion inference or extrapolated GPU admission','No new full-queue completion guarantee; GOAL queued, not built/deployed'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(timing,indent=2)+'\n',encoding='utf8')
intro='''## 2026-11-03 02:02 实际复查：0.2 seed41自然完成，42/43继续

check_readonly_20261103_020151及cross进度审计PASS，004935所有epoch/门控/诊断/正式前缀保留；已独立接受的0.05全部轻量SHA逐项精确保留。L02041完整400轮105500接受195300生产、40采集、train_S.exit0。L02042 Task8完整65轮，总265轮58485接受101270生产62隔离30准入autograd20采集；Task8自身29/11、BRANCH_COPY_EQ/CHECK_EQ通过。L02043已自然启动并完整Task6 100轮15700生产，自己的4隔离8autograd及初态/COPY_EQ/BRANCH_COPY_EQ/CHECK_EQ通过，Task6对已有S全tensor这次实际相等；正在首个10采集，尚无完成记录，保持阶段未来准入不套PASS。

总1965轮496185接受898170完整轮生产430隔离202准入autograd180完成采集、600正式。controller11852/GPU28055和29668正常，135科学及所有工程项不变；旧11860训练已自然结束，没有重训或新S调用。合并环境锁13采集acquired/12released，第13个L02043 Task6当前持锁，3评估均released，已发生区间互斥；未来采集/正式仍待核。采集是进行中，不能计作已完成10回合。

L02042 Task8近10轮52.579秒、余35轮30.67分钟纯计算；Task9未测参考L02041均67.7131秒，当前整run余纯计算143.53分钟，含准备保存剩余20采集规划160–190分钟。L02043保持阶段未测，参考L02041三阶段纯269.41分钟，仅条件规划。下一160分钟在当前42整run结束较早侧复查，不固定高频、不按时钟推断完成。

已外置准备final_acceptance_20261103_v1三份最终入口及6个CPU子脚本，WRAPPER_REVIEW仅STATIC_ONLY_NOT_EXECUTED：保留cross真实自然exit0/无进程、24CP/6恢复/24缓存/1200noise/54600gate/1171800生产/546隔离全验收；精确绑定已实测唯一L00541 Task6 1ULP例外，其他scale仍须逐位相等；记录实际Task6跨GPU tensor比较和硬件，不伪称λ单因素受控，并逐项核已接受L005证据不变。只在全组六run真实结束后使用该版本final_readonly→audit_final_local→compare_primary_s，不再直接套cross旧CPU包装或根旧比较器。本次未运行任何CPU最终请求、训练、环境或工程GPU验证。

GOAL接续授权/协议和基线SHA保留，尚未构建/冻结/部署；OBJECT全接受解释发布、按SHA清理自有hook后继续独立GOAL六run。未来监控须完成GOAL适配和实际容量预检，不能把OBJECT完成当最终目标或删除监控。0.05接受无需重复单独CPU，所有失败与35418/94、540+UNKNOWN账目保持。健康进展无须用户操作。

'''
assert abs(reference43-269.4132304395803)<.01
for name in ['MONITOR_HANDOFF.md','REPORT.md']:
 p=R/name;p.write_text(intro+p.read_text(encoding='utf8'),encoding='utf8')
active=js(R/'ACTIVE_ROOT.json');active['active_training_pids']={'L020_42':28055,'L020_43':29668};active['current_training_pid']=28055;active['engineering_state']='L005_ACCEPTED_L02041_COMPLETE_L02042_43_TRAINING';(R/'ACTIVE_ROOT.json').write_text(json.dumps(active,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
cfg=tomllib.loads(Path('C:/Users/Lenovo/.codex/automations/object-s-r-run/automation.toml').read_text(encoding='utf8'))
prompt='继续用户授权OBJECT→GOAL λ_R两档各三seed。先完整读取D:/论文课题研究/MLR+IFA/execution/20261101_lambda_r_object_sensitivity/MONITOR_HANDOFF.md及指定PROTOCOL/IMPLEMENTATION_NOTES/RUN_STATUS/ACTIVE_ROOT、AUTHORIZED_GOAL_FOLLOWUP、goal_followup_20261103/PROTOCOL与BASELINE_BINDING和最新EVIDENCE/LOCAL_AUDIT/TIMING_AUDIT，以本地最新为准。'+intro.replace('\n',' ')+'''用户GOAL只新增.05/.2各41/42/43、旧.1只读；各任务100轮，最多两训练、环境全互斥，优先.05正式和跨组填空槽方式。GOAL六run456600接受853200生产40200门控546隔离246准入autograd最多216方向autograd240采集1200正式，不能套OBJECT预算。候选execution/20261103_lambda_r_goal_sensitivity及root S_LAMBDA_R_GOAL41_42_43_20261103_v1尚未构建部署，监控须自主适配代码/冻结/实际容量和自身准入检查，在OBJECT全部完成验收发布及自有hook清理后唯一启动，不与OBJECT重叠。容量不足不删旧模型或付费，需实际证据通知。沿用同一监控，GOAL也全部接受解释发布清理后才关闭。当前OBJECT唯一cross工程controller11852、合同ee7929e8ae178125508a0feead81c0fa153f7a725f62f46bdf99fe88c9fbcead，科学合同6fde2ecbc3953f0af0709e70bbc3aaed3df9be57bf9e6bf59476985caafb135e/135项。禁止重跑任何controller/dispatch/build/validation/已完成轮或恢复旧机制，旧C/Q/S只读。现有cross check/audit_progress/record_progress继续使用；仅最终CPU改用final_acceptance_20261103_v1三入口且先确认真实cross/controller.exit0和无自身进程。前恢复controller.exit保持不存在，旧live/controller.exit1和43原失败SHA保留，43成功以恢复出口0为准。L005已接受600回合，结果82.0833对旧.1的85%；Task6模型tensor同而闭环不同，保留硬件推理/环境差异和开发集复用/预训练来源未闭合，不称λ单因素因果、不按成绩选参数。旧260轮4167门控/43原289轮5079门控保留，工程35418/94、放弃候选540+UNKNOWN不改账。torchrun显式c10d localhost:0唯一UUID，不用--standalone。采集事件旧live目录、评估cross目录合并核非重叠；自有pth/HOOK_ENABLED全OBJECT接受后核无目标进程和SHA才清理，不提前删。运输旧机制.session_control worker3700先核status/PID，不重启活跃worker不输出凭据，cross check复制旧目录queue按result真实目录取证。每次按AGENTS经publish.py审核轻量code/report/hash/scalars/outcomes串行nonforce推MLR核remoteSHA/receipt，禁CP/model/state/RNG/cache/凭据。健康进展静默，仅实质变化/完成/新失败/需行动通知。'''
cfg.update(prompt=prompt,rrule='RRULE:FREQ=MINUTELY;INTERVAL=160')
(R/'MONITOR_UPDATE_REVIEW.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='PASS_RECORDED',next_check_minutes=160,L02042_compute=remaining42,L02043_reference=reference43)))
