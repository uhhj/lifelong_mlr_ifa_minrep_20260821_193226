from pathlib import Path
import json,datetime,hashlib,tomllib
R=Path(__file__).resolve().parent.parent;D=R/'check_readonly_20261103_004935'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
a=js(D/'LOCAL_AUDIT.json');accept=js(R/'L005_ACCEPTANCE.json');s=js(R/'RUN_STATUS.json')
assert s['L005_acceptance']==accept and a['status']=='PASS_PROGRESS_NOT_FINAL'
timing=dict(status='PARTIAL_ACTUAL_PROGRESS_NOT_FINAL',observation=a['observation'],snapshot_sha256=a['snapshot_sha256'],current_run='L020_41',remaining_epochs=52,recent_epoch_seconds=a['counts']['L020_41']['recent_epoch_mean_seconds'],remaining_compute_minutes=a['counts']['L020_41']['current_stage_compute_minutes_estimate'],current_run_planning_minutes=[70,90],next_check_minutes=70,limits=['Preparation/save/final ten acquisitions are extra','Other L020 stages, seed43 and formal evaluations are not inferred complete','No new full-queue guaranteed ETA; L005 acceptance does not stop L020'])
(D/'TIMING_AUDIT.json').write_text(json.dumps(timing,indent=2)+'\n',encoding='utf8')
intro='''## 2026-11-03 用户要求的0.05单独验收完成，0.2继续

l005_acceptance_20261103_v2/final_readonly_20261103_004224实际远程CPU及本地复算PASS_COMPLETE_L005_LOCAL_ACCEPTANCE：12CP/3完整恢复/12缓存teacher/600初态noise/27300门控通过，316500接受/585900生产/273隔离/123准入autograd/108方向autograd/120采集预算精确，与已有0.1的600正式配对通过。L005_ACCEPTANCE.json与L005_FINAL_REPORT.md永久保留；无需重复单独CPU请求。三个自己的评估退出0且无L005进程，43恢复出口0与原失败SHA保留；不伪造全六run完成。

实际CPU前后及最新00:49:39 check_readonly_20261103_004935的L005全部轻量SHA一致。初次CPU在L00541 Task6 scale重算1ULP断言失败，原失败和12缓存诊断保留；仅这一明确case/stage/cache/input/旧值/新值/float64参考精确绑定在v2，其他检查未放宽，生产没有改变。三Task6模型tensor与0.1实际相等，但Task6闭环45/40/70%对35/45/70%不同；须保留硬件/推理或环境执行差异边界，不能把全部差值归于λ或称硬件受控单因素实验。

0.05/0.1均值FinalAll82.0833/85、AUC83.1944/86.4931、FWT79.5833/80.4167、NBT−7.5926/−11.0185（低好）、OldFinal78.3333/81.1111。最终差值−11.25/+2.5/0pp，30单元10胜7平13负，全部600配对SF45/FS64；新旧任务、全部负单元和训练代理解释已写报告。不选最佳λ，不因负结果追加实验。

最新004935进度审计PASS，001226全部epoch/门控/诊断/正式前缀保留。L02041 Task9完整48/100，总348轮；42 Task7完整79/100，总179轮；43未开始。41/42分别30/10采集、正式均0。controller11852/GPU11860和28055正常，135科学及全部工程项不变。环境采集10/18和评估3/6已完整互斥，余8采集3评估待核。没有中断0.2或新增训练/环境验收成本。

41近10轮69.3148秒，余52轮60.07分钟纯计算，含保存最后10采集规划70–90分钟。下一70分钟在当前整run结束较早侧复查；42后阶段、43与正式不按时钟推断完成。全部最终验收仍须cross/controller.exit0且无自身进程；全六CPU包装及比较器须外置版本化适配已验证的scale局部差异、硬件比较和实际完成源，原包装STATIC_ONLY不能直接套PASS。0.05接受不取消0.2或最终全六验收。自有pth/HOOK_ENABLED继续保留至全部接受后按SHA清理。成本35418/94与放弃候选540+UNKNOWN、原260轮4167门控/43原289轮5079门控保留。

'''
for name in ['MONITOR_HANDOFF.md','REPORT.md']:
 p=R/name;p.write_text(intro+p.read_text(encoding='utf8'),encoding='utf8')
config=tomllib.loads(Path('C:/Users/Lenovo/.codex/automations/object-s-r-run/automation.toml').read_text(encoding='utf8'))
old=config['prompt'];tail=old[old.index('唯一工程engineering_cross_lambda_20261102_v1'):]
prompt='继续用户授权OBJECT λ_R六run。先完整读D:/论文课题研究/MLR+IFA/execution/20261101_lambda_r_object_sensitivity/MONITOR_HANDOFF.md及指定文件，以本地最新为准。'+intro.replace('\n',' ')+tail
config.update(prompt=prompt,rrule='RRULE:FREQ=MINUTELY;INTERVAL=70')
(R/'MONITOR_UPDATE_REVIEW.json').write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='L005_ACCEPTED_L020_PARTIAL',next_check_minutes=70)))
