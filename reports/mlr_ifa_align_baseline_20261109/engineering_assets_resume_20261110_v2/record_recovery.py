from pathlib import Path
import json,sys,datetime,hashlib
D=Path(sys.argv[1]);R=Path(__file__).resolve().parent;BASE=R.parent;J=lambda p:json.loads(p.read_text(encoding='utf8'))
e=J(D/'EVIDENCE.json');a=J(D/'LOCAL_AUDIT.json');eng=J(D/'ENGINEERING_AUDIT.json');L={n:v['text'] for n,v in e['light'].items()}
assert eng['status']=='PASS_ENGINEERING_PROGRESS_NOT_FINAL' and eng['rng_boundary']['status']=='PASS_EXACT_PRE_CELL_RNG'
assert a['counts']['OBJECT_41']['formal_episodes']>=10,'Need actual completed rollout, not just started process'
assert a['epochs']==1200 and a['production_updates']==316500
assert any(p['pid']==40721 for p in e['processes'])
attempts=[json.loads(x) for n,v in L.items() if n.startswith('OBJECT_41/random/evaluation/attempts_') for x in v.splitlines()]
done=[x for x in attempts if x['event']=='vector_attempt_complete'];assert done and not any(x['event']=='technical_exception' for x in attempts)
obs=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
out=dict(status='PARTIAL_RECOVERED_ACTUAL_FORMAL_ROLLOUT_CONFIRMED',observed=obs,root='/data/mlr_ifa/experiment_workspaces/MLR_IFA_ALIGN_RANDOM_OBJECT_GOAL41_42_43_20261110_v2',controller=40721,contract=e['contract_sha256'],engineering_contract=e['engineering_contract_sha256'],engineering='engineering_assets_resume_20261110_v2',counts=a['counts'],epochs=a['epochs'],production_updates=a['production_updates'],formal_episodes=a['formal_episodes'],snapshot=str(D),evidence_sha256=hashlib.sha256((D/'EVIDENCE.json').read_bytes()).hexdigest(),old_failed_exits_preserved=True,rng_exact=True,completed_object_training_repeated=0,new_optimizer_diagnostics=0,queue='OBJECT600formal then GOAL3train at most2parallel then GOAL600formal',final_wrappers='engineering v2 final_readonly/audit_final_local; original base compare_baseline',final_wrapper_status='STATIC_ONLY_NOT_EXECUTED',remaining_total_time='Wait for more actual formal vectors; old stopped-queue ETA invalid')
(D/'RECOVERY_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
for n in ['ACTIVE_ROOT.json','RUN_STATUS.json']:(BASE/n).write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
report=f'''# ALIGN资源故障已修复并实际恢复评估

实际观察{obs}，PARTIAL_RECOVERED_ACTUAL_FORMAL_ROLLOUT_CONFIRMED。OBJECT41已完成{a['counts']['OBJECT_41']['formal_episodes']}/200正式回合，首个完整vector记录已验真；controller40721及评估进程正常。OBJECT三seed原400轮/12阶段ALIGN准入和所有模型/恢复文件SHA保留，0已完成轮重跑、0新增optimizer诊断。GOAL未开始，仍在OBJECT600正式全部完成后启动。

原故障为新source目录缺少libero/libero/assets。读取服务器既有/data/mlr_ifa/src/libero/libero/assets的585文件（422313213bytes）并冻结完整哈希，仅建立六个链接，没有复制大资产、下载模型或改评估源码。原10子进程在场景XML打开阶段失败，原日志/attempt/exit1/RNG文件永久保留。首个恢复包装缺少多进程main保护，spawn子进程在环境构造前断言退出，原失败也保留；v2增加main保护和run_module alter_sys=True，维持原评估模块的多进程启动语义。

原首单元Python/NumPy/CPU/CUDA随机状态经两次恢复均逐项精确通过，最后guard在正式环境调用前核验；原模型、动作采样和评估函数不变。两个失败attempt独立保留，不计为已完成正式回合或训练收益；不抹去历史环境构造尝试。原总controller.exit1、OBJECT41 evaluate.exit1、恢复v1 exit1均保持，不伪造成功。

唯一活动工程engineering_assets_resume_20261110_v2，合同404ffb7e4a04f4d52091401e77df82e20086cfd2c7001f88c7c449642fd75e12。禁止重复deploy/controller或恢复旧调度器。后续只用该工程transport.py check_readonly.py → audit_progress.py NEW PREVIOUS。最终仅该工程controller.exit0且无自身进程后，用该工程final_readonly.py→audit_final_local.py→基础compare_baseline.py；最终包装已静态适配真实完成源，仍待实际CPU执行。最初冻结科学合同ce307dca2bce082e36803d266067462cb35a95f421e371f74f7d20d42a5dda8f保持。

没有改变OBJECT→GOAL的顺序，没有启动后台聊天定时监控。有限远端队列会继续OBJECT600正式→GOAL三seed训练（最多两并行）→GOAL600正式；出错保留现场并停止新增运行。当前只确认修复和实际评估进展，不从首个vector外推全套件稳定吞吐。
'''
(BASE/'ASSET_RECOVERY_REPORT_20261110.md').write_text(report,encoding='utf8');(BASE/'LATEST_PROGRESS_REPORT.md').write_text(report,encoding='utf8')
p=BASE/'MONITOR_HANDOFF.md';p.write_text('## '+obs+' 故障已修复，实际评估恢复\n\n'+report+'\n\n'+p.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(out,ensure_ascii=False))
