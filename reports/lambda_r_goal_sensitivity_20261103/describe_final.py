"""Describe accepted fixed-grid outcomes and costs; no science execution."""
from pathlib import Path
import json,hashlib,statistics,datetime
R=Path(__file__).resolve().parent;D=R/'final_readonly_20261104_122613'
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');c=js(D/'PRIMARY_S_COMPARISON.json');e=js(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['evidence_sha256']==sha(D/'EVIDENCE.json')
assert c['source_Variant_final_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json') and c['paired_episodes']==1200
previous=js(R/'check_readonly_20261104_122528/EVIDENCE.json')
assert e['light']==previous['light'] and e['contract_sha256']==previous['contract_sha256'] and not e['processes']
assert e['light']['controller.exit']['text']=='0'
summary={};proxies={};all_cells=[]
for group in ['L005','L020']:
    cases={k:v for k,v in c['cases'].items() if k.startswith(group)}
    pairs=sum([v['paired_outcomes'] for v in cases.values()],[])
    cells=[dict(case=k,stage=int(st.split('_')[0]),task=int(st.split('_')[1]),**v) for k,x in cases.items() for st,v in x['cells'].items()]
    all_cells+=cells
    summary[group]=dict(metrics=c['suites'][group],NewFinal=statistics.mean(v['cells']['9_9']['Variant_percent'] for v in cases.values()),baseline_NewFinal=statistics.mean(v['cells']['9_9']['S_percent'] for v in cases.values()),SF_new_success_baseline_failure=sum(x['Variant'] and not x['S'] for x in pairs),FS_new_failure_baseline_success=sum(not x['Variant'] and x['S'] for x in pairs),wins=sum(x['S_minus_Variant_pp']<0 for x in cells),ties=sum(x['S_minus_Variant_pp']==0 for x in cells),losses=sum(x['S_minus_Variant_pp']>0 for x in cells))
for k,v in a['cases'].items():
    gr=v['gradient_records'];p=json.loads(e['light'][k+'/EPOCHS_S.json']['text'])[-1]['training_probe']
    off={kind:sum(x['count']*x['statistics']['execution_cost']['mean'] for x in v['offline'][k+'_'+kind+'_S_9'].values())/sum(x['count'] for x in v['offline'][k+'_'+kind+'_S_9'].values()) for kind in ['expert','retain']}
    proxies[k]=dict(Q_accepts=sum(x['Q_accepts'] for x in v['gate_audit']['by_stage'].values()),gate_blocks=6700,support_by_stage={st:x['support'] for st,x in v['gate_audit']['by_stage'].items()},gradient_samples=len(gr),negative_cosine_bc=sum(x['retain_vs_bc']['cosine'] is not None and x['retain_vs_bc']['cosine']<0 for x in gr),negative_cosine_exec=sum(x['retain_vs_exec']['cosine'] is not None and x['retain_vs_exec']['cosine']<0 for x in gr),last_training_probe=p,offline_final_costs=off,epoch_minutes=v['epoch_seconds']/60,task6_baseline_all_tensors_exact=v['task6_baseline_all_tensors_exact'])
assert all(x['task6_baseline_all_tensors_exact'] for x in proxies.values())
out=dict(status='PASS_FINAL_INTERPRETATION_RECOMPUTED',final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),original_completed_light_preserved=True,summary=summary,training_proxies=proxies,all_cells=all_cells,all_new_negative_cells=[x for x in all_cells if x['S_minus_Variant_pp']>0],all_baseline_negative_cells=[x for x in all_cells if x['S_minus_Variant_pp']<0],no_additional_scientific_runs=True)
(D/'INTERPRETATION_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
lines=['# GOAL固定λ_R三档敏感性结果：全部接受','',
'六个新增run已完成实际远端CPU和本地独立复算验收：λ_R=0.05、0.2各seed41/42/43。历史0.1结果直接复用。最终均值依次为81.6667%、85.0000%、82.9167%。0.05的旧任务最终成绩较低；0.2平均旧任务成绩略高，但新任务学习明显较弱，尤其seed41。本次结果展示系数改变后的新旧任务取舍及种子差异，不据此选择最佳λ，也不追加实验。','',
'训练与1200正式回合于2026-11-04 02:20:17自然结束，root/controller.exit为0且无自身进程。唯一最终CPU请求于12:26提交，12:33:36实际完成，随后本地复算和1200对历史基线的逐项配对通过。此前0.05的描述结果已由本次完整CPU验收确认。','',
'## 三档完整结果','',
'AUC按每任务在可见阶段的平均后再平均；FWT为任务初学成绩；NBT为后续成绩相对初学成绩的平均下降，越低越好。其余成绩均越高越好。所有值为三seed平均，差值单位为百分点。','',
'|指标|0.05|既有0.1|0.2|','|---|---:|---:|---:|']
for k in ['AUC','FWT','NBT','OldFinal','FinalAll']:
    lines.append('|%s|%.4f|%.4f|%.4f|'%(k,c['suites']['L005'][k]['Variant_mean'],c['suites']['L005'][k]['S_mean'],c['suites']['L020'][k]['Variant_mean']))
lines+=['|NewFinal|%.4f|%.4f|%.4f|'%(summary['L005']['NewFinal'],summary['L005']['baseline_NewFinal'],summary['L020']['NewFinal']),'',
'|seed|0.05最终|0.1最终|0.2最终|0.05−0.1|0.2−0.1|','|---|---:|---:|---:|---:|---:|']
for seed in [41,42,43]:
    x=c['cases']['L005_'+str(seed)];y=c['cases']['L020_'+str(seed)]
    lines.append('|%d|%.2f|%.2f|%.2f|%+.2f|%+.2f|'%(seed,x['Variant_metrics']['FinalAll'],x['S_metrics']['FinalAll'],y['Variant_metrics']['FinalAll'],-x['delta_pp']['FinalAll'],-y['delta_pp']['FinalAll']))
lines+=['',
'0.05相对0.1：最终均值低3.3333pp，旧任务最终低4.4444pp，最终新任务均值相同。新任务相同的均值掩盖seed41/42各高10pp、seed43低20pp的差异。seed41最终Task6低25pp而Task8/9分别高15/10pp，最终总分相抵；seed42主要在Task8低25pp；seed43的Task8/9低15/20pp，Task6/7则高10/5pp。不能将0.05称为逐任务一致变差，也不能从最终新任务均值相等推出训练过程相同。','',
'0.2相对0.1：旧任务最终均值高1.1111pp，NBT低6.2037pp，但FWT低6.6667pp、最终新任务低11.6667pp，最终总分仍低2.0833pp。主要负单元是seed41最终Task9仅40%对80%，使该seed最终总分低12.5pp；seed42和43最终总分分别高2.5/3.75pp。因此0.2的负均值不是三个seed一致下降，更不能解释为“保持越强越好”或“所有任务都退化”。','',
'NBT以各任务刚学完的成绩为参照，初学成绩较低可能使下降幅度变小。0.05和0.2都具有较低FWT，需同时看最终旧任务成绩；0.2的OldFinal确实略高，而0.05的OldFinal较低。负NBT表示相对初学成绩的平均后续改善，不等于最终绝对成功率最高。','',
'## 回合配对和全部负结果','',
'SF在本报告定义为新系数成功、0.1失败；FS为新系数失败、0.1成功。每档600回合的stage/task/initial_index、初态文件与初态hash、Normal噪声hash、env_seed和attempt逐项一致。每档阶段/任务单元30个，每单元20回合。','',
'|新系数|SF|FS|单元胜/平/负（相对0.1）|','|---|---:|---:|---|']
for g in summary:
    x=summary[g];lines.append('|%s|%d|%d|%d/%d/%d|'%(g,x['SF_new_success_baseline_failure'],x['FS_new_failure_baseline_success'],x['wins'],x['ties'],x['losses']))
lines+=['',
'SF/FS统计跨全部学习阶段，不能直接等同最后阶段的FinalAll差值；同一初态跨阶段重复出现，回合亦嵌套于seed，不按600个独立随机样本宣称显著性。下附全部60单元，正负与平局均保留；PRIMARY_S_COMPARISON.json保存不一致回合索引及完整配对。','',
'## 门控、梯度与训练代理','',
'|case|Q接受/6700块|Q接受率|12次梯度样本中保持与BC/执行梯度夹角为负数|末Task9新任务执行探针|末Task9保持探针|CPU旧样本执行代价|CPU自身缓存保持距离|','|---|---:|---:|---|---:|---:|---:|---:|']
for k,v in proxies.items():
    lines.append('|%s|%d|%.2f%%|%d / %d|%.6f|%.6f|%.6f|%.6f|'%(k,v['Q_accepts'],100*v['Q_accepts']/6700,v['negative_cosine_bc'],v['negative_cosine_exec'],v['last_training_probe']['exec_new'],v['last_training_probe']['retain_support'],v['offline_final_costs']['expert'],v['offline_final_costs']['retain']))
lines+=['',
'0.05合计Q接受9224/20100块（45.89%），0.2为9189/20100块（45.72%），接受比例相近不代表所选参数路径相同。各run以自己的成功缓存计算q；例如seed42的Task6支持度0.9，seed43的Task8支持度0.7。缓存内容随策略改变，不能把三档当作固定缓存、同一中间状态的单步λ因果对照。每stage自身准入及完整状态等价检查均通过。','',
'梯度诊断只取预定的Task7–9各0/9/49/99轮首批次，表中是12个样本的方向冲突计数；不是所有更新的统计，也不单独证明闭环收益来源。0.2 seed41的最后保持探针及CPU缓存保持距离小于0.05 seed41，但Task9闭环成功率40%低于90%；这直接说明较低保持代理不能保证新任务闭环性能。CPU执行代价和保持距离不等于成功率，保持距离还依赖各自缓存，跨run比较只能作描述。原始门控、q、梯度和CPU复算标量均保留。','',
'## 独立验收、预算和适用边界','',
'实际CPU严格加载24个阶段checkpoint，检查6份完整恢复状态、24份teacher缓存并重放，核1200初态与独立生成的noise、40200门控块和全部预算；所有scale精确通过，没有套用OBJECT的1ULP例外。最终CPU合计7706次前向调用、23370个窗口，0 optimizer更新、0环境回合。验收前后轻量科学证据逐项SHA一致。24次采集和6次正式评估的全局环境锁均完整成对且互不重叠。','',
'每新增run为76100接受更新、142200生产物理optimizer调用、91隔离准入调用、41准入autograd和36方向autograd；六run合计456600接受、853200生产、546隔离、246准入及216方向autograd。工程验证另计5704 optimizer/36 autograd/0环境；两个部署前包装失败发生在零更新、零环境之前，原证据保留。GOAL生产没有重启、重复epoch、补采或正式重评。每run40训练采集、200正式，总240采集1200正式；每档与旧0.1的600回合配对，同一基线被复用两次而非1200个独立基线回合。旧OBJECT的35418/94工程成本和540已记放弃候选+UNKNOWN仍留原账，不混入GOAL。','',
'逐阶段接受更新、物理optimizer预算及采集尝试次数与旧0.1相等；实际缓存窗口、推理终止步数、FLOPs、墙钟和工程验证总成本不因此完全相等。两个训练并行只用于吞吐，采集和正式环境始终互斥。','',
'六run官方初始化tensor/RNG与基线相同，六份Task6模型与基线全部tensor实际相等。但Task6闭环0.05和0.2均为85/70/70%，旧0.1为90/70/75%；seed42均值相同仍有成败互换。这发生在不使用保持损失的Task6，说明存在硬件/推理或环境执行差异边界，不能将全部差值归于λ。新增运行在用户授权的3090，历史基线在4080 SUPER；这是固定网格的敏感性结果，不能称硬件完全受控的λ单因素因果实验。','',
'12:25最终前检查时nvidia-smi返回9，不能访问驱动；此时训练和评估已自然退出0、全前缀和文件hash完整。后续实际CPU验收通过，无GPU需求。未修改驱动、重启服务器或重评；异常原因及开始时间没有被本次证据确定。','',
'开发初态被反复用于方法开发，预训练数据来源的独立性仍未完全闭合；初态20–29采集与0–19评估分离及hash/noise配对并不能证明全局无泄露。只报告这三个seed的观察，不称独立确认、普适显著性或最佳λ。','',
'OBJECT先前已接受、发布并按SHA清理其专有pth及HOOK_ENABLED；GOAL使用独立environment_locked，不安装site-packages hook，因此无GOAL自有hook需删除。OBJECT→GOAL用户授权范围至此已全部完成科学验收和解释，待本报告出版核验后关闭同一监控。','',
'## 全部阶段/任务单元（新系数−0.1）','',
'|case|stage/task|新系数成功率|0.1成功率|差值pp|SF|FS|','|---|---|---:|---:|---:|---:|---:|']
for x in all_cells:
    lines.append('|%s|%d/%d|%.1f|%.1f|%+.1f|%d|%d|'%(x['case'],x['stage'],x['task'],x['Variant_percent'],x['S_percent'],-x['S_minus_Variant_pp'],len(x['Variant_success_S_fail']),len(x['Variant_fail_S_success'])))
lines+=['','证据目录：`final_readonly_20261104_122613`。核心文件：LOCAL_FINAL_AUDIT.json、PRIMARY_S_COMPARISON.json、INTERPRETATION_AUDIT.json；原始证据hash和受审轻量scalars/outcomes保留，模型/state/cache/RNG载荷不发布。']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
accept=dict(status='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE_AND_INTERPRETATION',created=datetime.datetime.now().astimezone().isoformat(),final_evidence=D.name,evidence_sha256=sha(D/'EVIDENCE.json'),local_final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),interpretation_sha256=sha(D/'INTERPRETATION_AUDIT.json'),report_sha256=sha(R/'FINAL_REPORT.md'),contract=a['contract'],counts={k:a[k] for k in ['epochs','main_updates','physical_training_optimizer_calls','isolated_optimizer_updates','formal_extra_autograd','admission_extra_autograd','training_collection_episodes','formal_episodes']},engineering=dict(optimizer=5704,autograd=36,environments=0),CPU=dict(checkpoints=24,recoveries=6,caches=24,initial_noise_episodes=1200,gates=40200,calls=7706,windows=23370,optimizer_updates=0,environments=0),environment_mutex=a['environment_mutex_audit'],hardware_task6_all_exact=True,scientific_light_unchanged=True,goal_site_hook_created=False,publication='PENDING_VERIFIED_RECEIPT',limits=out.get('limits',[]))
(R/'ACCEPTANCE.json').write_text(json.dumps(accept,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=accept['status'],summary=summary),ensure_ascii=False))
