"""Record the reviewed L005-only result after actual CPU and paired recomputation."""
from pathlib import Path
import json,sys,hashlib,datetime
R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1]).resolve();js=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_FINAL_AUDIT.json');c=js(D/'PRIMARY_S_COMPARISON.json');i=js(D/'INTERPRETATION_AUDIT.json');e=js(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_L005_LOCAL_ACCEPTANCE' and c['status']=='PASS_L005_EXISTING_S_600_PAIRED_EPISODES'
assert a['evidence_sha256']==sha(D/'EVIDENCE.json') and c['source_Variant_final_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
assert e['group_hashes_before']==e['group_hashes_after'] and e['optimizer_updates']==e['environment_calls']==0
assert (a['epochs'],a['main_updates'],a['physical_training_optimizer_calls'],a['isolated_optimizer_updates'],a['formal_episodes'])==(1200,316500,585900,273,600)
assert sum(x['gate_audit']['blocks'] for x in a['cases'].values())==27300
now=datetime.datetime.now().astimezone().isoformat()
accept=dict(status=a['status'],scope='L005_ONLY_L020_CONTINUES',accepted_at=now,final_directory=str(D.resolve()),evidence_sha256=sha(D/'EVIDENCE.json'),local_final_audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),interpretation_sha256=sha(D/'INTERPRETATION_AUDIT.json'),cases=list(a['cases']),checkpoints=12,full_state_restorations=3,teacher_caches=12,formal_episodes=600,gate_blocks=27300,accepted_updates=316500,production_optimizer_calls=585900,own_isolated_calls=273,admission_autograd=123,direction_autograd=108,training_collection_episodes=120,acceptance_optimizer_calls=0,acceptance_environment_calls=0,group_hashes=e['group_hashes_before'],completion_source='Three actual evaluate EXIT0 and no L005 process; train exit0 for41/42, verified resumed exit0 for43 with original failure retained',full_six_run_acceptance='PENDING_L020',scale_exception='L00541 Task6 only: exact cache/input hashes and measured old/current/float64 values bound, 1 float32 ULP; original failed attempt retained',hardware_boundary='User-authorized migration; descriptive comparison, not hardware-controlled lambda-only causal effect',engineering_optimizer_calls_global_to_date=35418,engineering_autograd_global_to_date=94,abandoned_recorded_candidates=540,abandoned_unlogged_cost='UNKNOWN')
dump(R/'L005_ACCEPTANCE.json',accept)
lines=['# OBJECT λ_R=0.05 三 seed：独立验收与0.1只读配对', '', f'验收时间：{now}。0.05组三seed独立接受；0.2组继续运行，全六run总体仍为PARTIAL。', '',
'0.05最终四任务平均成功率为82.0833%，已有0.1为85.0000%，差−2.9167个百分点。三个seed的差值分别为−11.25、+2.50和0个百分点，方向不一致。本次保留固定档位结果，不依据成绩选择λ或追加实验。', '',
'## 实际验收', '',
'真实CPU检查通过12个阶段模型、3份完整恢复状态、12份成功缓存teacher重放、600回合初态/noise、27300个门控块及逐阶段预算。本组1200完整epoch、316500接受更新、585900生产optimizer调用、273隔离调用、123准入autograd、108方向autograd、120训练采集与600正式均核对完整。与已有0.1的初始化tensor/RNG、三份Task6模型全部tensor及600回合配对字段均通过。既有0.1未重训重评。', '',
'L00541/42自己的train/evaluate出口均0；L00543使用已验证的恢复出口0，原train_S.exit1和旧调度失败原SHA永久保留。三个L005评估自然完成且没有自身进程，CPU审计前后该组全部轻量文件SHA逐项一致；0.2不需要退出。未伪造全队列controller成功。此次验收0训练更新、0新环境回合。', '',
'首次外置CPU断言失败及只读诊断原样保留。12份缓存里仅L00541 Task6的一个归一化分量在当前主机重算相差1 ULP（2.9802322387695312e-08）；缓存/元数据、输入数据SHA一致。v2仅绑定这份缓存、输入、旧值、重算值和独立float64参考，其他项保持原精度检查，生产仍使用原保存值。不是修改损失、阈值或结果。详见l005_acceptance_20261103_v2/SCALE_REVIEW.md。', '',
'## 配对指标', '', '|指标（百分比）|λ=0.05|λ=0.1|0.05−0.1（百分点）|','|---|---:|---:|---:|']
for k,x in c['suites']['L005'].items():lines.append(f"|{k}|{x['Variant_mean']:.4f}|{x['S_mean']:.4f}|{-x['S_minus_Variant_pp']:+.4f}|")
lines+=['','NBT越低越好；负值表示按该定义旧任务后续表现较其初学阶段有净改善。FWT在这里按任务首次学完的对角线成功率计算，不是零样本迁移。','', '|seed|0.05最终|0.1最终|差值（百分点）|','|---|---:|---:|---:|']
for case,x in c['cases'].items():lines.append(f"|{case[-2:]}|{x['Variant_metrics']['FinalAll']:.2f}|{x['S_metrics']['FinalAll']:.2f}|{-x['delta_pp']['FinalAll']:+.2f}|")
lines+=['','## 新旧任务及配对成败', '',
'seed41的最终差距同时出现在旧Task6（50%对70%）、旧Task7（85%对90%）和新Task9（80%对100%）；Task8持平。seed42最终高2.5个百分点，但Task6较低（60%对65%），改善来自Task8和新Task9。seed43最终持平，Task6增加5个百分点、Task8减少5个百分点互相抵消；其Task8刚学完阶段45%对90%的明显差距没有被最终均值表达出来。因而不能把结果概括为所有任务都对系数不敏感，或较小系数只影响旧任务。', '',
'在全部30个阶段/任务单元中，0.05相对0.1为10胜、7平、13负。以下SF/FS是两方法间配对成功/失败，不把同一初态在多个阶段出现视为独立随机样本。', '', '|范围|回合数|0.05成功/0.1失败|0.05失败/0.1成功|0.05成功数|0.1成功数|','|---|---:|---:|---:|---:|---:|']
for name,x in i['paired_summary'].items():lines.append('|%s|%d|%d|%d|%d|%d|'%(name,x['n'],x['L005_success_L010_failure'],x['L005_failure_L010_success'],x['L005_successes'],x['L010_successes']))
lines+=['','## 0.05全部负单元', '', '|seed|阶段/任务|0.05|0.1|差值|SF|FS|','|---|---|---:|---:|---:|---:|---:|']
for x in i['all_negative_cells']:lines.append('|%s|%s|%g|%g|%+g|%d|%d|'%(x['case'][-2:],x['cell'],x['L005'],x['L010'],x['delta_L005_minus_L010'],len(x['gain_initials']),len(x['loss_initials'])))
lines+=['','全部正单元及逐回合初态索引也保存在INTERPRETATION_AUDIT/PRIMARY_S_COMPARISON中，没有筛掉负单元或失败回合。', '',
'## 门控、梯度与训练代理', '',
'两档均保持同样的双候选10步门控，改变的是保持合成系数；各run沿自己的轨迹采集成功缓存，缓存内容和支持度不是固定的。0.05三seed的Q接受次数分别4120、4532、4245，共12897/27300（47.2418%）；0.1分别4268、4107、4000，共12375/27300（45.3297%）。更高的Q接受比例没有对应更高的最终均值，门控在固定训练探针上相对同状态C满足条件，不是对未见闭环成功的保证。', '',
'0.05的36个预定方向诊断点中，保持梯度与当前BC点积为负13次、与执行误差点积为负10次。以seed41 Task9末次诊断为例，两余弦约−0.234/−0.197；seed42同阶段约+0.159/+0.372。方向随阶段和seed变化，不支持“λ减半必然减少冲突并改善闭环”的推断。梯度范数和损失尺度不同，BC可为负的密度损失；不能仅比较损失数字大小来判断保持项实际作用。', '',
'自身训练采集的Task6支持度为0.2/0.1/0.5，对应26/13/63缓存窗口；Task7为1.0/0.9/0.9，Task8为1.0/1.0/0.7。它们是固定训练采集的成功比例，并非独立估计的泛化成功概率。旧任务专家执行误差、最佳专家误差与路由regret已在CPU逐样本复算；它们以及缓存保持误差仅作为离线代理，不替代正式闭环证据。终端Task9缓存不参与后续训练。', '',
'## 解释边界与成本', '',
'Task6无保持项，且三个seed的Task6模型tensor均与0.1相等，但正式Task6成功率仍出现45%对35%、40%对45%、70%对70%的差异；seed43即使均值相同，成败身份也并非全同。初态/noise/env_seed配对字段一致，因此这不是可以简单归因于λ的差异。当前结果跨用户授权的4080SUPER→3090迁移并包含工程中断恢复；实际机制可能涉及推理数值或环境执行差异，本次审计不声称定位其唯一原因。接受意味着结果与执行记录可追溯、预算和配对字段通过，不意味着硬件完全受控的λ单因素因果确认。', '',
'0.05和0.1的接受更新、生产物理optimizer次数及采集次数相等；自身缓存、FLOPs及墙钟不保证相等。全队列至今工程隔离验证35418 optimizer/94 autograd单列，已记录放弃候选540次及中断最后未记账成本UNKNOWN保留，不能把585900称为包括工程与故障的精确实际总成本。', '',
'已有0.1的600回合本次只读配对一次；未来0.2再次配对不会增加独立基线回合。保留开发集反复使用、预训练来源未闭合及三seed样本量限制，不称独立确认、全局无泄露、统计显著或普适最优。0.2继续执行原预算，全部结束后再报告三档，不因本组结果改科学协议。', '',
f'原始实际CPU证据：`{D.relative_to(R).as_posix()}`。对应LOCAL_FINAL_AUDIT、PRIMARY_S_COMPARISON、INTERPRETATION_AUDIT的SHA均见L005_ACCEPTANCE.json。']
(R/'L005_FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
accept['report_sha256']=sha(R/'L005_FINAL_REPORT.md');dump(R/'L005_ACCEPTANCE.json',accept)
print(json.dumps({k:accept[k] for k in ['status','scope','accepted_at','final_directory']},ensure_ascii=False))
