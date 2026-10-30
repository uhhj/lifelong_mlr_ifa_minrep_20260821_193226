"""Explain accepted GOAL results; OBJECT acceptance is deliberately excluded."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,statistics,sys,datetime
A=Path(__file__).resolve().parent;R=A.parent;D=Path(sys.argv[1])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');c=js(D/'PRIMARY_S_COMPARISON.json');e=js(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE'
assert c['status']=='PASS_GOAL_NEW_C_EXISTING_S_600_PAIRED_EPISODES_EQUAL_STAGE_OPTIMIZER_BUDGET'
assert c['paired_episodes']==600 and c['source_C_final_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
assert set(c['cases'])=={'GOAL_41','GOAL_42','GOAL_43'}
binding=js(R/'S_ACCEPTANCE.json');sf=Path(binding['sources']['GOAL']['final_path'])
h=js(sf/'PRIMARY_C_COMPARISON.json');assert h['status']=='PASS_ACCEPTED_C_S_600_EPISODE_PAIRING'
for name,digest in h['sources'].items():
 p=Path(name)
 if not p.is_absolute():p=sf.parent/p
 assert sha(p)==digest,(name,digest)
def metric(rows,field):
 r={(s,t):F(sum(x[field] for x in rows if (x['stage'],x['task'])==(s,t)),20) for s in range(6,10) for t in range(6,s+1)}
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
history={};paired_counts={};training=[];offline=[]
for seed in (41,42,43):
 case='GOAL_'+str(seed);v=c['cases'][case];pairs=v['paired_outcomes'];old=[x for x in h['paired_outcomes'] if int(x['seed'])==seed]
 assert len(old)==len(pairs)==200
 key=lambda x:(x['stage'],x['task'],x['initial_index'])
 oo={key(x):x for x in old}
 for x in pairs:
  y=oo[key(x)];assert x['S']==y['S_success'] and all(x[k]==y[k] for k in ('initial_sha256','noise_sha256'))
 history[str(seed)]={k:float(x*100) for k,x in metric(old,'C_success').items()}
 for k,vv in history[str(seed)].items():assert abs(vv-h['seeds'][str(seed)]['C_metrics'][k]['percent'])<1e-10
 paired_counts[str(seed)]={label:dict(C_success_S_fail=sum(x['C'] and not x['S'] for x in rr),C_fail_S_success=sum(not x['C'] and x['S'] for x in rr),episodes=len(rr)) for label,rr in [('all',pairs),('final',[x for x in pairs if x['stage']==9])]}
 ep=json.loads(e['light'][case+'/EPOCHS_C.json']['text'])
 for stage in (7,8,9):
  pp={x['epoch']:x for x in ep if x['stage']==stage};u,w=pp[99],pp[199]
  training.append(dict(seed=seed,stage=stage,BC_epoch100=u['totals']['bc']/u['batches'],BC_epoch200=w['totals']['bc']/w['batches'],expert_epoch100=u['aux_totals']['expert_selected_sum']/u['aux_totals']['count'],expert_epoch200=w['aux_totals']['expert_selected_sum']/w['aux_totals']['count'],lr_epoch101_first=pp[100]['lr_first'],lr_final_after=w['lr_after']))
 for task,x in a['offline'][case+'_C_9'].items():offline.append(dict(seed=seed,task=int(task),**x))
keys=['AUC','FWT','NBT','OldFinal','FinalAll'];suites=c['suites']['GOAL']
oldmean={k:statistics.mean(v[k] for v in history.values()) for k in keys}
negative=c['all_negative_cells'];positive=sum(x['S_minus_C_pp']>0 for v in c['cases'].values() for x in v['cells'].values());ties=sum(x['S_minus_C_pp']==0 for v in c['cases'].values() for x in v['cells'].values())
extra=dict(status='PASS_GOAL_INTERPRETATION_RECOMPUTED',observation=e['observed'],source_comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),historical_comparison_sha256=sha(sf/'PRIMARY_C_COMPARISON.json'),historical_C_means=oldmean,historical_C_by_seed=history,paired_counts=paired_counts,training_proxy=training,offline_final=offline,cells=dict(S_better=positive,tied=ties,C_better=len(negative)),whole_experiment_accepted=False)
(D/'INTERPRETATION_AUDIT.json').write_text(json.dumps(extra,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
observed=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
lines=['# GOAL等优化器预算C与已有完整S：独立验收报告','',f'验收观察：{observed}。仅GOAL seed41/42/43接受；OBJECT继续运行，全六C总体仍PARTIAL。','',
'GOAL三个新C已完成2100轮、426600生产optimizer调用、273自身隔离调用和600正式回合。每seed Task6/7/8/9生产调用10000/27800/37000/67400，与对应已有S逐阶段相等；S未重训或重评。','',
'实际CPU验收通过12个阶段检查点、3份完整恢复状态、9份teacher缓存重放；600回合的初态文件、实际字节、噪声与环境seed逐项配对，所有回合attempt=1。GOAL三次train/evaluate均exit0、GOAL完成标记存在且无GOAL运行进程；审计前后GOAL轻量证据hash精确一致。冻结88科学项和3工程项保留。此次只读CPU诊断0 optimizer、0环境回合；不要求或伪造全六C controller成功。','',
'## 三seed均值','',
'|指标|旧预算C（历史）|等预算新C|已有完整S|S−新C pp|','|---|---:|---:|---:|---:|']
for k in keys:
 x=suites[k];lines.append('|%s|%.4f|%.4f|%.4f|%+.4f|'%(k,oldmean[k],x['C_mean'],x['S_mean'],x['S_minus_C_pp']))
lines += ['', 'AUC按每任务已见阶段平均后再跨任务平均；FWT沿用实验定义，为四个新任务刚学完时的对角成功率，不能解释为未训练任务的零样本迁移。NBT衡量相对各任务初学完成时的平均下降，越低越好。OldFinal为最终阶段旧任务6–8均值；FinalAll为最终四任务均值。', '',
'增加C预算后，GOAL的AUC由82.50升到83.7153，FWT由85.00升到87.0833；最终平均成功率仍为80.00，旧任务最终均值从80.5556降到77.2222。这说明本次增加更新改善了部分新任务拟合，却没有提高最终总体闭环成功率。与新C比较，S的AUC高3.5417 pp、最终成功率高5.00 pp、旧任务最终高6.6667 pp，平均NBT低2.3148 pp。就这三个seed而言，增加C预算没有消除S的平均优势；不能据此估计一般性的“追上概率”。', '',
'旧C对照仅作历史描述。新C同时使用延长后的学习率调度终点，不能将历史差值严格归因于额外100轮本身；也没有在新C第100轮另做闭环评估。', '',
'## seed差异与新旧任务','', '|seed|新C FinalAll|S FinalAll|S−C|ΔAUC|ΔNBT（负值好）|ΔOldFinal|','|---|---:|---:|---:|---:|---:|---:|']
for seed in (41,42,43):
 v=c['cases']['GOAL_'+str(seed)];lines.append('|%d|%.2f|%.2f|%+.2f|%+.4f|%+.4f|%+.4f|'%(seed,v['C_metrics']['FinalAll'],v['S_metrics']['FinalAll'],v['delta_pp']['FinalAll'],v['delta_pp']['AUC'],v['delta_pp']['NBT'],v['delta_pp']['OldFinal']))
lines += ['',
'seed41：新C最终83.75%，S为82.50%，新C略高1.25 pp。最终旧任务平均相等：S在Task8多2次成功，但Task6少2次；最新Task9少1次。S的NBT更差，不应声称所有seed保持都更强。',
'seed42：S最终高3.75 pp，优势来自旧任务（Task6多1次、Task8多4次成功），最新Task9反而比新C少2次成功。',
'seed43：S最终高12.50 pp，Task6/7各多2次、Task8/9各多3次成功；但早期Task7和Task8阶段存在S落后的单元。这体现任务与阶段差异，不是单调全面改善。','',
'## 全部30个阶段任务单元与配对变化','',f'S优于新C {positive} 个、相等 {ties} 个、新C优于S {len(negative)} 个。下表SF表示C成功而S失败，FS表示C失败而S成功；每单元20个相同初态，Δ=5×(FS−SF) pp。','',
'|seed|stage/task|C成功率|S成功率|Δpp|SF|FS|','|---|---|---:|---:|---:|---:|---:|']
for case,v in c['cases'].items():
 for cell,x in v['cells'].items():lines.append('|%s|%s|%.1f|%.1f|%+.1f|%d|%d|'%(case[-2:],cell,x['C_percent'],x['S_percent'],x['S_minus_C_pp'],len(x['C_success_S_fail']),len(x['C_fail_S_success'])))
lines += ['', 'S相对新C的所有负单元：'+ '；'.join('%s %d/%d %+.0f pp'%(x['case'],x['stage'],x['task'],x['S_minus_C_pp']) for x in negative)+'。它们全部保留，不重评或追加实验。', '',
'|seed|最终80回合 SF|最终80回合 FS|净增成功|','|---|---:|---:|---:|']
for seed,x in paired_counts.items():
 p=x['final'];lines.append('|%s|%d|%d|%+d|'%(seed,p['C_success_S_fail'],p['C_fail_S_success'],p['C_fail_S_success']-p['C_success_S_fail']))
lines += ['', '逐回合初态索引、checkpoint哈希与SF/FS清单见PRIMARY_S_COMPARISON.json。新C自身各阶段相对刚学完时的SF/FS见LOCAL_FINAL_AUDIT.json/paired_within_C；跨方法差异与同方法遗忘是两个不同的比较。', '',
'## 损失、梯度与闭环的边界','',
'|seed|stage|第100轮平均BC|第200轮平均BC|第100轮旧样本执行cost|第200轮旧样本执行cost|','|---|---:|---:|---:|---:|---:|']
for x in training:lines.append('|%d|%d|%.6f|%.6f|%.6f|%.6f|'%(x['seed'],x['stage'],x['BC_epoch100'],x['BC_epoch200'],x['expert_epoch100'],x['expert_epoch200']))
lines += ['',
'这里BC是记录的batch标量平均，不是闭环成功概率；连续动作的负对数似然可为负。第101轮学习率仍为正，最终更新结束后才衰减到0，额外预算不是后100轮零学习率空转。训练代理下降也不能推出闭环成功率必然提高，实际最终均值和旧任务结果已显示两者可分离。',
'C保留原损失，不使用S的成功保持缓存。记录的部分梯度诊断中IFA/route项为0，不能把所有追加更新说成每个损失项均产生非零梯度。诊断只覆盖固定训练batch及固定epoch，不能外推成全程梯度恒为0或恒有收益。CPU固定训练窗口的执行cost/route/regret按真实CP重新计算，详见LOCAL_FINAL_AUDIT及RECOMPUTED文件；这些不是独立验证集，不作为测试选优或机制因果证据。','',
'## 成本与适用范围','',
'双方GOAL生产调用各426600、自身隔离各273；C接受更新426600，S接受更新228300，S丢弃的候选也计入物理调用。C无新增训练采集；已有S有120次训练采集，S保持缓存由其自身成功轨迹形成。因此只匹配optimizer调用，不等FLOPs、墙钟、额外交互、每次更新的学习率或可用训练信息。C准入额外autograd36、训练方向诊断144独立记账；本次CPU前向调用与窗口数见LOCAL_FINAL_AUDIT/final_cpu，不计作训练调用。',
'三个seed和评估初态已用于开发，不能宣称新独立确认集或统计显著；S的优势不能只归因于门控一个机制。OBJECT尚未接受，不向另一套件外推。旧机制已停止，旧中断UNKNOWN及OBJECT历史工程诊断仍在原账，不并入本次GOAL预算。', '',
'本报告接受GOAL数据完整性和预先规定的配对比较，结果有正有负；不会因负单元补采、重训、重评、调参或追加seed。OBJECT继续原队列，全部完成后再做六C总验收。']
report='\n'.join(lines)+'\n'
(D/'GOAL_FINAL_REPORT.md').write_text(report,encoding='utf8');(R/'GOAL_FINAL_REPORT.md').write_text(report,encoding='utf8')
receipt=dict(status='ACCEPTED_GOAL_THREE_C_EQUAL_OPTIMIZER_BUDGET',suite='GOAL',seeds=[41,42,43],observation=observed,final_directory=str(D),audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),report_sha256=sha(D/'GOAL_FINAL_REPORT.md'),whole_six_C_status='PARTIAL_OBJECT_PENDING',environment_retries=0,new_S_runs=0)
(R/'GOAL_ACCEPTANCE.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(receipt=receipt,metrics=suites,paired_counts=paired_counts,cells=extra['cells']),ensure_ascii=False,indent=2))
