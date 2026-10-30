"""Explain accepted six-C results using verified existing paired records only."""
from pathlib import Path
from fractions import Fraction as F
import json, hashlib, statistics, sys, datetime
R=Path(__file__).resolve().parent; D=Path(sys.argv[1])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
write=lambda p,x:p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
a=js(D/'LOCAL_FINAL_AUDIT.json'); c=js(D/'PRIMARY_S_COMPARISON.json'); e=js(D/'EVIDENCE.json')
assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['evaluation']['episodes']==1200
assert c['status']=='PASS_NEW_C_EXISTING_S_1200_PAIRED_EPISODES_EQUAL_STAGE_OPTIMIZER_BUDGET'
assert c['source_C_final_sha256']==sha(D/'LOCAL_FINAL_AUDIT.json')
keys=['AUC','FWT','NBT','OldFinal','FinalAll']
def metric(rows,field):
 r={(s,t):F(sum(x[field] for x in rows if (x['stage'],x['task'])==(s,t)),20) for s in range(6,10) for t in range(6,s+1)}
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
history={}; oldmeans={}; paired={}; training=[]; within={}; source_hashes={}; cellcounts={}; historical_initial=[]
binding=js(R/'S_ACCEPTANCE.json')
for suite,ref in binding['sources'].items():
 sf=Path(ref['final_path']); hp=sf/'PRIMARY_C_COMPARISON.json'; h=js(hp)
 assert h['status']=='PASS_ACCEPTED_C_S_600_EPISODE_PAIRING'; source_hashes[str(hp)]=sha(hp)
 if suite=='GOAL':
  for name,digest in h['sources'].items():
   p=Path(name);p=p if p.is_absolute() else sf.parent/p
   assert sha(p)==digest;source_hashes[str(p)]=digest
 else:
  old=R.parent/'20261027_c_goal42_43_object41_42_43/final_readonly_20261027_222119'
  for prefix,folder in [('C',old),('S',sf)]:
   for label,name in [('final_audit','LOCAL_FINAL_AUDIT.json'),('evidence','EVIDENCE.json')]:
    p=folder/name;assert sha(p)==h[prefix+'_'+label+'_sha256'];source_hashes[str(p)]=sha(p)
 history[suite]={}
 for seed in (41,42,43):
  case=f'{suite}_{seed}';v=c['cases'][case];pairs=v['paired_outcomes'];old=[x for x in h['paired_outcomes'] if int(x['seed'])==seed]
  key=lambda x:(x['stage'],x['task'],x['initial_index'])
  oo={key(x):x for x in old};assert len(oo)==len(pairs)==200
  for x in pairs:
   y=oo[key(x)];assert x['S']==y['S_success'] and all(x[k]==y[k] for k in ['initial_sha256','noise_sha256'])
  history[suite][str(seed)]={k:float(x*100) for k,x in metric(old,'C_success').items()}
  for k,value in history[suite][str(seed)].items():assert abs(value-h['seeds'][str(seed)]['C_metrics'][k]['percent'])<1e-10
  paired[case]={name:dict(SF=sum(x['C'] and not x['S'] for x in rr),FS=sum(not x['C'] and x['S'] for x in rr),episodes=len(rr)) for name,rr in [('all',pairs),('final',[x for x in pairs if x['stage']==9]),('old_final',[x for x in pairs if x['stage']==9 and x['task']<9]),('new_final',[x for x in pairs if x['stage']==x['task']==9])]}
  ep=json.loads(e['light'][case+'/EPOCHS_C.json']['text'])
  for stage in (7,8,9):
   pp={x['epoch']:x for x in ep if x['stage']==stage};u,w=pp[99],pp[199]
   assert pp[100]['lr_first']>0 and w['lr_after']==0
   training.append(dict(case=case,stage=stage,BC100=u['totals']['bc']/u['batches'],BC200=w['totals']['bc']/w['batches'],exec100=u['aux_totals']['expert_selected_sum']/u['aux_totals']['count'],exec200=w['aux_totals']['expert_selected_sum']/w['aux_totals']['count'],lr101=pp[100]['lr_first']))
  cr={key(x):x for x in pairs}
  within[case]={}
  for t in [6,7,8]:
   initial=[cr[t,t,i] for i in range(20)];final=[cr[9,t,i] for i in range(20)]
   within[case][str(t)]={method:dict(SF=sum(x[method] and not y[method] for x,y in zip(initial,final)),FS=sum(not x[method] and y[method] for x,y in zip(initial,final))) for method in ['C','S']}
  historical_initial.append(dict(case=case,old_C_task6=sum(x['C_success'] for x in old if x['stage']==6)*5,new_C_task6=v['cells']['6_6']['C_percent']))
 oldmeans[suite]={k:statistics.mean(v[k] for v in history[suite].values()) for k in keys}
 cells=[x for case,v in c['cases'].items() if case.startswith(suite+'_') for x in v['cells'].values()]
 cellcounts[suite]=dict(S_better=sum(x['S_minus_C_pp']>0 for x in cells),tied=sum(x['S_minus_C_pp']==0 for x in cells),C_better=sum(x['S_minus_C_pp']<0 for x in cells))
extra=dict(status='PASS_SIX_C_INTERPRETATION_RECOMPUTED',comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),historical_sources=source_hashes,historical_C_by_seed=history,historical_C_means=oldmeans,historical_task6=historical_initial,paired_counts=paired,within_method_final_vs_acquisition=within,training_proxy=training,cells=cellcounts)
write(D/'INTERPRETATION_AUDIT.json',extra)
observed=datetime.datetime.fromtimestamp(e['observed'],datetime.timezone(datetime.timedelta(hours=8))).isoformat()
lines=['# 两套件六C等优化器预算：最终验收与配对解释','',f'实际CPU验收观察：{observed}。仅六C重训；已有完整S未重训、重评或补采。','',
'六C全部700轮、共4200轮和1012500生产optimizer调用、546自身隔离调用，1200正式回合全部完成。每阶段生产调用及自身隔离与对应S相等。工程调度器controller.exit为0且无本轮运行进程；原调度器根controller.exit保持不存在。实际CPU24个阶段CP、6份完整恢复状态、18份teacher缓存重放及1200回合初态字节/噪声检查通过，已有S的1200回合逐项配对通过。GOAL此前独立接受的文件与报告保留。','',
'## 核心比较','', '|套件|指标|旧C（历史）|等预算新C|已有S|S−新C pp|','|---|---|---:|---:|---:|---:|']
for suite,vals in c['suites'].items():
 for k,x in vals.items():lines.append('|%s|%s|%.4f|%.4f|%.4f|%+.4f|'%(suite,k,oldmeans[suite][k],x['C_mean'],x['S_mean'],x['S_minus_C_pp']))
lines+=['','AUC先对每个任务的已见阶段取均值，再跨四任务平均；FWT沿用实验中的刚学完任务对角成功率定义，不是未训练任务的零样本迁移。NBT越低越好，可为负：负值表示后续阶段相对刚学完时有净改善；它不保证每个任务都不遗忘。OldFinal是最终旧任务6–8均值；FinalAll是最终四任务均值。','',
'GOAL：追加C预算后AUC和FWT较旧C提高，但最终总体成功率仍为80%，旧任务最终均值反而下降。S最终85%，平均领先新C 5 pp；seed41新C略好，seed42/43由S领先。额外优化没有消除这三个seed的S均值优势。','',
'OBJECT：额外C预算也没有使新C达到S的最终均值。S的收益在seed41明显集中于最终Task6恢复和Task9；seed42在旧Task6上获益，但新Task7刚学完时低于C，最终Task8也低于C；seed43在Task7/8获益，却仍未解决最终Task6仅30%的问题。不能将均值优势写成全面胜出或门控单机制的因果结论。','',
'旧C比较是历史描述，不能把差值严格归因于增加100轮：本轮同时延长了学习率衰减终点；历史环境/硬件轨迹也有差异（旧OBJECT C为RTX3090，本轮新C与S为RTX4080 SUPER；OBJECT41历史Task6为45%，本轮与S匹配的Task6为35%）。本轮新C与已有S的官方初始化、实际完整初始状态、硬件记录及Task6 tensor一致已逐项验真；这不意味着旧C也逐位相同。','',
'## 各seed与最终新旧任务','', '|case|新C FinalAll|S FinalAll|ΔFinalAll|ΔAUC|ΔNBT|ΔOldFinal|最终Task9 C/S|','|---|---:|---:|---:|---:|---:|---:|---|']
for case,v in c['cases'].items():
 x=v['cells']['9_9'];lines.append('|%s|%.2f|%.2f|%+.2f|%+.4f|%+.4f|%+.4f|%.1f / %.1f|'%(case,v['C_metrics']['FinalAll'],v['S_metrics']['FinalAll'],v['delta_pp']['FinalAll'],v['delta_pp']['AUC'],v['delta_pp']['NBT'],v['delta_pp']['OldFinal'],x['C_percent'],x['S_percent']))
lines+=['','GOAL41：最终旧任务均值持平，S在Task8的收益被Task6损失抵消，最新Task9也少一次成功。GOAL42：S主要改善旧任务6/8，最新Task9较C少两次成功。GOAL43：最终四个任务均提高，但早期7/8阶段仍有负单元。','',
'OBJECT41：最终Task6新C25%、S70%（+45 pp），Task9 85%→100%，Task7 95%→90%是代价。OBJECT42：最终Task6 45%→65%，Task8 100%→90%；刚学完Task7新C95%、S60%（−35 pp），必须与后续保持收益同时报告。OBJECT43：最终Task6两者30%，Task7/8分别由95/75升至100/90，Task9同为100%。','',
'## 全部单元与配对SF/FS','',
'下表SF为C成功而S失败，FS为C失败而S成功；每单元相同20个初态，Δ=5×(FS−SF) pp。相同成功率仍可能有相互抵消的轨迹差异。','', '|case|stage/task|C%|S%|Δpp|SF|FS|','|---|---|---:|---:|---:|---:|---:|']
for case,v in c['cases'].items():
 for cell,x in v['cells'].items():lines.append('|%s|%s|%.1f|%.1f|%+.1f|%d|%d|'%(case,cell,x['C_percent'],x['S_percent'],x['S_minus_C_pp'],len(x['C_success_S_fail']),len(x['C_fail_S_success'])))
lines+=['','所有S负单元（不剔除、不补评）：'+'；'.join('%s %d/%d %+.0f pp'%(x['case'],x['stage'],x['task'],x['S_minus_C_pp']) for x in c['all_negative_cells'])+'。','', '|case|最终80回合SF|FS|净增成功|旧任务净增|最新任务净增|','|---|---:|---:|---:|---:|---:|']
for case,x in paired.items():
 f=x['final'];o=x['old_final'];n=x['new_final'];lines.append('|%s|%d|%d|%+d|%+d|%+d|'%(case,f['SF'],f['FS'],f['FS']-f['SF'],o['FS']-o['SF'],n['FS']-n['SF']))
lines+=['','同方法遗忘另行核算：以下SF/FS比较该方法任务刚学完与最终阶段，不能与上面的跨方法SF/FS混为一谈。','', '|case|旧task|C SF/FS|S SF/FS|','|---|---:|---|---|']
for case,ts in within.items():
 for t,x in ts.items():lines.append('|%s|%s|%d / %d|%d / %d|'%(case,t,x['C']['SF'],x['C']['FS'],x['S']['SF'],x['S']['FS']))
lines+=['','## 训练损失和离线诊断的边界','', '|case|stage|BC第100轮|BC第200轮|旧样本执行cost第100轮|第200轮|','|---|---:|---:|---:|---:|---:|']
for x in training:lines.append('|%s|%d|%.6f|%.6f|%.6f|%.6f|'%(x['case'],x['stage'],x['BC100'],x['BC200'],x['exec100'],x['exec200']))
lines+=['','额外100轮实际有正学习率，第101轮非零、最后更新后才归零。本轮18个扩展阶段的第200轮BC和旧样本执行cost均低于第100轮。BC为batch标量平均，连续动作负对数似然可为负；训练损失继续降低不能保证闭环成功率提高。没有对新C第100轮另做闭环评估，不能将其后闭环变化直接测定或归因于追加轮数。原C目标没有加入S的成功保持项，额外更新仍在优化原C目标。方向诊断仅覆盖固定batch/epoch，局部为零的项不能外推为全程零梯度。','',
'独立CPU重放的旧窗口执行cost、expert regret、margin路由和teacher距离见LOCAL_FINAL_AUDIT/offline与RECOMPUTED文件。它们来自训练窗口，不是独立验证集，不用于重新选择检查点或调整超参数。完整闭环矩阵及负单元优先于训练代理解释；本实验比较完整方法，不隔离支持度、保持损失和门控各自贡献。','',
'## 调用预算与适用范围','',
'GOAL双方生产调用各426600，OBJECT各585900；双方自身隔离各546。新C保留1012500次生产更新，S保留544800次更新、其余是被丢弃候选但照样计费。C无训练采集，已有S额外240次训练采集形成自身成功缓存；每次更新LR、前向/探针次数、FLOPs、墙钟和可用信息并不相等。OBJECT历史3914次工程验证、18额外autograd及旧中断UNKNOWN保持原账，不能隐去或算作零。','',
'新C另有72准入autograd、288训练方向autograd；最终CPU仅前向重放，0 optimizer、0环境回合。传输包装仅在每个CPU case完成后输出空白并flush，六个审计脚本字节不变，以避免整段无输出触发SSH空闲超时；不修改科学冻结源。','',
'只有两个套件各三个已用于开发的seed和既有初态，不能给出统计显著、普适追上概率或独立确认集结论。既有S不重跑、旧机制不恢复；所有正负结果保留，不因负单元追加训练、评估、种子或扫参。']
lines+=['','## 实际离线重放与闭环的对应','',
'最终阶段CPU重放中，GOAL三个C的全部旧任务窗口均为expert regret=0、margin route=0，然而其最终旧任务均值仍只有77.2222%。这说明在所记录旧训练窗口上选到最低cost专家，不等价于闭环轨迹保持成功；它不能证明所有新访问状态上的路由都已解决。',
'OBJECT41最终Task6有764个旧窗口，仅5个窗口没有选择最低cost专家，平均执行cost为0.00138972、regret为0.00000495538，但20个正式初态仅5个成功（25%）。OBJECT43最终Task6为761个窗口中12个非最低cost选择，平均regret为0.000262928，闭环为30%。这些数据支持“训练窗口代理与长轨迹成功存在缺口”的描述，不足以确定缺口由哪一机制造成。',
'OBJECT41 Task9第200轮首个诊断batch中，BC梯度L2为1351.827，已加权执行项为0.0404353，IFA及route项在该batch为零。该尺度比较只描述当前参数化和所选batch；未计算跨项梯度夹角，也不能把局部零梯度解释为全程没有作用，更不能仅据范数推断闭环失败因果。']
lines+=['','## 已记录耗时','', '|case|训练epoch合计分钟|正式vector合计分钟|','|---|---:|---:|']
for case,x in a['costs'].items():
 lines.append('|%s|%.2f|%.2f|'%(case,x['epoch_seconds']/60,x['evaluation_vector_seconds']/60))
lines+=['','训练列是epoch内部计时之和，逐轮保存、teacher准备、调度和最终CPU验收另计；正式列是实际vector计时之和，不能冒充完整墙钟时间。跨seed与历史运行资源不同，不据此作严格速度归因。']
report='\n'.join(lines)+'\n';(D/'FINAL_REPORT.md').write_text(report,encoding='utf8');(R/'FINAL_REPORT.md').write_text(report,encoding='utf8')
receipt=dict(status='ACCEPTED_SIX_C_EQUAL_OPTIMIZER_BUDGET',observation=observed,final_directory=str(D),audit_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),comparison_sha256=sha(D/'PRIMARY_S_COMPARISON.json'),report_sha256=sha(D/'FINAL_REPORT.md'),training_epochs=4200,production_calls=1012500,isolated_calls=546,formal_episodes=1200,new_S_runs=0,environment_retries=0,completion_source='engineering_suite_order_20261030_v1/controller.exit')
write(R/'ACCEPTANCE.json',receipt)
print(json.dumps(dict(receipt=receipt,means=c['suites'],historical_C_means=oldmeans,paired_counts=paired,cells=cellcounts),ensure_ascii=False,indent=2))
