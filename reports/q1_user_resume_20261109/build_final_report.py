"""Recompute six metrics and all paired cells from accepted, immutable outcomes."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,statistics,datetime,csv
R=Path(__file__).resolve().parent
J=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
sources={'OBJECT':R/'final_readonly_20261110_121750','GOAL':R/'final_goal_readonly_20261110_121810'}
prior={'OBJECT':R.parent/'20261108_object_s_q1_ablation','GOAL':R.parent/'20261108_goal_s_q1_ablation'}
keys=['FWT','NBT','AUC','OldFinal','NewFinal','FinalAll']
def metrics(rows):
 m={(s,t):F(sum(x['success'] for x in rows if (x['stage'],x['task'])==(s,t)),20) for s in range(6,10) for t in range(6,s+1)}
 assert len(rows)==200
 d=dict(AUC=sum(sum(m[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(m[t,t] for t in range(6,10))/4,NBT=sum(sum(m[t,t]-m[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(m[9,t] for t in range(6,9))/3,NewFinal=m[9,9],FinalAll=sum(m[9,t] for t in range(6,10))/4)
 return {k:float(v*100) for k,v in d.items()}
def light(v,n):return json.loads(v['light'][n]['text'])
out={'status':'PASS_ACCEPTED_TWO_SUITES_DESCRIPTIVE_COMPARISON','suites':{},'sources':{},'all_cells':[],'all_negative_S_cells':[]}
csvrows=[]
for suite,d in sources.items():
 a=J(d/'LOCAL_FINAL_AUDIT.json');v=J(d/'EVIDENCE.json');c=J(d/'PRIMARY_S_COMPARISON.json')
 assert a['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and a['evidence_sha256']==sha(d/'EVIDENCE.json')
 assert c['source_Variant_final_sha256']==sha(d/'LOCAL_FINAL_AUDIT.json') and c['paired_episodes']==600
 binding=J(R.parent/'20261109_support_calibration_rerun'/suite/'S_ACCEPTANCE.json')['sources'][suite]
 sd=Path(binding['final_path']);sv=J(sd/'EVIDENCE.json')
 assert sha(sd/'EVIDENCE.json')==binding['evidence_sha256'] and sha(sd/'LOCAL_FINAL_AUDIT.json')==binding['audit_sha256']
 pa=J(prior[suite]/'ACCEPTANCE.json');pd=prior[suite]/pa['final_directory'];pv=J(pd/'EVIDENCE.json')
 assert all(sha(pd/n)==h for n,h in pa['sources'].items())
 out['sources'][suite]={'current_final':str(d),'current_audit_sha256':sha(d/'LOCAL_FINAL_AUDIT.json'),'baseline':binding,'prior_final':str(pd),'prior_acceptance_sha256':sha(prior[suite]/'ACCEPTANCE.json')}
 records={};pairs=[]
 for seed in [41,42,43]:
  cp=f'Q1_{seed}/';sp=f'seed{seed}/';cr=light(v,cp+'OUTCOMES.json');sr=light(sv,sp+'OUTCOMES.json');pr=light(pv,cp+'OUTCOMES.json')
  mm={'q1_rerun':metrics(cr),'S':metrics(sr),'q1_previous':metrics(pr)}
  for k in ['AUC','FWT','NBT','OldFinal','FinalAll']:
   assert abs(mm['q1_rerun'][k]-c['cases'][f'Q1_{seed}']['Variant_metrics'][k])<1e-9
   assert abs(mm['S'][k]-c['cases'][f'Q1_{seed}']['S_metrics'][k])<1e-9
  for method,m in mm.items():csvrows.append(dict(suite=suite,seed=seed,method=method,**m))
  for p in c['cases'][f'Q1_{seed}']['paired_outcomes']:pairs.append(dict(seed=seed,**p))
  for cell,x in c['cases'][f'Q1_{seed}']['cells'].items():
   stage,task=map(int,cell.split('_'));row=dict(suite=suite,seed=seed,stage=stage,task=task,**x);out['all_cells'].append(row)
   if x['S_minus_Variant_pp']<0:out['all_negative_S_cells'].append(row)
  proxy={}
  for method,e,p in [('q1_rerun',v,cp),('S',sv,sp)]:
   tr=light(e,p+'TRAIN_S.json');gr=light(e,p+'GRADIENTS_S.json')
   proxy[method]={'success_trajectories':[light(e,p+f'MEMORY_S_{t}.json')['collection_successes'] for t in range(6,10)],'cache_windows':[light(e,p+f'MEMORY_S_{t}.json')['count'] for t in range(6,10)],'Q_accepts':tr['Q_accepts'],'gate_blocks':tr['gate_blocks'],'Q_accept_percent':100*tr['Q_accepts']/tr['gate_blocks'],'gradient_records':len(gr),'retain_vs_bc_negative':sum(x['retain_vs_bc']['cosine']<0 for x in gr),'retain_vs_exec_negative':sum(x['retain_vs_exec']['cosine']<0 for x in gr),'epoch_seconds':tr['summed_epoch_seconds']}
  records[str(seed)]={'metrics':mm,'S_minus_q1_rerun':{k:mm['S'][k]-mm['q1_rerun'][k] for k in keys},'proxy':proxy,'hardware':c['cases'][f'Q1_{seed}']['hardware'],'task6_success':{'S':sum(x['success'] for x in sr if x['stage']==6),'q1_rerun':sum(x['success'] for x in cr if x['stage']==6)}}
 summary={method:{k:{'mean':statistics.mean(records[str(s)]['metrics'][method][k] for s in [41,42,43]),'sample_sd':statistics.stdev(records[str(s)]['metrics'][method][k] for s in [41,42,43])} for k in keys} for method in ['S','q1_rerun','q1_previous']}
 cells=[x for x in out['all_cells'] if x['suite']==suite]
 wins={'S_wins':sum(x['S_minus_Variant_pp']>0 for x in cells),'ties':sum(x['S_minus_Variant_pp']==0 for x in cells),'q1_wins':sum(x['S_minus_Variant_pp']<0 for x in cells)}
 def paired(rows):return {'episodes':len(rows),'SF':sum(x['S'] and not x['Variant'] for x in rows),'FS':sum(x['Variant'] and not x['S'] for x in rows),'both_success':sum(x['S'] and x['Variant'] for x in rows),'both_fail':sum(not x['S'] and not x['Variant'] for x in rows)}
 out['suites'][suite]={'cases':records,'summary':summary,'cell_counts':wins,'all_stage_pairs':paired(pairs),'final_stage_pairs':paired([x for x in pairs if x['stage']==9]),'mean_delta_S_minus_q1':{k:summary['S'][k]['mean']-summary['q1_rerun'][k]['mean'] for k in keys}}
out['limits']=['Current q1 rerun and previous q1 round are both retained; same historical S is reused, not independent repeated baseline runs.','Three development seeds only; no significance claim or causal effect isolated from execution/cache differences.','GOAL seed41 has effective support weights one throughout in both S and q1, so its differences cannot be caused by non-unit support weights.','OBJECT recovery: 540 isolated optimizer calls, original discarded recorded540 plus unrecorded UNKNOWN; no completed epoch replayed.','Equal nominal per-stage updates and training collection counts do not ensure equal cache contents, FLOPs or wall time.','Pretraining provenance and repeated use of development evaluation states remain unresolved limits.']
(R/'FULL_COMPARISON.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
with (R/'FULL_METRICS.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['suite','seed','method']+keys);w.writeheader();w.writerows(csvrows)
lines=['# 固定q=1重复运行：OBJECT与GOAL最终验收和支持度校准比较','',
'本轮六run已通过实际远程CPU核验与本地独立复算。已有完整S和前次q=1均直接读取已接受结果，没有重训或重评。以下S−q1均以本轮q1为对照，单位为百分点；NBT越低越好，其余指标越高越好。',
'','## 三个seed的完整指标','', '|套件|seed|方案|FWT↑|NBT↓|AUC↑|OldFinal↑|NewFinal↑|FinalAll↑|','|---|---:|---|---:|---:|---:|---:|---:|---:|']
names={'S':'完整S','q1_rerun':'本轮固定q=1','q1_previous':'前次固定q=1'}
for suite,z in out['suites'].items():
 for seed in ['41','42','43']:
  for method in ['S','q1_rerun']:
   m=z['cases'][seed]['metrics'][method];lines.append(f'|{suite}|{seed}|{names[method]}|'+'|'.join(f'{m[k]:.2f}' for k in keys)+'|')
lines+=['','## 三seed均值±样本标准差','', '|套件|方案|FWT↑|NBT↓|AUC↑|OldFinal↑|NewFinal↑|FinalAll↑|','|---|---|---:|---:|---:|---:|---:|---:|']
for suite,z in out['suites'].items():
 for method in ['S','q1_rerun','q1_previous']:
  lines.append(f'|{suite}|{names[method]}|'+'|'.join(f"{z['summary'][method][k]['mean']:.2f}±{z['summary'][method][k]['sample_sd']:.2f}" for k in keys)+'|')
lines+=['','## 本轮支持度校准是否有收益','',
'本轮完整S在两个套件的平均AUC、最终总体成功率和最终旧任务成功率均高于固定q=1；OBJECT三个seed的最终总体成功率均更高，GOAL为两个seed更高、一个更低。这个结果支持在本轮完整系统中保留支持度校准，但不能将差值全部解释为校准的独立因果收益。',
'','OBJECT的重复稳定性尚不足：前次q=1平均AUC和最终成功率高于同一份完整S，本轮则低于S，方向发生反转。GOAL两轮平均方向均有利于S，但本轮差距明显缩小。不能只保留本轮有利结果，或把同一份S重复配对视作两次独立基线实验。',
'','GOAL seed41中，实际用于后续保持的Task6–8成功轨迹数在两方案都是10/10，因而保持权重均为1；Task9缓存不再参与后续任务。该seed结果仍有差异，说明执行路径和在线缓存生成的差别不能忽略。同型号GPU、相同初始化或Task6模型tensor一致，均不足以保证整个训练与闭环逐位一致。',
'','NBT衡量相对于首次学成时的后续变化，而非最终旧任务水平；应结合FWT与OldFinal解释。NewFinal为最终阶段新任务Task9的成功率，不等于跨四阶段平均FWT。']
for suite,z in out['suites'].items():
 delta=z['mean_delta_S_minus_q1'];p=z['all_stage_pairs'];f=z['final_stage_pairs'];w=z['cell_counts']
 lines += ['',f"{suite}：S−q1的AUC {delta['AUC']:+.4f}、FWT {delta['FWT']:+.4f}、NBT {delta['NBT']:+.4f}、OldFinal {delta['OldFinal']:+.4f}、NewFinal {delta['NewFinal']:+.4f}、FinalAll {delta['FinalAll']:+.4f} pp。30个阶段/任务单元中S胜/平/负={w['S_wins']}/{w['ties']}/{w['q1_wins']}；全部600回合SF/FS={p['SF']}/{p['FS']}，最终阶段240回合SF/FS={f['SF']}/{f['FS']}。SF=S成功而q1失败，FS反向；这些逐回合计数具有阶段/任务/seed相关性，不作为独立重复样本的显著性检验。"]
lines+=['','## 成功支持度与训练代理','', '|套件|seed|方案|Task6/7/8/9成功数（每项/10）|保持候选接受块/全部块|接受比例|保持与BC负余弦探针数|','|---|---:|---|---|---|---:|---|']
for suite,z in out['suites'].items():
 for seed,r in z['cases'].items():
  for method,p in r['proxy'].items():lines.append(f"|{suite}|{seed}|{names[method]}|"+'/'.join(map(str,p['success_trajectories']))+f"|{p['Q_accepts']}/{p['gate_blocks']}|{p['Q_accept_percent']:.2f}%|{p['retain_vs_bc_negative']}/{p['gradient_records']}|")
lines+=['','OBJECT首个保持阶段的既有S支持度为0.2/0.2/0.6，固定q=1实际提高了低支持历史的保持权重；GOAL的非单位权重更少。后续缓存也随训练轨迹改变，不能只凭成功条数相同就认定缓存内容相同。门控接受比例与梯度余弦是训练代理，不能替代闭环收益，也不能把代理变化直接当作成绩差异的已证实原因。',
'','## 全部阶段/任务单元','', '|套件|seed|阶段/任务|完整S|本轮q1|S−q1|SF|FS|','|---|---:|---|---:|---:|---:|---:|---:|']
for x in out['all_cells']:lines.append(f"|{x['suite']}|{x['seed']}|{x['stage']}/{x['task']}|{x['S_percent']:.1f}|{x['Variant_percent']:.1f}|{x['S_minus_Variant_pp']:+.1f}|{len(x['Variant_fail_S_success'])}|{len(x['Variant_success_S_fail'])}|")
lines+=['','所有S相对q1的负单元均包含在上表；完整逐回合配对身份、初态/noise/env_seed检查见两个最终目录的PRIMARY_S_COMPARISON.json。前次q1逐seed完整六指标同时保存在FULL_METRICS.csv，未覆盖旧报告。',
'','## 验收和成本','',
'本轮实际验收24个checkpoint、6份完整恢复状态、24份成功缓存teacher重放和1200回合初态/噪声；同时复算47400门控块及全部预算。六run共2400完整轮、544800接受更新、1012500名义生产物理optimizer调用、546自身隔离调用、246准入autograd和216方向autograd、240训练采集与1200正式回合。CPU验收新增optimizer=0、环境回合=0。',
'','OBJECT41/42扩容重启后经过授权精确恢复；已保存完整轮未重跑，原部分块前缀保持。恢复诊断540次optimizer单列，原被放弃候选已记540次加未记账部分UNKNOWN，因此实际总历史物理调用不是精确的1012500。旧S与前次q1的历史工程成本保留原账。',
'','OBJECT全600正式自然完成后才启动GOAL。最终成功源为工程总controller.exit0、OBJECT工程出口0及GOAL原出口0；旧全局与旧OBJECT出口仍不存在，未伪造成功标记。本轮科学148项和恢复工程7项均保持。',
'','三seed来自已反复使用的开发评估状态；前端预训练来源仍未完全闭合。验收说明执行和记录满足冻结合同，不把该比较称为独立留出确认、全局无泄露或严格硬件受控的单因素因果实验。',
'','## 后续实验','',
'本报告和接受证据发布后，启动已授权的MLR+IFA+ALIGN六run。采用原Random示范成员，仅加入ALIGN；不包含RA、额外执行监督E、成功保持R、支持度q或门控。按OBJECT三训练→600正式→GOAL三训练→600正式，最多两个训练并行。已有所有对照只读复用。']
(R/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
accept={'status':'PASS_COMPLETE_TWO_SUITES_Q1_RERUN_ACCEPTANCE','observed':datetime.datetime.now().astimezone().isoformat(),'seeds':[41,42,43],'suites':out['sources'],'checkpoints':24,'recoveries':6,'teacher_caches':24,'formal_episodes':1200,'training_collection_episodes':240,'gate_blocks':47400,'final_optimizer_updates':0,'final_environment_calls':0,'full_comparison_sha256':sha(R/'FULL_COMPARISON.json'),'report_sha256':sha(R/'FINAL_REPORT.md'),'limits':out['limits']}
(R/'ACCEPTANCE.json').write_text(json.dumps(accept,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps({s:{'summary':z['summary'],'delta':z['mean_delta_S_minus_q1'],'pairs':z['all_stage_pairs'],'cells':z['cell_counts']} for s,z in out['suites'].items()},ensure_ascii=False,indent=2))
