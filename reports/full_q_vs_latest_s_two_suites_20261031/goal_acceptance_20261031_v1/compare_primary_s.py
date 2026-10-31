"""Compare newly accepted Q with existing accepted S; no models/runs executed."""
from pathlib import Path
from fractions import Fraction as F
import sys,json,hashlib,statistics
R=Path(__file__).resolve().parent.parent;D=Path(sys.argv[1])
js=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=js(D/'LOCAL_FINAL_AUDIT.json');e=js(D/'EVIDENCE.json');binding=js(R/'S_ACCEPTANCE.json')
assert a['status']=='PASS_COMPLETE_GOAL_LOCAL_ACCEPTANCE' and a['formal_episodes']==600
assert a['evidence_sha256']==sha(D/'EVIDENCE.json')
assert a['main_updates']==228300 and a['isolated_optimizer_updates']==273
def metric(r):
 return dict(AUC=sum(sum(r[s,t] for s in range(t,10))/(10-t) for t in range(6,10))/4,FWT=sum(r[t,t] for t in range(6,10))/4,NBT=sum(sum(r[t,t]-r[s,t] for s in range(t+1,10))/(9-t) for t in range(6,9))/3,OldFinal=sum(r[9,t] for t in range(6,9))/3,FinalAll=sum(r[9,t] for t in range(6,10))/4)
def table(rows):
 return metric({(s,t):F(sum(x['success'] for x in rows if (x['stage'],x['task'])==(s,t)),20) for s in range(6,10) for t in range(6,s+1)})
results={};negative=[];sources={};suites={}
for suite,ref in [('GOAL',binding['sources']['GOAL'])]:
 folder=Path(ref['final_path']);sa=js(folder/'LOCAL_FINAL_AUDIT.json');se=js(folder/'EVIDENCE.json')
 assert sha(folder/'LOCAL_FINAL_AUDIT.json')==ref['audit_sha256'] and sha(folder/'EVIDENCE.json')==ref['evidence_sha256']==sa['evidence_sha256']
 assert sa['status']=='PASS_COMPLETE_LOCAL_ACCEPTANCE' and sa['formal_episodes']==600
 sources[suite]=ref
 for seed in (41,42,43):
  case=suite+'_'+str(seed);cl=e['light'];sl=se['light'];cp=case+'/';sp='seed%d/'%seed
  cr=json.loads(cl[cp+'OUTCOMES.json']['text']);sr=json.loads(sl[sp+'OUTCOMES.json']['text']);assert len(cr)==len(sr)==200
  key=lambda x:(x['stage'],x['task'],x['initial_index'])
  cc={key(x):x for x in cr};ss={key(x):x for x in sr};assert cc.keys()==ss.keys() and len(cc)==200
  ci=json.loads(cl[cp+'INITIAL_Q.json']['text']);si=json.loads(sl[sp+'INITIAL_S.json']['text'])
  for k in ['tensors','rng','hardware','official_cp_sha256','parent_initial_sha256']:assert ci[k]==si[k],(case,k)
  assert json.loads(cl[cp+'TASK6_PARITY_Q.json']['text'])['same_seed_new_arm_comparator']=='ACCEPTED_S_ALL_TENSORS_EXACT'
  ch=json.loads(cl[cp+'EPOCHS_Q.json']['text']);sh=json.loads(sl[sp+'EPOCHS_S.json']['text'])
  budgets={}
  for stage in range(6,10):
   ccall=sum(x['batches'] for x in ch if x['stage']==stage)
   scall=sum(x['batches']*(1 if stage==6 else 2) for x in sh if x['stage']==stage)
   ca=json.loads(cl[cp+'ADMISSION_Q_%d.json'%stage]['text'])['optimizer_calls'];sa_calls=json.loads(sl[sp+'ADMISSION_S_%d.json'%stage]['text'])['optimizer_calls']
   assert ccall*(1 if stage==6 else 2)==scall and ca==sa_calls==(4 if stage==6 else 29)
   budgets[stage]=dict(Q_production=ccall,S_production=scall,Q_isolated=ca,S_isolated=sa_calls)
  pair=[]
  for k in sorted(cc):
   c,s=cc[k],ss[k]
   for field in ['initial_file_sha256','initial_sha256','noise_sha256','env_seed','attempt']:assert c[field]==s[field],(case,k,field)
   pair.append(dict(stage=k[0],task=k[1],initial_index=k[2],Q=c['success'],S=s['success'],initial_sha256=c['initial_sha256'],noise_sha256=c['noise_sha256'],Q_cp_sha256=c['cp_sha256'],S_cp_sha256=s['cp_sha256']))
  cm=table(cr);sm=table(sr)
  for k in cm:assert str(cm[k])==a['cases'][case]['metrics'][k]['fraction'] and str(sm[k])==sa['seeds'][str(seed)]['metrics'][k]['fraction']
  cells={}
  for stage in range(6,10):
   for task in range(6,stage+1):
    rr=[x for x in pair if (x['stage'],x['task'])==(stage,task)];assert len(rr)==20
    row=dict(Q_percent=5*sum(x['Q'] for x in rr),S_percent=5*sum(x['S'] for x in rr),Q_success_S_fail=[x['initial_index'] for x in rr if x['Q'] and not x['S']],Q_fail_S_success=[x['initial_index'] for x in rr if not x['Q'] and x['S']])
    row['S_minus_Q_pp']=row['S_percent']-row['Q_percent'];cells['%d_%d'%(stage,task)]=row
    if row['S_minus_Q_pp']<0:negative.append(dict(case=case,stage=stage,task=task,**row))
  results[case]=dict(Q_metrics={k:float(v*100) for k,v in cm.items()},S_metrics={k:float(v*100) for k,v in sm.items()},delta_pp={k:float((sm[k]-cm[k])*100) for k in cm},stage_budgets=budgets,cells=cells,paired_outcomes=pair)
 keys=['AUC','FWT','NBT','OldFinal','FinalAll']
 suites[suite]={k:dict(Q_mean=statistics.mean(results[suite+'_'+str(seed)]['Q_metrics'][k] for seed in (41,42,43)),S_mean=statistics.mean(results[suite+'_'+str(seed)]['S_metrics'][k] for seed in (41,42,43)),S_minus_Q_pp=statistics.mean(results[suite+'_'+str(seed)]['delta_pp'][k] for seed in (41,42,43))) for k in keys}
out=dict(status='PASS_GOAL_Q_EXISTING_S_600_PAIRED_EPISODES_EQUAL_ACCEPTED_UPDATES',source_Q_final_sha256=sha(D/'LOCAL_FINAL_AUDIT.json'),source_S=sources,paired_episodes=600,suites=suites,cases=results,all_negative_cells=negative,limits=['Only Q newly run; S accepted results reused','Equal accepted updates and collection counts; Q single candidate uses fewer physical optimizer calls; unequal FLOPs/wall time and own cache contents','Both methods 100 epochs/task, original accepted-update scheduler and warmup500','S historical engineering diagnostics remain separate; Q startup v1 zero optimizer and environments independently audited','Three development seeds, no significance or single-mechanism causal claim'])
(D/'PRIMARY_S_COMPARISON.json').write_text(json.dumps(out,indent=2)+'\n')
lines=['# 新Q与已有完整S：等接受更新、关闭门控的配对结果','', '仅GOAL套件三seed，600回合初态/noise/env_seed逐项配对；S未重训或重评。数值差值均为S−新Q，NBT越低越好。','']
for suite,rows in suites.items():
 lines+=['## '+suite,'','|指标|新Q均值|已有S均值|S−Q百分点|','|---|---:|---:|---:|']
 lines+=['|%s|%.4f|%.4f|%+.4f|'%(k,v['Q_mean'],v['S_mean'],v['S_minus_Q_pp']) for k,v in rows.items()]
 lines+=['','|seed|ΔAUC|ΔFWT|ΔNBT|ΔOldFinal|ΔFinalAll|','|---|---:|---:|---:|---:|---:|']
 for seed in (41,42,43):lines.append('|%d|'%seed+'|'.join('%+.4f'%results[suite+'_'+str(seed)]['delta_pp'][k] for k in keys)+'|')
 lines+=['']
lines+=['## 全部S相对新Q的负单元','','|case|阶段/任务|新Q|已有S|Δ百分点|Q成功S失败|Q失败S成功|','|---|---|---:|---:|---:|---:|---:|']
lines+=['|%s|%d/%d|%.1f|%.1f|%+.1f|%d|%d|'%(x['case'],x['stage'],x['task'],x['Q_percent'],x['S_percent'],x['S_minus_Q_pp'],len(x['Q_success_S_fail']),len(x['Q_fail_S_success'])) for x in negative]
lines+=['','上述仅自动汇总；最终报告须结合新旧任务、SF/FS、训练/离线代理及所有负单元人工解释。两者各120次训练采集、同配额但各自产生缓存，不是固定缓存的单步因果实验。Q单候选不匹配S双候选物理optimizer/FLOPs/墙钟；历史工程调用另记。']
(D/'PRIMARY_S_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps(dict(status=out['status'],suites=suites),indent=2))
